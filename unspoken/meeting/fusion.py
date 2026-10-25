"""Fusion: audio.segment + video.speakers -> meeting.utterance.

Who said it: the voice label of the segment is matched to the seat whose face was speaking (ASD
probability) during the segment; the name comes from a check-in ("Hi, I'm Priya", "我是…", or chat `here`)
and is learned once as name <-> voice <-> seat. The room screen's zone talking while no seat does means
the words came from media (a video or a remote call). Faces with no check-in are guests. Everything
learned here is session-only and wiped when the meeting ends.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from ..core.events import Event
from .base import Agent, MeetingState, Person, new_id
from .contracts import MT, SegmentData, Speaker, SpeakersData, UtteranceData
from .reflex import checkin_name

SPEAKING_MIN = 0.30       # mean ASD probability over the segment needed to pin a seat
MEDIA_MIN = 0.50          # share of frames with the screen zone talking to call it media
PAD_S = 0.35              # frames this close to the segment edges still count
MEDIA_VOICE_PREFIX = "voice-screen"
CHECKIN_MIN_CONF = 0.70   # conservative evidence floor, not calibrated identity verification


@dataclass
class _Frame:
    t: float
    seats: dict[str, float]       # seat -> speaking probability
    tracks: dict[str, str | None]  # track_id -> seat
    screen_talking: bool


class FusionAgent(Agent):
    name = "fusion"

    def __init__(self, bus, state: MeetingState, settings, config) -> None:
        super().__init__(bus, state, settings, config)
        self.frames: deque[_Frame] = deque(maxlen=600)   # ~2 minutes at 5 Hz
        self.pending_chat: list[str] = []   # `here` candidates for the next non-media turn only
        self.absent_after = float(config["meeting"].get("absent_after_seconds", 3.0))

    def start(self) -> None:
        self.subscribe(MT.SPEAKERS, self._on_speakers)
        self.subscribe(MT.SEGMENT, self._on_segment)
        self.subscribe(MT.CHAT, self._on_chat)
        self.subscribe(MT.STATUS, self._on_status)

    # ------------------------------------------------------------------ roster
    def person_by_voice(self, voice: str | None) -> Person | None:
        return next((p for p in self.state.people.values() if voice and p.voice == voice), None)

    def person_by_seat(self, seat: str | None) -> Person | None:
        return next((p for p in self.state.people.values() if seat and p.seat == seat), None)

    def person_by_name(self, name: str) -> Person | None:
        return next((p for p in self.state.people.values() if p.name and p.name.lower() == name.lower()), None)

    def _get_or_create(self, voice: str | None, seat: str | None, t: float) -> Person:
        p = self.person_by_voice(voice) or self.person_by_seat(seat)
        if p is None:
            p = Person(key=new_id("p"), joined_at=t)
            self.state.people[p.key] = p
        if voice and not p.voice:
            p.voice = voice
        if seat and not p.seat:
            p.seat = seat
        elif seat and p.seat != seat and self.person_by_seat(seat) is None:
            p.seat = seat                   # they moved
        p.present, p.last_seen = True, t
        return p

    def _name(self, p: Person, name: str) -> None:
        other = self.person_by_name(name)
        if other is not None and other is not p:
            # the same person seen earlier (chat check-in, or a face before a voice): merge into one record
            p.voice = p.voice or other.voice
            p.seat = p.seat or other.seat
            p.joined_at = min(p.joined_at, other.joined_at) if other.joined_at else p.joined_at
            self.state.people.pop(other.key, None)
        p.name, p.guest = name, False

    # ------------------------------------------------------------------ handlers
    async def _on_status(self, ev: Event) -> None:
        if ev.data.get("state") in ("started", "ended"):
            self.state.wipe_people()      # voices, faces, names: session-only
            self.frames.clear()
            self.pending_chat.clear()

    async def _on_chat(self, ev: Event) -> None:
        if ev.data.get("command") != "here":
            return
        name = str((ev.data.get("args") or {}).get("name") or ev.data.get("user") or "").strip()
        if not name:
            return
        p = self.person_by_name(name)
        if p is None:
            p = Person(key=new_id("p"), name=name, guest=False, joined_at=ev.ts, last_seen=ev.ts)
            self.state.people[p.key] = p
        else:
            p.present, p.guest = True, False
        # An unbound chat registration may explicitly retry after an abstention. Already
        # associated people do not acquire a second voice merely by checking in again.
        pending_name = p.name or name
        if not p.voice and not p.seat and pending_name not in self.pending_chat:
            self.pending_chat.append(pending_name)
        await self.log(f"{name} checked in from chat", ev)

    async def _on_speakers(self, ev: Event) -> None:
        d: SpeakersData = ev.data  # type: ignore[assignment]
        t = float(d.get("t", ev.ts))
        seats: dict[str, float] = {}
        tracks: dict[str, str | None] = {}
        for face in d.get("faces") or []:
            seat = face.get("seat")
            tracks[str(face.get("track_id"))] = seat
            if seat and seat != "screen":  # a face on the display is not a room participant
                seats[seat] = max(seats.get(seat, 0.0), float(face.get("speaking") or 0.0))
                p = self.person_by_seat(seat)
                if p is None:
                    p = Person(key=new_id("p"), seat=seat, guest=True, joined_at=t)
                    self.state.people[p.key] = p
                p.present, p.last_seen = True, t
        self.frames.append(_Frame(t, seats, tracks, bool(d.get("screen_talking"))))
        for p in self.state.people.values():
            if p.seat and p.present and p.seat not in seats and t - p.last_seen > self.absent_after:
                p.present = False

    async def _on_segment(self, ev: Event) -> None:
        seg: SegmentData = ev.data  # type: ignore[assignment]
        text = str(seg.get("text") or "").strip()
        if not text:
            return
        start, end = float(seg.get("start", ev.ts)), float(seg.get("end", ev.ts))
        voice = seg.get("voice")
        if voice in (None, "", "voice-?"):              # diarization off, or audio too short to label
            voice = None
        seat, seat_score, media_share = self.match_seat(start, end)
        source = "person"
        conf = float(seg.get("voice_conf", 0.5))  # zero is weak evidence, not the default
        if voice and str(voice).startswith(MEDIA_VOICE_PREFIX) or (seat is None and media_share >= MEDIA_MIN):
            source = "media"
            speaker: Speaker = {"name": None, "voice": voice, "seat": None, "conf": round(max(conf, media_share), 3),
                                "guest": False}
        else:
            # Consume the whole attempt before any await, including weak/unknown speech or
            # an already-known speaker. Several queued registrations are ambiguous, not a
            # queue of names to hand out to future voices. Explicit introductions can still
            # merge their matching roster entries after this one-turn opportunity expires.
            pending_chat, self.pending_chat = self.pending_chat, []
            known = self.person_by_voice(voice)
            seated = self.person_by_seat(seat)
            if (known is None or not known.name) and seat is None and not any(f.seats for f in self.frames):
                # Remembering an anonymous label is not evidence that it came from a person.
                source = "unknown"
            name = checkin_name(text)
            # Window-majority diarization can label a segment containing several speakers.
            # A parsed name must not turn that uncertain voice into a checked-in person (which
            # also affects guest privacy policy). No voice: require strong *current* seat ASD.
            identity_supported = source == "person" and (
                conf >= CHECKIN_MIN_CONF if voice else bool(seat and seat_score >= CHECKIN_MIN_CONF))
            if identity_supported:
                # Low ASD need not disqualify a known strong voice, but it cannot attach a
                # new voice to somebody else's named seat or change a stored association.
                trusted_seat = seat
                if seat_score < CHECKIN_MIN_CONF and ((known and known.name) or (seated and seated.name)):
                    trusted_seat = None
                p = self._get_or_create(voice, trusted_seat if (known is None or not known.seat) else None, end)
                if trusted_seat and p.seat != trusted_seat and self.person_by_seat(trusted_seat) is None:
                    p.seat = trusted_seat
            else:
                # Do not let a weak voice/seat pairing poison the association for a future
                # strong turn. Anonymous voice labels can still be tracked as guests.
                p = known or (seated if not voice else None)
                if p is None:
                    p = self._get_or_create(voice, None, end)
                else:
                    p.present, p.last_seen = True, end
            if name and identity_supported:
                self._name(p, name)
                await self.log(f"{name} checked in: voice {voice or '?'}, seat {p.seat or '?'}", ev)
            elif name:
                await self.log(f"Heard a check-in for {name}, but speaker evidence is uncertain; "
                               "identity not bound. Confirm the participant manually.", ev)
            elif identity_supported and not p.name and len(pending_chat) == 1 and not any(
                    q.voice and q.name == pending_chat[0] for q in self.state.people.values()):
                self._name(p, pending_chat[0])
            speaker = p.as_speaker(min(1.0, 0.5 * conf + 0.5 * (seat_score if seat else conf)))
            if not identity_supported:
                # A learned name is not proof about this particular (possibly mixed) turn.
                # Mask only the emitted view; retain the association for later clear speech.
                # Gate on the current voice evidence above, not the voice/ASD average: a
                # strong known voice must not lose its identity because ASD is weak.
                speaker.update(name=None, guest=True)
        utt: UtteranceData = {
            "utterance_id": new_id("u"), "segment_id": str(seg.get("segment_id") or ev.id),
            "start": start, "end": end, "text": text, "lang": str(seg.get("lang") or "en"),
            "speaker": speaker, "source": source, "context": self.state.context_line(),
        }
        self.state.clock = max(self.state.clock, end)
        await self.emit(MT.UTTERANCE, dict(utt), ev, ts=end)

    # ------------------------------------------------------------------ matching
    def match_seat(self, start: float, end: float) -> tuple[str | None, float, float]:
        """(seat, mean speaking probability, share of frames with the screen talking) over the segment."""
        window = [f for f in self.frames if start - PAD_S <= f.t <= end + PAD_S]
        if not window:
            return None, 0.0, 0.0
        totals: dict[str, float] = {}
        for f in window:
            for seat, prob in f.seats.items():
                totals[seat] = totals.get(seat, 0.0) + prob
        media_share = sum(1 for f in window if f.screen_talking) / len(window)
        if not totals:
            return None, 0.0, media_share
        seat = max(totals, key=totals.get)
        score = totals[seat] / len(window)
        if score < SPEAKING_MIN:
            return None, score, media_share
        return seat, score, media_share
