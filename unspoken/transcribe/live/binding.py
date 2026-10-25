"""Online voice <-> seat binding that can revise itself, and the per-segment speaker decision.

Evidence: for every final segment the diarizer gives a voice label and the ASD gives, per seat, the mean
speaking probability of the face in that seat during the segment. `SeatBinder.add` accumulates
duration-weighted probability per (voice, seat). Once a voice has `min_dur` seconds of evidence it binds to
the seat that holds `share_min` of its evidence, if its per-second score there is at least `rel_min` of the
strongest voice's score on that seat (so a quiet off-camera voice whose noise leaks onto a seat does not
join the person sitting there). Several voices may bind to the same seat (the online diarizer splits one
person into voice-A, voice-E, ... and this merges them back into one person); one voice binds to at most one
seat. Bindings are re-evaluated after every segment (with hysteresis) and `recompute()` reports the changes
so already-emitted captions can be revised. An unbound voice stays an off-camera voice (`voice-X`).

`decide` picks the speaker of one segment: the segment's own ASD evidence when it is strong (someone on
camera clearly talking), else the voice's binding, else the bare voice label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .captions import UNKNOWN_VOICE


@dataclass
class Decision:
    speaker: str                 # "person-<seat>" | "voice-X" | "voice-?"
    source: str                  # "video" (segment ASD) | "fused" (voice binding) | "audio" (voice only) | "none"
    conf: float
    seat: str | None
    from_binding: bool           # True when the speaker came from the voice binding (revisable)


def rank_seats(seat_probs: dict[str, float]) -> tuple[str | None, float, float]:
    """(best seat, its score, runner-up score)."""
    if not seat_probs:
        return None, 0.0, 0.0
    ranked = sorted(seat_probs.items(), key=lambda kv: -kv[1])
    best, score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return best, float(score), float(second)


class SeatBinder:
    """Binding is scale-free on purpose: MotionASD's probabilities are small (seat means ~0.01-0.02 on the
    AMI close-ups) but the RATIO between seats is consistent per voice, so a voice binds to the seat that
    holds `share_min` of its evidence, provided its per-second score there is at least `rel_min` of the
    strongest voice's score on that seat (the seat's owner). Off-camera voices leave about equal noise on
    both seats and fall under both tests. A better ASD (real probabilities) makes the per-segment
    `strong_min` rule in `decide` kick in on top of this."""

    def __init__(self, *, speaking_min: float = 0.30, margin: float = 0.10, min_dur: float = 3.0,
                 strong_min: float = 0.50, strong_margin: float = 0.25, share_min: float = 0.60,
                 rel_min: float = 0.40, abs_floor: float = 0.003, switch_gain: float = 1.3,
                 similarity: Callable[[str, str], float | None] | None = None, merge_sim: float = 0.30) -> None:
        # similarity(voice_a, voice_b) -> cosine of the diarizer's centroids (None = unknown). A voice joins a
        # seat another voice already holds only when it sounds like that voice (>= merge_sim): the online
        # diarizer splits ONE person into near-threshold fragments, while two people are far apart, so this
        # keeps a wrong ASD vote from collapsing two people into one on-camera person.
        self.similarity = similarity
        self.merge_sim = merge_sim
        self.speaking_min = speaking_min        # per-segment: mean ASD probability that pins a seat on its own
        self.margin = margin
        self.min_dur = min_dur                  # seconds of evidence before a voice may bind
        self.strong_min = strong_min            # per-segment ASD this clear overrides the voice binding
        self.strong_margin = strong_margin
        self.share_min = share_min              # share of a voice's evidence the winning seat must hold
        self.rel_min = rel_min                  # ... and its score there vs the seat's strongest voice
        self.abs_floor = abs_floor              # below this per-second score nothing binds (pure zeros)
        self.switch_gain = switch_gain          # a bound voice moves only to a seat scoring this much more
        self.evidence: dict[str, dict[str, float]] = {}   # voice -> seat -> sum(duration * probability)
        self.dur: dict[str, float] = {}                   # voice -> seconds of evidence
        self.bound: dict[str, str] = {}                   # voice -> seat

    # ------------------------------------------------------------------ evidence
    def add(self, voice: str | None, duration: float, seat_probs: dict[str, float]) -> None:
        if not voice or voice == UNKNOWN_VOICE or duration <= 0:
            return
        ev = self.evidence.setdefault(voice, {})
        for seat, p in seat_probs.items():
            ev[seat] = ev.get(seat, 0.0) + duration * float(p)
        self.dur[voice] = self.dur.get(voice, 0.0) + duration

    def scores(self, voice: str) -> dict[str, float]:
        d = self.dur.get(voice, 0.0)
        return {s: v / d for s, v in self.evidence.get(voice, {}).items()} if d > 0 else {}

    def seat_of(self, voice: str | None) -> str | None:
        return self.bound.get(voice) if voice else None

    def _owner_score(self, seat: str) -> float:
        """Per-second score of the strongest voice on `seat` (voices with enough evidence only)."""
        return max((self.scores(v).get(seat, 0.0) for v, d in self.dur.items() if d >= self.min_dur), default=0.0)

    def _may_join(self, voice: str, seat: str) -> bool:
        """True unless the seat is held by voices this one does not sound like."""
        holders = [v for v, s in self.bound.items() if s == seat and v != voice]
        if not holders or self.similarity is None:
            return True
        sims = [self.similarity(voice, h) for h in holders]
        known = [x for x in sims if x is not None]
        return not known or max(known) >= self.merge_sim

    def _qualifies(self, voice: str, seat: str, loose: bool = False) -> bool:
        sc = self.scores(voice)
        total = sum(sc.values())
        s = sc.get(seat, 0.0)
        if total <= 0 or s < self.abs_floor:
            return False
        share_min = self.share_min - (0.10 if loose else 0.0)
        rel_min = self.rel_min * (0.75 if loose else 1.0)
        owner = self._owner_score(seat)
        return s / total >= share_min and (owner <= 0 or s / owner >= rel_min)

    def recompute(self) -> dict[str, tuple[str | None, str | None]]:
        """Re-evaluate every voice; returns {voice: (old seat, new seat)} for the bindings that changed."""
        changes: dict[str, tuple[str | None, str | None]] = {}
        for voice, d in self.dur.items():
            if d < self.min_dur:
                continue
            best, score, _ = rank_seats(self.scores(voice))
            old = self.bound.get(voice)
            if old is None:
                new = best if best is not None and self._qualifies(voice, best) and self._may_join(voice, best) else None
            elif best is not None and best != old and self._qualifies(voice, best) and \
                    score >= self.switch_gain * self.scores(voice).get(old, 0.0):
                new = best
            else:
                new = old if self._qualifies(voice, old, loose=True) else None
            if new != old:
                changes[voice] = (old, new)
                if new is None:
                    self.bound.pop(voice, None)
                else:
                    self.bound[voice] = new
        return changes

    # ------------------------------------------------------------------ decision
    def decide(self, voice: str | None, voice_conf: float, seat_probs: dict[str, float]) -> Decision:
        best, score, second = rank_seats(seat_probs)
        bound = self.seat_of(voice)
        if best is not None and score >= self.strong_min and score - second >= self.strong_margin:
            return Decision(f"person-{best}", "video", round(min(1.0, 0.5 + score / 2), 3), best, False)
        if bound is not None:
            conf = 0.5 * float(voice_conf) + 0.5 * self.scores(voice or "").get(bound, 0.0)
            return Decision(f"person-{bound}", "fused", round(min(1.0, conf), 3), bound, True)
        if best is not None and score >= self.speaking_min and score - second >= self.margin:
            return Decision(f"person-{best}", "video", round(min(1.0, 0.4 + score / 2), 3), best, False)
        if voice and voice != UNKNOWN_VOICE:
            return Decision(voice, "audio", round(float(voice_conf), 3), None, True)
        return Decision(UNKNOWN_VOICE, "none", 0.0, None, True)

    def reset(self) -> None:
        self.evidence.clear()
        self.dur.clear()
        self.bound.clear()


__all__ = ["SeatBinder", "Decision", "rank_seats"]
