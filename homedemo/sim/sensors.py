"""Mocked sensors: turn the World's ground truth into the sensor.* payloads of contracts.py.

The hub publishes on change (and re-reports everything on boot or when the World forces it),
with source "sim.<kind>" and ts = the simulated clock. Scores carry seeded jitter scaled by
World.noise; the labels (who, what, where a voice came from) never depend on the jitter.
"""
from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, Any, Iterable

from ..contracts import BusAPI, Event, Speaker, T
from .model import Device, Door, Mic, Person, Room, angle_diff, bearing, distance

if TYPE_CHECKING:
    from .world import World

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
TV_CONE_DEG = 15.0        # a voice within this bearing of a playing TV is heard as the TV
AZIMUTH_JITTER_DEG = 5.0  # at noise 1.0
LUX_MIN_CHANGE = 1.0      # publish lux when it moved by >= max(1 lux, 3 %)
LUX_REL_CHANGE = 0.03
TEMP_MIN_CHANGE = 0.05

# (base, amplitude) of each mocked score at noise 1.0; noise 0 gives the base exactly.
SCORE: dict[str, tuple[float, float]] = {
    "camera_conf": (0.885, 0.085),     # 0.80 .. 0.97
    "face_match": (0.92, 0.05),
    "voice_enrolled": (0.925, 0.045),  # 0.88 .. 0.97
    "voice_guest": (0.30, 0.15),       # < 0.5
    "voice_unknown": (0.15, 0.10),     # TV, outside, strangers
    "asr_conf": (0.92, 0.05),
    "doa_conf": (0.90, 0.06),
}

_MISSING = object()


def detect_lang(text: str) -> str:
    return "zh" if CJK.search(text) else "en"


class SensorHub:
    def __init__(self, world: World, bus: BusAPI) -> None:
        self.world = world
        self.bus = bus
        self._last: dict[tuple[str, str], Any] = {}  # (kind, id) -> last published canonical value
        self._tracks: dict[str, str] = {}            # entity id -> camera track id

    def reset(self) -> None:
        self._last.clear()
        self._tracks.clear()

    # ------------------------------------------------------------------ scores
    def score(self, name: str) -> float:
        base, amp = SCORE[name]
        value = base + self.world.rng.uniform(-amp, amp) * self.world.noise
        return round(min(1.0, max(0.0, value)), 3)

    def speaker_for(self, person: Person | None, match: float | None = None) -> Speaker:
        """Mocked voice ID: enrolled residents match strongly, guests weakly, anyone else not at all."""
        if person is None:
            spk: Speaker = {"id": None, "name": None, "role": "unknown", "match": self.score("voice_unknown")}
        elif person.guest or not person.voice_enrolled:
            spk = {"id": None, "name": None, "role": "guest" if person.guest else "unknown",
                   "match": self.score("voice_guest")}
        else:
            spk = {"id": person.id, "name": person.name, "role": person.role,  # type: ignore[typeddict-item]
                   "match": self.score("voice_enrolled")}
        if match is not None:
            spk["match"] = float(match)
        return spk

    # ------------------------------------------------------------------ publishing
    async def _emit(self, topic: str, kind: str, data: dict[str, Any], cause: str | None) -> Event:
        ev = Event(topic, data, source=f"sim.{kind}", ts=self.world.clock, cause=cause)
        await self.bus.publish(ev)
        return ev

    async def clock(self, cause: str | None = None) -> None:
        await self._emit(T.CLOCK, "clock", self.world.clock_data(), cause)

    async def report_all(self, cause: str | None = None) -> None:
        """Boot report: the clock and every sensor, so agents start from the true state."""
        await self.clock(cause)
        await self.sync(cause, force=True)

    async def sync(self, cause: str | None = None, *, force: bool = False, force_devices: Iterable[str] = (),
                   force_lux: Iterable[str] = (), force_temp: Iterable[str] = ()) -> None:
        """Publish whatever changed since the last sync: devices, doors, rooms (motion, lux,
        temperature, camera frame) and phones, in home.json order."""
        w = self.world
        force_devices, force_lux, force_temp = set(force_devices), set(force_lux), set(force_temp)
        for dev in w.devices.values():
            await self._device(dev, cause, force or dev.device_id in force_devices)
        for door in w.doors.values():
            await self._door(door, cause, force)
        for room in w.rooms.values():
            await self._motion(room, cause, force)
            await self._lux(room, cause, force or room.id in force_lux)
            await self._temp(room, cause, force or room.id in force_temp)
            if room.id in w.cameras:
                await self._camera(room, cause, force)
        for person in w.people.values():
            if person.has_phone:
                await self._phone(person, cause, force)

    def _changed(self, key: tuple[str, str], value: Any, force: bool) -> bool:
        if not force and self._last.get(key, _MISSING) == value:
            return False
        self._last[key] = value
        return True

    async def _device(self, dev: Device, cause: str | None, force: bool) -> None:
        state = copy.deepcopy(dev.state)
        if self._changed(("device", dev.device_id), json.dumps(state, sort_keys=True), force):
            await self._emit(T.DEVICE, "device", {"device_id": dev.device_id, "kind": dev.kind,
                                                  "room": dev.room, "state": state}, cause)

    async def _door(self, door: Door, cause: str | None, force: bool) -> None:
        locked = self.world.door_locked(door)
        if self._changed(("door", door.door_id), (door.open, locked), force):
            await self._emit(T.DOOR, "door", {"door_id": door.door_id, "open": door.open, "locked": locked}, cause)

    async def _motion(self, room: Room, cause: str | None, force: bool) -> None:
        occupied = bool(self.world.occupants(room.id))
        if self._changed(("motion", room.id), occupied, force):
            await self._emit(T.MOTION, "motion", {"room": room.id, "motion": occupied}, cause)

    async def _lux(self, room: Room, cause: str | None, force: bool) -> None:
        key = ("lux", room.id)
        new, last = round(room.lux, 1), self._last.get(key)
        if force or last is None or abs(new - last) >= max(LUX_MIN_CHANGE, LUX_REL_CHANGE * last):
            self._last[key] = new
            await self._emit(T.LUX, "lux", {"room": room.id, "lux": new}, cause)

    async def _temp(self, room: Room, cause: str | None, force: bool) -> None:
        key = ("temp", room.id)
        new, last = round(room.celsius, 2), self._last.get(key)
        if force or last is None or abs(new - last) >= TEMP_MIN_CHANGE:
            self._last[key] = new
            await self._emit(T.TEMP, "temp", {"room": room.id, "celsius": new}, cause)

    async def _phone(self, person: Person, cause: str | None, force: bool) -> None:
        if self._changed(("phone", person.id), person.phone_home, force):
            await self._emit(T.PHONE, "phone", {"person_id": person.id, "home": person.phone_home}, cause)

    # ------------------------------------------------------------------ camera
    def _track(self, entity_id: str) -> str:
        if entity_id not in self._tracks:
            self._tracks[entity_id] = f"t{len(self._tracks) + 1:02d}"
        return self._tracks[entity_id]

    @staticmethod
    def _person_attrs(p: Person) -> dict[str, Any]:
        attrs: dict[str, Any] = {"age_group": p.age_group, **p.attrs}
        if p.idle_min:
            attrs["idle_min"] = round(p.idle_min, 1)
        if not p.guest:
            attrs["identity"] = {"id": p.id}  # the face-match score is added at publish time
        return attrs

    async def _camera(self, room: Room, cause: str | None, force: bool) -> None:
        w = self.world
        people, pets = w.people_in(room.id), w.pets_in(room.id)
        frame = [("person", self._track(p.id), self._person_attrs(p)) for p in people]
        frame += [(pet.species, self._track(pet.id), dict(pet.attrs)) for pet in pets]
        caption = "; ".join(dict.fromkeys(e.activity for e in [*people, *pets] if e.activity)) or None
        # Compare before jitter so a frame is only re-published when what the camera sees changed.
        if not self._changed(("camera", room.id), json.dumps([frame, caption], sort_keys=True), force):
            return
        detections = []
        for label, track, attrs in frame:
            attrs = copy.deepcopy(attrs)
            if "identity" in attrs:
                attrs["identity"]["match"] = self.score("face_match")
            detections.append({"label": label, "conf": self.score("camera_conf"), "track_id": track, "attrs": attrs})
        await self._emit(T.CAMERA, "camera", {"room": room.id, "camera_id": w.cameras[room.id].camera_id,
                                              "detections": detections, "activity": caption}, cause)

    # ------------------------------------------------------------------ audio
    def mic_for(self, room_id: str, xy: tuple[float, float]) -> Mic | None:
        """The mic in that room, else the nearest one (a hallway voice is picked up next door)."""
        in_room = [m for m in self.world.mics if m.room == room_id]
        if in_room:
            return in_room[0]
        return min(self.world.mics, key=lambda m: distance(m.x, m.y, *xy), default=None)

    def hears_as(self, mic: Mic, azimuth: float, source_room: Room) -> str:
        """What the mic array infers: outside, the TV it is pointed at while it plays, else a person."""
        if not source_room.indoor:
            return "outside"
        for tv in self.world.devices_in(mic.room, "tv"):
            if tv.state.get("on") and angle_diff(azimuth, bearing(mic.x, mic.y, tv.x, tv.y)) <= TV_CONE_DEG:
                return "tv"
        return "person"

    async def audio(self, *, text: str, source_room: str, source_xy: tuple[float, float], speaker: Speaker,
                    lang: str | None = None, asr_conf: float | None = None, wake_word: bool = False,
                    cause: str | None = None) -> dict[str, Any]:
        w = self.world
        mic = self.mic_for(source_room, source_xy)
        if mic is None:
            raise ValueError("the home has no microphone")
        true_az = bearing(mic.x, mic.y, *source_xy)
        heard_az = (true_az + w.rng.uniform(-AZIMUTH_JITTER_DEG, AZIMUTH_JITTER_DEG) * w.noise) % 360
        doa = {"azimuth_deg": round(heard_az, 1), "source": self.hears_as(mic, true_az, w.rooms[source_room]),
               "conf": self.score("doa_conf")}
        data = {"room": mic.room, "mic_id": mic.mic_id, "text": text, "lang": lang or detect_lang(text),
                "asr_conf": self.score("asr_conf") if asr_conf is None else float(asr_conf),
                "wake_word": bool(wake_word), "speaker": speaker, "doa": doa}
        await self._emit(T.AUDIO, "audio", data, cause)
        return data
