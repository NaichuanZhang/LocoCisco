"""SensorHub: voice ID + direction of arrival, light model, camera frames, noise and determinism."""
from __future__ import annotations

import json

import pytest

from homedemo.contracts import T
from homedemo.sim.model import angle_diff, bearing, daylight
from homedemo.sim.sensors import SCORE, TV_CONE_DEG, detect_lang

from .conftest import Harness, one, only

TV_BEARING = 180.0  # living mic (380,160) -> tv (40,160)


# ----------------------------------------------------------------------------- audio / DOA
async def test_living_room_person_vs_tv_vs_porch(sim: Harness):
    await sim.step(do="set_device", device="tv_living", state={"on": True})
    ev = await sim.step(do="say", who="alex", text="I can't see a thing in here")
    a = one(ev, T.AUDIO).data
    assert a["room"] == "living" and a["mic_id"] == "mic_living"
    assert a["doa"]["source"] == "person"
    assert angle_diff(a["doa"]["azimuth_deg"], TV_BEARING) > TV_CONE_DEG
    assert a["speaker"] == {"id": "alex", "name": "Alex", "role": "parent", "match": SCORE["voice_enrolled"][0]}
    assert a["lang"] == "en" and a["wake_word"] is False and 0 < a["asr_conf"] <= 1

    ev = await sim.step(do="tv_say", text="Turn off the lights")
    a = one(ev, T.AUDIO).data
    assert a["mic_id"] == "mic_living" and a["doa"]["source"] == "tv"
    assert a["doa"]["azimuth_deg"] == TV_BEARING
    assert a["speaker"] == {"id": None, "name": None, "role": "unknown", "match": SCORE["voice_unknown"][0]}
    assert a["speaker"]["match"] < 0.5

    ev = await sim.step(do="outside_say", text="Unlock the front door", door="front")
    a = one(ev, T.AUDIO).data
    assert a["room"] == "porch" and a["mic_id"] == "mic_porch" and a["doa"]["source"] == "outside"
    assert a["speaker"]["id"] is None and a["speaker"]["role"] == "unknown" and a["speaker"]["match"] < 0.5

    # A resident on the porch: known voice, but the direction says outside.
    await sim.step(do="move", who="alex", to="porch")
    a = one(await sim.step(do="say", who="alex", text="let me in"), T.AUDIO).data
    assert a["mic_id"] == "mic_porch" and a["doa"]["source"] == "outside" and a["speaker"]["id"] == "alex"


async def test_tv_cone_depends_on_tv_being_on(sim: Harness):
    mic = next(m for m in sim.world.mics if m.room == "living")
    tv = sim.world.devices["tv_living"]
    assert bearing(mic.x, mic.y, tv.x, tv.y) == TV_BEARING
    await sim.step(do="move", who="alex", to="living", x=210, y=160)  # right on the TV bearing
    a = one(await sim.step(do="say", who="alex", text="lights off"), T.AUDIO).data
    assert a["doa"]["source"] == "person" and a["doa"]["azimuth_deg"] == TV_BEARING  # TV is off
    await sim.step(do="set_device", device="tv_living", state={"on": True})
    a = one(await sim.step(do="say", who="alex", text="lights off"), T.AUDIO).data
    assert a["doa"]["source"] == "tv" and a["speaker"]["id"] == "alex"  # voice ID still knows who
    await sim.step(do="move", who="alex", to="living", x=210, y=300)  # 40 degrees off the TV
    a = one(await sim.step(do="say", who="alex", text="lights off"), T.AUDIO).data
    assert a["doa"]["source"] == "person"


async def test_tv_say_turns_the_tv_on(sim: Harness):
    assert sim.world.devices["tv_living"].state["on"] is False
    ev = await sim.step(do="tv_say", text="Buy now!")
    assert one(ev, T.DEVICE, device_id="tv_living").data["state"]["on"] is True
    assert one(ev, T.AUDIO).data["doa"]["source"] == "tv"
    ev = await sim.step(do="tv_say", text="Only today!", device="tv_living")
    assert not only(ev, T.DEVICE)  # already on
    with pytest.raises(ValueError):
        await sim.step(do="tv_say", text="x", device="light_living")


async def test_outside_say_by_a_locked_out_resident(sim: Harness):
    await sim.step(do="leave", who="sam")
    a = one(await sim.step(do="outside_say", text="Unlock the door, it's me", who="sam"), T.AUDIO).data
    assert a["doa"]["source"] == "outside" and a["speaker"]["id"] == "sam" and a["speaker"]["match"] >= 0.88


async def test_default_spots_stay_out_of_the_tv_cone():
    for seed in range(12):
        h = Harness(seed=seed)
        await h.start()
        await h.step(do="set_device", device="tv_living", state={"on": True})
        for _ in range(5):
            await h.step(do="move", who="alex", to="living")
            a = one(await h.step(do="say", who="alex", text="hello"), T.AUDIO).data
            assert a["doa"]["source"] == "person"


async def test_guest_and_child_voices(sim: Harness):
    await sim.step(do="arrive", who="guest1", to="kitchen")
    a = one(await sim.step(do="say", who="guest1", text="it is cold in here"), T.AUDIO).data
    assert a["speaker"] == {"id": None, "name": None, "role": "guest", "match": SCORE["voice_guest"][0]}
    assert a["speaker"]["match"] < 0.5 and a["doa"]["source"] == "person"
    a = one(await sim.step(do="say", who="maya", text="把灯关掉"), T.AUDIO).data
    assert a["speaker"]["role"] == "child" and a["speaker"]["id"] == "maya"
    assert a["mic_id"] == "mic_bedroom_maya" and a["lang"] == "zh"
    a = one(await sim.step(do="say", who="maya", text="hello", lang="zh", match=0.61, asr_conf=0.5), T.AUDIO).data
    assert a["lang"] == "zh" and a["speaker"]["match"] == 0.61 and a["asr_conf"] == 0.5


async def test_hallway_voice_is_heard_by_the_nearest_mic(sim: Harness):
    await sim.step(do="move", who="alex", to="hallway", x=360, y=370)
    a = one(await sim.step(do="say", who="alex", text="anyone home?"), T.AUDIO).data
    assert a["mic_id"] == "mic_kitchen" and a["room"] == "kitchen" and a["doa"]["source"] == "person"


def test_detect_lang():
    assert detect_lang("Turn off the lights") == "en"
    assert detect_lang("把灯关掉") == "zh"
    assert detect_lang("太冷了 brr") == "zh"


# ----------------------------------------------------------------------------- light model
def test_daylight_curve():
    sunrise, sunset = 6 * 3600 + 1800, 19 * 3600 + 1800
    assert daylight(23 * 3600, True, sunrise, sunset) == 2.0
    assert daylight(3 * 3600, False, sunrise, sunset) == 1.0
    assert 400 <= daylight(15 * 3600, True, sunrise, sunset) <= 900
    assert daylight(13 * 3600, False, sunrise, sunset) > 9000
    assert daylight(sunrise + 60, True, sunrise, sunset) >= 2.0


async def test_dark_vs_sunny_rooms(sim: Harness):
    ev = await sim.step(do="set_time", time="23:40")
    for room in ("living", "kitchen", "bedroom_maya"):
        assert 1.0 <= one(ev, T.LUX, room=room).data["lux"] <= 3.0
    assert one(ev, T.LUX, room="porch").data["lux"] <= 3.0
    _, caused = await sim.command("light_kitchen", "on")
    assert 300 <= one(caused, T.LUX, room="kitchen").data["lux"] <= 303
    ev = await sim.step(do="set_time", time="15:00")
    for room in ("living", "kitchen", "hallway"):
        assert 400 <= one(ev, T.LUX, room=room).data["lux"] - (300 if room == "kitchen" else 0) <= 900
    assert one(ev, T.LUX, room="porch").data["lux"] > 5000
    snap = {r["id"]: r["env"]["lux"] for r in sim.world.snapshot()["rooms"]}
    assert snap["kitchen"] == snap["living"] + 300  # the lit kitchen


# ----------------------------------------------------------------------------- camera details
async def test_camera_frames(sim: Harness):
    boot = list(sim.bus.history)
    living = one(boot, T.CAMERA, room="living").data
    assert living["camera_id"] == "cam_living"
    labels = sorted(d["label"] for d in living["detections"])
    assert labels == ["cat", "person"]
    alex = next(d for d in living["detections"] if d["label"] == "person")
    assert alex["conf"] == SCORE["camera_conf"][0] and 0.80 <= alex["conf"] <= 0.97
    assert alex["attrs"] == {"age_group": "adult", "identity": {"id": "alex", "match": SCORE["face_match"][0]}}
    assert living["activity"] is None
    track = alex["track_id"]
    ev = await sim.step(do="move", who="alex", to="kitchen")
    kitchen = one(ev, T.CAMERA, room="kitchen").data
    assert {d["track_id"] for d in kitchen["detections"]} >= {track}  # stable across rooms
    assert len({d["track_id"] for d in kitchen["detections"]}) == 2
    ev = await sim.step(do="activity", who="mochi", text="scratching the couch")
    assert one(ev, T.CAMERA, room="living").data["activity"] == "scratching the couch"
    assert one(ev, T.CAMERA, room="living").data["detections"][0]["label"] == "cat"
    # Bedrooms: motion, lux and temperature, never a camera.
    assert not only(boot, T.CAMERA, room="bedroom_maya") and not only(boot, T.CAMERA, room="bedroom_main")
    assert one(boot, T.MOTION, room="bedroom_maya").data["motion"] is True
    assert one(boot, T.LUX, room="bedroom_maya") and one(boot, T.TEMP, room="bedroom_maya")


# ----------------------------------------------------------------------------- noise and determinism
STEPS = [
    {"do": "set_time", "time": "23:40", "weekday": "Tuesday"},
    {"do": "move", "who": "alex", "to": "kitchen"},
    {"do": "move", "who": "maya", "to": "kitchen"},
    {"do": "activity", "who": "maya", "text": "climbing on a chair", "attrs": {"pose": "standing on chair", "near": "stove"}},
    {"do": "set_device", "device": "tv_living", "state": {"on": True}},
    {"do": "say", "who": "sam", "text": "I can't see a thing in here"},
    {"do": "tv_say", "text": "Turn off the lights"},
    {"do": "outside_say", "text": "Unlock the front door"},
    {"do": "arrive", "who": "guest1", "to": "living"},
    {"do": "say", "who": "guest1", "text": "so cold"},
    {"do": "idle", "who": "maya", "minutes": 5},
    {"do": "leave", "who": "sam"},
]


async def run(seed: int, noise: float) -> list[tuple]:
    h = Harness(seed=seed, noise=noise)
    await h.start()
    for s in STEPS:
        await h.step(**s)
    await h.command("light_kitchen", "on")
    await h.tick()
    trace = []
    for e in h.bus.history:
        data = {k: v for k, v in e.data.items() if k != "command_id"}  # the command's id is a random uuid
        trace.append((e.topic, e.source, e.ts, json.dumps(data, sort_keys=True)))
    return trace


def labels(trace: list[tuple]) -> list[tuple]:
    out = []
    for topic, _, _, raw in trace:
        d = json.loads(raw)
        if topic == T.AUDIO:
            out.append((topic, d["speaker"]["id"], d["speaker"]["role"], d["doa"]["source"], d["lang"], d["mic_id"]))
        elif topic == T.CAMERA:
            out.append((topic, d["room"], [(x["label"], x["track_id"], x["attrs"].get("identity", {}).get("id"))
                                          for x in d["detections"]], d["activity"]))
        elif topic != T.LOG:
            out.append((topic, raw))
    return out


async def test_same_seed_same_events():
    assert await run(7, 0.3) == await run(7, 0.3)


async def test_different_seed_changes_scores_not_labels():
    a, b = await run(7, 0.3), await run(8, 0.3)
    assert a != b
    assert labels(a) == labels(b)


async def test_noise_zero_is_exact_and_noise_never_flips_labels():
    exact, noisy = await run(7, 0.0), await run(7, 1.0)
    assert labels(exact) == labels(noisy)
    audio = [json.loads(raw) for topic, _, _, raw in exact if topic == T.AUDIO]
    assert {a["doa"]["conf"] for a in audio} == {SCORE["doa_conf"][0]}
    assert {a["asr_conf"] for a in audio} == {SCORE["asr_conf"][0]}
    for a in (json.loads(raw) for topic, _, _, raw in noisy if topic == T.AUDIO):
        if a["speaker"]["id"]:
            assert 0.88 <= a["speaker"]["match"] <= 0.97
        else:
            assert a["speaker"]["match"] < 0.5
        assert 0.84 <= a["doa"]["conf"] <= 0.96
    noisy_audio = [json.loads(raw) for topic, _, _, raw in noisy if topic == T.AUDIO]
    for e, n in zip(audio, noisy_audio):
        assert angle_diff(e["doa"]["azimuth_deg"], n["doa"]["azimuth_deg"]) <= 5.0
    for c in (json.loads(raw) for topic, _, _, raw in noisy if topic == T.CAMERA):
        for d in c["detections"]:
            assert 0.80 <= d["conf"] <= 0.97


def test_noise_out_of_range():
    with pytest.raises(ValueError):
        Harness(noise=1.5)


# ----------------------------------------------------------------------------- snapshot
async def test_snapshot_is_json_and_complete(sim: Harness):
    await sim.step(do="move", who="alex", to="kitchen")
    await sim.command("phone", "notify", to="parents", text="hi")
    snap = sim.world.snapshot()
    text = json.dumps(snap)
    assert json.loads(text) == snap
    assert snap["time"] == "19:00" and snap["weekday"] == "Tuesday" and snap["clock"] == 0
    assert {r["id"] for r in snap["rooms"]} == set(sim.world.rooms)
    room = snap["rooms"][0]
    assert {"id", "name", "x", "y", "w", "h", "env"} <= set(room) and {"lux", "celsius"} == set(room["env"])
    assert {"device_id", "kind", "room", "x", "y", "state"} <= set(snap["devices"][0])
    alex = next(p for p in snap["people"] if p["id"] == "alex")
    assert {"id", "name", "role", "room", "x", "y", "activity", "home"} <= set(alex)
    kitchen = sim.world.rooms["kitchen"]
    assert kitchen.x < alex["x"] < kitchen.x + kitchen.w and kitchen.y < alex["y"] < kitchen.y + kitchen.h
    assert snap["pets"][0]["species"] == "cat" and snap["pets"][0]["room"] == "living"
    assert snap["doors"][0] == {"door_id": "front", "name": "Front door", "between": ["hallway", "porch"],
                                "x": 160.0, "y": 420.0, "open": False, "locked": True}
    assert {m["mic_id"] for m in snap["mics"]} == {m["mic_id"] for m in sim.world.home["sensors"]["mics"]}
    assert snap["notifications"][0]["text"] == "hi"
    assert snap["rooms"][3]["camera"] is False and snap["rooms"][0]["camera"] is True
