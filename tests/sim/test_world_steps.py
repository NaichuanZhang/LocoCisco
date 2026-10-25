"""Every WORLD step of the scenario DSL, the boot report, reset and tick."""
from __future__ import annotations

import pytest

from homedemo.contracts import SCENARIO_STEPS, T
from homedemo.sim.world import RUNNER_STEPS

from .conftest import Harness, one, only


def test_every_world_step_has_a_handler():
    world_steps = set(SCENARIO_STEPS) - RUNNER_STEPS
    assert world_steps == {"set_time", "move", "activity", "idle", "say", "tv_say", "outside_say",
                           "set_env", "set_device", "phone", "leave", "arrive"}
    for name in world_steps:
        assert callable(getattr(Harness().world, f"_step_{name}"))


async def test_boot_report_covers_every_sensor():
    h = Harness()
    await h.world.start()
    await h.bus.drain()
    ev = list(h.bus.history)
    clock = one(ev, T.CLOCK)
    assert clock.data == {"time": "19:00", "weekday": "Tuesday", "school_night": True}
    assert {e.data["device_id"] for e in only(ev, T.DEVICE)} == {d["device_id"] for d in h.world.home["devices"]}
    assert one(ev, T.DOOR).data == {"door_id": "front", "open": False, "locked": True}
    motion = {e.data["room"]: e.data["motion"] for e in only(ev, T.MOTION)}
    assert motion == {"living": True, "kitchen": True, "hallway": False, "bedroom_main": False,
                      "bedroom_maya": True, "porch": False}
    assert {e.data["room"] for e in only(ev, T.CAMERA)} == {"living", "kitchen", "hallway", "porch"}
    assert len(only(ev, T.LUX)) == len(only(ev, T.TEMP)) == 6
    assert {e.data["person_id"] for e in only(ev, T.PHONE)} == {"alex", "sam"}  # Maya has no phone
    assert all(e.source.startswith("sim.") and e.ts == 0.0 for e in ev)


async def test_start_is_idempotent_for_command_subscription(sim: Harness):
    await sim.world.start()
    await sim.bus.drain()
    sim.new()
    _, caused = await sim.command("light_hallway", "on")
    assert len(only(caused, T.APPLIED)) == 1


async def test_set_time_jumps_clock_and_daylight(sim: Harness):
    ev = await sim.step(do="set_time", time="23:40", weekday="Tuesday")
    assert one(ev, T.CLOCK).data == {"time": "23:40", "weekday": "Tuesday", "school_night": True}
    assert sim.world.clock == pytest.approx(4 * 3600 + 40 * 60)
    assert all(e.ts == sim.world.clock for e in ev)
    assert 1.0 <= one(ev, T.LUX, room="living").data["lux"] <= 3.0  # dusk -> night
    # Earlier clock time means the next day; weekday rolls over by itself.
    ev = await sim.step(do="set_time", time="07:00")
    assert one(ev, T.CLOCK).data["weekday"] == "Wednesday"
    assert sim.world.school_night is False
    ev = await sim.step(do="set_time", weekday="Saturday")
    assert one(ev, T.CLOCK).data == {"time": "07:00", "weekday": "Saturday", "school_night": False}
    with pytest.raises(ValueError):
        await sim.step(do="set_time", time="25:00")


async def test_move_person_updates_motion_and_cameras(sim: Harness):
    ev = await sim.step(do="move", who="alex", to="hallway")
    assert one(ev, T.MOTION, room="hallway").data["motion"] is True
    assert not only(ev, T.MOTION, room="living")  # the cat is still there
    hall = one(ev, T.CAMERA, room="hallway").data
    assert [d["attrs"]["identity"]["id"] for d in hall["detections"]] == ["alex"]
    living = one(ev, T.CAMERA, room="living").data
    assert [d["label"] for d in living["detections"]] == ["cat"]
    alex = sim.world.people["alex"]
    room = sim.world.rooms["hallway"]
    assert room.x < alex.x < room.x + room.w and room.y < alex.y < room.y + room.h
    assert next(p for p in sim.world.snapshot()["people"] if p["id"] == "alex")["room"] == "hallway"


async def test_move_pet_empties_room(sim: Harness):
    await sim.step(do="move", who="alex", to="kitchen")
    ev = await sim.step(do="move", who="mochi", to="bedroom_main")
    assert one(ev, T.MOTION, room="living").data["motion"] is False
    assert one(ev, T.MOTION, room="bedroom_main").data["motion"] is True
    assert one(ev, T.CAMERA, room="living").data["detections"] == []
    assert not only(ev, T.CAMERA, room="bedroom_main")  # bedrooms have no camera


async def test_move_with_explicit_position_and_unknowns(sim: Harness):
    await sim.step(do="move", who="alex", to="kitchen", x=500, y=100)
    assert (sim.world.people["alex"].x, sim.world.people["alex"].y) == (500.0, 100.0)
    with pytest.raises(ValueError):
        await sim.step(do="move", who="nobody", to="kitchen")
    with pytest.raises(ValueError):
        await sim.step(do="move", who="alex", to="garage")


async def test_activity_child_near_stove(sim: Harness):
    await sim.step(do="move", who="maya", to="kitchen")
    ev = await sim.step(do="activity", who="maya", text="climbing on a chair next to the stove",
                        attrs={"pose": "standing on chair", "near": "stove", "holding": ["pan"]})
    cam = one(ev, T.CAMERA, room="kitchen").data
    maya = next(d for d in cam["detections"] if d["attrs"].get("identity", {}).get("id") == "maya")
    assert maya["label"] == "person"
    assert maya["attrs"]["age_group"] == "child"
    assert maya["attrs"]["pose"] == "standing on chair"
    assert maya["attrs"]["near"] == "stove"
    assert maya["attrs"]["holding"] == ["pan"]
    assert cam["activity"] == "climbing on a chair next to the stove"
    stove = sim.world.devices["stove_kitchen"]
    p = sim.world.people["maya"]
    assert abs(p.x - stove.x) <= 40 and abs(p.y - stove.y) <= 40
    # A later activity replaces the attrs; moving rooms drops pose/near but keeps clothing.
    ev = await sim.step(do="activity", who="maya", text="eating at the table", attrs={"wearing": ["pyjamas"]})
    attrs = one(ev, T.CAMERA, room="kitchen").data["detections"][-1]["attrs"]
    assert "pose" not in attrs and attrs["wearing"] == ["pyjamas"]
    await sim.step(do="activity", who="maya", text="standing", attrs={"pose": "standing", "wearing": ["coat"]})
    ev = await sim.step(do="move", who="maya", to="hallway")
    attrs = one(ev, T.CAMERA, room="hallway").data["detections"][0]["attrs"]
    assert "pose" not in attrs and attrs["wearing"] == ["coat"]
    assert one(ev, T.CAMERA, room="hallway").data["activity"] is None


async def test_activity_in_bedroom_is_private(sim: Harness):
    ev = await sim.step(do="activity", who="maya", text="reading in bed", attrs={"pose": "lying down"})
    assert ev == []  # no camera in bedrooms, nothing else changed
    assert sim.world.people["maya"].activity == "reading in bed"


async def test_idle_jumps_clock_keeps_presence(sim: Harness):
    ev = await sim.step(do="idle", who="alex", minutes=40)
    assert one(ev, T.CLOCK).data["time"] == "19:40"
    assert sim.world.clock == 40 * 60
    assert not only(ev, T.MOTION)  # presence stays true, nothing to report
    cam = one(ev, T.CAMERA, room="living").data
    alex = next(d for d in cam["detections"] if d["label"] == "person")
    assert alex["attrs"]["idle_min"] == 40
    ev = await sim.step(do="idle", who="alex", minutes=20)
    assert one(ev, T.CAMERA, room="living").data["detections"][0]["attrs"]["idle_min"] == 60
    assert sim.world.snapshot()["people"][0]["idle_min"] == 60
    with pytest.raises(ValueError):
        await sim.step(do="idle", who="alex", minutes=0)


async def test_idle_in_bedroom_only_moves_the_clock(sim: Harness):
    await sim.step(do="set_time", time="23:00")
    ev = await sim.step(do="idle", who="maya", minutes=30)
    assert [e.topic for e in ev] == [T.CLOCK]
    assert one(ev, T.CLOCK).data["time"] == "23:30"


async def test_set_env_overrides_until_next_set_time(sim: Harness):
    ev = await sim.step(do="set_env", room="kitchen", lux=900, celsius=17)
    assert one(ev, T.LUX, room="kitchen").data["lux"] == 900.0
    assert one(ev, T.TEMP, room="kitchen").data["celsius"] == 17.0
    _, caused = await sim.command("light_kitchen", "on")
    assert one(caused, T.LUX, room="kitchen").data["lux"] == 1200.0
    ev = await sim.step(do="set_time", time="23:00")
    assert one(ev, T.LUX, room="kitchen").data["lux"] == pytest.approx(302.0)
    ev = await sim.step(do="set_env", room="porch", celsius=5)
    assert one(ev, T.TEMP, room="porch").data["celsius"] == 5.0 and not only(ev, T.LUX)
    with pytest.raises(ValueError):
        await sim.step(do="set_env", room="kitchen")


async def test_set_device(sim: Harness):
    ev = await sim.step(do="set_device", device="tv_living", state={"on": True, "volume": 30})
    dev = one(ev, T.DEVICE, device_id="tv_living").data
    assert dev["kind"] == "tv" and dev["room"] == "living"
    assert dev["state"] == {"on": True, "volume": 30, "playing": None}
    ev = await sim.step(do="set_device", device="lock_front", state={"locked": False})
    assert one(ev, T.DEVICE, device_id="lock_front").data["state"] == {"locked": False}
    assert one(ev, T.DOOR, door_id="front").data == {"door_id": "front", "open": False, "locked": False}
    ev = await sim.step(do="set_device", device="stove_kitchen", state={"burner_on": True})
    assert one(ev, T.DEVICE, device_id="stove_kitchen").data["state"] == {"burner_on": True}
    ev = await sim.step(do="set_device", device="light_bedroom_maya", state={"on": True})
    assert one(ev, T.DEVICE, device_id="light_bedroom_maya").data["state"] == {"on": True, "brightness": 100}
    assert one(ev, T.LUX, room="bedroom_maya").data["lux"] > 300
    ev = await sim.step(do="set_device", device="light_bedroom_maya", state={"brightness": 0})
    assert one(ev, T.DEVICE, device_id="light_bedroom_maya").data["state"] == {"on": False, "brightness": 0}
    with pytest.raises(ValueError):
        await sim.step(do="set_device", device="tv_living", state={"colour": "red"})
    with pytest.raises(ValueError):
        await sim.step(do="set_device", device="tv_living", state={})
    with pytest.raises(ValueError):
        await sim.step(do="set_device", device="toaster", state={"on": True})


async def test_phone(sim: Harness):
    ev = await sim.step(do="phone", who="alex", home=False)
    assert one(ev, T.PHONE).data == {"person_id": "alex", "home": False}
    assert await sim.step(do="phone", who="alex", home=False) == []  # no change, no event
    with pytest.raises(ValueError):
        await sim.step(do="phone", who="maya", home=False)  # no phone
    with pytest.raises(ValueError):
        await sim.step(do="phone", who="alex")


async def test_leave_and_arrive(sim: Harness):
    ev = await sim.step(do="leave", who="sam")
    doors = only(ev, T.DOOR, door_id="front")
    assert [d.data["open"] for d in doors] == [True, False] and all(d.data["locked"] for d in doors)
    assert one(ev, T.MOTION, room="kitchen").data["motion"] is False
    assert one(ev, T.CAMERA, room="kitchen").data["detections"] == []
    assert one(ev, T.PHONE).data == {"person_id": "sam", "home": False}
    sam = next(p for p in sim.world.snapshot()["people"] if p["id"] == "sam")
    assert sam["home"] is False and sam["room"] is None and sam["x"] is None
    assert await sim.step(do="leave", who="sam") == []  # already out

    ev = await sim.step(do="arrive", who="guest1", to="hallway", name="Jo")
    assert [d.data["open"] for d in only(ev, T.DOOR)] == [True, False]
    assert one(ev, T.MOTION, room="hallway").data["motion"] is True
    guest = one(ev, T.CAMERA, room="hallway").data["detections"][0]
    assert guest["label"] == "person" and guest["attrs"] == {"age_group": "adult"}  # no identity for guests
    assert not only(ev, T.PHONE)
    snap = next(p for p in sim.world.snapshot()["people"] if p["id"] == "guest1")
    assert snap["name"] == "Jo" and snap["role"] == "guest" and snap["guest"] is True and snap["home"] is True

    ev = await sim.step(do="arrive", who="sam", to="kitchen")
    assert one(ev, T.PHONE).data == {"person_id": "sam", "home": True}
    assert one(ev, T.MOTION, room="kitchen").data["motion"] is True
    assert sim.world.people["sam"].home is True
    # Moving someone who is out brings them home; arriving when already home is a move.
    await sim.step(do="leave", who="alex")
    ev = await sim.step(do="move", who="alex", to="living")
    assert one(ev, T.PHONE).data == {"person_id": "alex", "home": True}
    ev = await sim.step(do="arrive", who="alex", to="kitchen")
    assert not only(ev, T.DOOR) and sim.world.people["alex"].room == "kitchen"
    with pytest.raises(ValueError):
        await sim.step(do="arrive", who="guest2", to="living", role="boss")


async def test_say_requires_presence(sim: Harness):
    await sim.step(do="leave", who="alex")
    with pytest.raises(ValueError):
        await sim.step(do="say", who="alex", text="hello")
    with pytest.raises(ValueError):
        await sim.step(do="say", who="alex", text="")
    with pytest.raises(ValueError):
        await sim.step(do="say", who="mochi", text="meow")


async def test_unknown_and_runner_steps_raise(sim: Harness):
    for bad in ({"do": "dance"}, {"do": "wait", "seconds": 1}, {"do": "reply", "answer": "Yes"},
                {"do": "expect", "what": "device"}, {"do": "note", "text": "x"}, {"no_do": 1}, {"do": "_build"}):
        with pytest.raises(ValueError):
            await sim.world.step(bad)
    assert sim.new() == []


async def test_reset_restores_initial_state(sim: Harness):
    initial = sim.world.snapshot()
    await sim.step(do="set_time", time="23:40")
    await sim.step(do="move", who="alex", to="kitchen")
    await sim.command("light_kitchen", "on")
    await sim.command("phone", "notify", to="parents", text="hi")
    assert sim.world.notifications
    sim.new()
    await sim.world.reset()
    await sim.bus.drain()
    ev = sim.new()
    assert one(ev, T.CLOCK).data["time"] == "19:00"
    assert len(only(ev, T.DEVICE)) == 10 and len(only(ev, T.CAMERA)) == 4
    assert sim.world.snapshot() == initial  # same seed -> same spots


async def test_tick_drifts_temperature_and_reports_minutes(sim: Harness):
    await sim.command("thermostat_main", "set_setpoint", delta_c=1.5)
    ev = await sim.tick()
    temps = {e.data["room"]: e.data["celsius"] for e in only(ev, T.TEMP)}
    assert temps == {r: 19.5 for r in ("living", "kitchen", "hallway", "bedroom_main", "bedroom_maya")}
    await sim.tick()
    ev = await sim.tick()
    assert {e.data["celsius"] for e in only(ev, T.TEMP)} == {20.5}
    assert await sim.tick() == []  # settled, nothing to report
    assert sim.world.clock == 4.0
    ev = await sim.tick(56)
    assert one(ev, T.CLOCK).data["time"] == "19:01"
    assert sim.world.rooms["porch"].celsius == 12.0  # outdoors does not follow the thermostat


async def test_tick_follows_daylight(sim: Harness):
    await sim.step(do="set_time", time="15:00")
    before = sim.world.rooms["living"].lux
    seen = []
    for _ in range(10):
        seen += await sim.tick(360)  # one hour in 6-minute ticks
    assert sim.world.rooms["living"].lux < before
    assert only(seen, T.LUX, room="living")
    assert sim.world.time_str == "16:00"
