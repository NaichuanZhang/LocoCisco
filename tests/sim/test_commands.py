"""action.command -> device state, action.applied, sensor.device and the sensor changes it causes."""
from __future__ import annotations

from homedemo.contracts import T

from .conftest import Harness, one, only


async def test_light_on_dim_off(sim: Harness):
    lux_before = sim.world.rooms["living"].lux
    cmd, ev = await sim.command("light_living", "on")
    applied = one(ev, T.APPLIED)
    assert applied.data == {"device_id": "light_living", "state": {"on": True, "brightness": 100}, "command_id": cmd.id}
    assert applied.cause == cmd.id and applied.source == "sim.world"
    dev = one(ev, T.DEVICE, device_id="light_living")
    assert dev.data["state"] == {"on": True, "brightness": 100} and dev.cause == cmd.id
    assert one(ev, T.LUX, room="living").data["lux"] == round(lux_before + 300, 1)
    assert [e.topic for e in ev].index(T.APPLIED) < [e.topic for e in ev].index(T.DEVICE)

    _, ev = await sim.command("light_living", "dim", brightness=30)
    assert one(ev, T.APPLIED).data["state"] == {"on": True, "brightness": 30}
    assert one(ev, T.LUX, room="living").data["lux"] == round(lux_before + 90, 1)

    _, ev = await sim.command("light_living", "off")
    assert one(ev, T.APPLIED).data["state"] == {"on": False, "brightness": 0}
    assert one(ev, T.LUX, room="living").data["lux"] == lux_before

    _, ev = await sim.command("light_living", "off")  # no change: still applied + reported, no lux
    assert one(ev, T.APPLIED) and one(ev, T.DEVICE) and not only(ev, T.LUX)


async def test_tv_on_off(sim: Harness):
    _, ev = await sim.command("tv_living", "on", volume=12)
    assert one(ev, T.APPLIED).data["state"] == {"on": True, "volume": 12, "playing": None}
    assert one(ev, T.DEVICE, device_id="tv_living")
    assert not only(ev, T.LUX)
    _, ev = await sim.command("tv_living", "off")
    assert one(ev, T.APPLIED).data["state"]["on"] is False


async def test_thermostat_setpoint(sim: Harness):
    _, ev = await sim.command("thermostat_main", "set_setpoint", delta_c=1.5)
    assert one(ev, T.APPLIED).data["state"] == {"setpoint_c": 20.5}
    _, ev = await sim.command("thermostat_main", "set_setpoint", setpoint_c=22)
    assert one(ev, T.DEVICE, device_id="thermostat_main").data["state"] == {"setpoint_c": 22.0}
    _, ev = await sim.command("thermostat_main", "set_setpoint", delta_c=-50)
    assert one(ev, T.APPLIED).data["state"] == {"setpoint_c": 10.0}  # clamped
    assert not only(ev, T.TEMP)  # rooms drift on tick, not instantly


async def test_lock_unlock_reports_door(sim: Harness):
    cmd, ev = await sim.command("lock_front", "unlock")
    assert one(ev, T.APPLIED).data["state"] == {"locked": False}
    door = one(ev, T.DOOR, door_id="front")
    assert door.data == {"door_id": "front", "open": False, "locked": False} and door.cause == cmd.id
    assert sim.world.snapshot()["doors"][0]["locked"] is False
    _, ev = await sim.command("lock_front", "lock")
    assert one(ev, T.DOOR).data["locked"] is True


async def test_notify(sim: Harness):
    cmd, ev = await sim.command("phone", "notify", to="parents", text="Maya is near the stove")
    assert one(ev, T.APPLIED).data == {"device_id": "phone", "command_id": cmd.id,
                                       "state": {"to": "parents", "text": "Maya is near the stove", "delivered": True}}
    assert not only(ev, T.DEVICE)
    notes = sim.world.snapshot()["notifications"]
    assert len(notes) == 1 and notes[0]["to"] == "parents" and notes[0]["text"] == "Maya is near the stove"
    assert notes[0]["command_id"] == cmd.id and notes[0]["time"] == "19:00"
    _, ev = await sim.command(None, "notify", to="alex", text="hi")
    assert one(ev, T.APPLIED).data["device_id"] == "phone"
    assert len(sim.world.notifications) == 2


async def test_bad_commands_become_trace_logs(sim: Harness):
    for device, command, args in (("tv_living", "dim", {"brightness": 5}), ("toaster", "on", {}),
                                  ("light_living", "dim", {}), ("thermostat_main", "set_setpoint", {}),
                                  ("lock_front", "open", {})):
        _, ev = await sim.command(device, command, **args)
        assert not only(ev, T.APPLIED)
        assert one(ev, T.LOG).data["level"] == "warn"
    assert len(sim.bus.errors) == 5 and all(isinstance(e, ValueError) for _, _, e in sim.bus.errors)
    assert sim.world.devices["tv_living"].state["on"] is False
