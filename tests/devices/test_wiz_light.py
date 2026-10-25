"""WizLight (the Device) end to end against the fake UDP bulb: scenes, set, debounce, failure contract."""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from unspoken.contracts import Device
from unspoken.devices.fakes import FakeWizBulb, scene_pilots_match
from unspoken.devices.scenes import SCENE_NAMES
from unspoken.devices.wiz import WizLight


@pytest.fixture
async def bulb():
    b = await FakeWizBulb.start()
    yield b
    await b.stop()


@pytest.fixture
async def lamp(bulb):
    light = WizLight("lamp", "127.0.0.1", bulb.port, timeout=0.2, retries=2, debounce_s=0.05)
    yield light
    await light.aclose()


def test_wizlight_is_a_device():
    assert isinstance(WizLight("lamp", "127.0.0.1"), Device)


@pytest.mark.parametrize("scene", [s for s in SCENE_NAMES if s != "off"])
async def test_every_scene_reaches_the_bulb(bulb, lamp, scene):
    st = await lamp.apply("scene", {"scene": scene})
    assert st["healthy"] is True and st["error"] == "" and st["scene"] == scene and st["on"] is True
    assert bulb.scene == scene and scene_pilots_match(bulb.pilot, scene)


async def test_on_off_and_set(bulb, lamp):
    await lamp.apply("scene", {"scene": "talk"})
    st = await lamp.apply("off")
    assert st["on"] is False and bulb.pilot["state"] is False and bulb.pilot["temp"] == 4000   # colour kept
    st = await lamp.apply("on")
    assert st["on"] is True and st["scene"] == "talk" and bulb.scene == "talk"
    st = await lamp.apply("set", {"brightness": 40, "color": "#ff8800"})
    assert (st["on"], st["brightness"], st["color"], st["temp"], st["scene"]) == (True, 40, "#ff8800", None, None)
    assert (bulb.pilot["r"], bulb.pilot["g"], bulb.pilot["b"], bulb.pilot["dimming"]) == (255, 136, 0, 40)
    st = await lamp.apply("set", {"brightness": 3})            # below the floor -> 10, colour kept
    assert st["brightness"] == 10 and st["color"] == "#ff8800" and bulb.pilot["dimming"] == 10
    st = await lamp.apply("set", {"brightness": 0})            # 0 means off
    assert st["on"] is False and bulb.pilot["state"] is False


async def test_bad_commands_are_reported_not_raised_and_send_nothing(bulb, lamp):
    await lamp.apply("scene", {"scene": "talk"})
    sent = len(bulb.requests)
    for cmd, args in (("scene", {"scene": "disco"}), ("scene", {}), ("set", {}), ("set", {"color": "red"}),
                      ("explode", {}), ("set", {"brightness": "loud"})):
        st = await lamp.apply(cmd, args)
        assert st["error"].startswith("bad command"), (cmd, args, st)
        assert st["healthy"] is True and st["scene"] == "talk"          # last known state, still healthy
    assert len(bulb.requests) == sent


async def test_a_burst_is_debounced_to_the_newest_command(bulb, lamp):
    await lamp.apply("scene", {"scene": "talk"})
    await asyncio.sleep(0.06)                                             # outside the window: next send is immediate
    bulb.requests.clear()
    scenes = ["slides", "break", "privacy", "paused", "off_record", "bright", "dim", "talk", "privacy", "off_record"]
    t0 = time.perf_counter()
    results = await asyncio.gather(*(lamp.apply("scene", {"scene": s}) for s in scenes))
    elapsed = time.perf_counter() - t0
    sent = bulb.requests_for("setPilot")
    assert 1 <= len(sent) <= 2, sent                                      # leading + at most one trailing send
    assert bulb.scene == "off_record"                                     # the newest wins
    assert results[-1]["scene"] == "off_record" and all(r["healthy"] for r in results)
    assert elapsed < 0.5


async def test_two_commands_apart_in_time_both_go_out(bulb, lamp):
    await lamp.apply("scene", {"scene": "talk"})
    await asyncio.sleep(0.06)
    await lamp.apply("scene", {"scene": "slides"})
    assert [r["params"].get("temp") for r in bulb.requests_for("setPilot")] == [4000, 2700]


async def test_timeouts_retry_then_fall_back_to_last_known_state(bulb, lamp):
    await lamp.apply("scene", {"scene": "talk"})
    bulb.drop_next = 2                                                    # two lost datagrams: still fine
    st = await lamp.apply("scene", {"scene": "slides"})
    assert st["healthy"] is True and bulb.scene == "slides"

    bulb.drop_next = 100                                                  # the bulb is gone
    t0 = time.perf_counter()
    st = await lamp.apply("scene", {"scene": "off_record"})
    assert st["healthy"] is False and "failed after 3 attempt" in st["error"]
    assert st["scene"] == "slides" and st["on"] is True                   # last known state, not the failed command
    assert 0.5 < time.perf_counter() - t0 < 1.5                            # 3 x 0.2 s timeout
    assert await lamp.healthy() is False

    sent_before = lamp.client.sent
    t0 = time.perf_counter()
    st = await lamp.apply("off")                                          # known down: one attempt only
    assert st["healthy"] is False and lamp.client.sent == sent_before + 1
    assert time.perf_counter() - t0 < 0.45

    bulb.drop_next = 0                                                    # it is back
    st = await lamp.apply("scene", {"scene": "privacy"})
    assert st["healthy"] is True and st["error"] == "" and bulb.scene == "privacy"
    assert await lamp.healthy() is True


async def test_rejected_params_keep_the_bulb_healthy_but_report(bulb, lamp):
    await lamp.apply("scene", {"scene": "talk"})
    bulb.reject_next = 1
    st = await lamp.apply("scene", {"scene": "dim"})
    assert st["healthy"] is True and "Invalid params" in st["error"] and st["scene"] == "talk"
    assert lamp.client.sent == 2                                          # no retry for a rejection
    st = await lamp.apply("scene", {"scene": "dim"})
    assert st["error"] == "" and st["scene"] == "dim"


async def test_state_refreshes_from_the_bulb_after_the_ttl(bulb):
    now = [100.0]
    lamp = WizLight("lamp", "127.0.0.1", bulb.port, timeout=0.2, debounce_s=0.0, state_ttl_s=2.0, clock=lambda: now[0])
    try:
        assert await lamp.healthy() is True                               # first contact: getPilot probe
        await lamp.apply("scene", {"scene": "talk"})
        bulb.pilot = {"state": True, "r": 255, "g": 0, "b": 0, "dimming": 70}   # someone used the WiZ app
        assert (await lamp.state())["scene"] == "talk"                    # cached: within the ttl
        now[0] += 3.0
        assert (await lamp.state())["scene"] == "off_record"              # refreshed from the bulb
    finally:
        await lamp.aclose()


async def test_unreachable_host_never_raises():
    lamp = WizLight("lamp", "127.0.0.1", 1, timeout=0.3, retries=2)      # nothing listens on UDP 1
    try:
        st = await lamp.apply("scene", {"scene": "off_record"})
        assert st["healthy"] is False and st["error"] and st["on"] is False
        assert await lamp.healthy() is False
        assert (await lamp.state())["healthy"] is False
        assert await lamp.system_config() == {}
    finally:
        await lamp.aclose()


async def test_system_config_fills_info(bulb, lamp):
    cfg = await lamp.system_config()
    assert cfg["moduleName"] == "ESP01_SHRGB_03" and lamp.info["mac"] == bulb.mac


async def test_live_wiz_bulb_round_trip():
    """A real bulb: UNSPOKEN_LIVE=1 UNSPOKEN_WIZ_HOST=192.168.8.21 pytest tests/devices -k live."""
    if os.environ.get("UNSPOKEN_LIVE") != "1" or not os.environ.get("UNSPOKEN_WIZ_HOST"):
        pytest.skip("set UNSPOKEN_LIVE=1 and UNSPOKEN_WIZ_HOST=<bulb ip> to run against a real bulb")
    lamp = WizLight("lamp", os.environ["UNSPOKEN_WIZ_HOST"])
    try:
        before = await lamp.state()
        assert before["healthy"], before
        for scene in ("privacy", "paused", "off_record", "talk"):
            st = await lamp.apply("scene", {"scene": scene})
            assert st["healthy"] and st["scene"] == scene, st
            await asyncio.sleep(0.7)
        assert (await lamp.state())["scene"] == "talk"
    finally:
        await lamp.aclose()
