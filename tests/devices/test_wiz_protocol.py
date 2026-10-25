"""WiZ wire format and the UDP client against FakeWizBulb: round trip, retries, rejection, stale replies."""
from __future__ import annotations

import asyncio
import json

import pytest

from unspoken.devices.fakes import FakeWizBulb
from unspoken.devices.wiz import (REGISTRATION_PARAMS, WizClient, WizError, WizRejected, decode_reply,
                                  discover_wiz, encode_request)


@pytest.fixture
async def bulb():
    b = await FakeWizBulb.start()
    yield b
    await b.stop()


def test_encode_request_is_the_wiz_json_shape():
    raw = encode_request("setPilot", {"state": True, "r": 255, "g": 0, "b": 0, "dimming": 70})
    assert json.loads(raw) == {"method": "setPilot", "params": {"state": True, "r": 255, "g": 0, "b": 0, "dimming": 70}}
    assert json.loads(encode_request("getPilot")) == {"method": "getPilot", "params": {}}
    assert json.loads(encode_request("registration", REGISTRATION_PARAMS))["params"]["register"] is False


def test_decode_reply_rejects_garbage():
    assert decode_reply(b'{"method":"getPilot","env":"pro","result":{"state":true}}')["result"] == {"state": True}
    for bad in (b"", b"nope", b"[1,2]", b"\xff\xfe"):
        with pytest.raises(ValueError):
            decode_reply(bad)


async def test_set_then_get_pilot_round_trip(bulb):
    c = WizClient("127.0.0.1", bulb.port, timeout=0.5, retries=2)
    try:
        assert await c.request("setPilot", {"state": True, "r": 255, "g": 0, "b": 0, "dimming": 70}) == {"success": True}
        got = await c.request("getPilot")
        assert got["state"] is True and (got["r"], got["g"], got["b"], got["dimming"]) == (255, 0, 0, 70)
        assert got["mac"] == bulb.mac
        cfg = await c.request("getSystemConfig")
        assert cfg["moduleName"] == "ESP01_SHRGB_03" and cfg["fwVersion"] == "1.21.4"
        assert c.sent == 3 and c.failures == 0
    finally:
        await c.aclose()


async def test_dropped_datagrams_are_retried_up_to_two_times(bulb):
    c = WizClient("127.0.0.1", bulb.port, timeout=0.1, retries=2)
    try:
        bulb.drop_next = 2
        assert await c.request("setPilot", {"state": True}) == {"success": True}
        assert c.sent == 3 and c.failures == 2 and len(bulb.requests_for("setPilot")) == 3
        bulb.drop_next = 3                                  # one more than the retries allow
        with pytest.raises(WizError, match="after 3 attempt"):
            await c.request("setPilot", {"state": False})
        assert c.sent == 6
    finally:
        await c.aclose()


async def test_bulb_error_is_not_retried(bulb):
    c = WizClient("127.0.0.1", bulb.port, timeout=0.5, retries=2)
    try:
        with pytest.raises(WizRejected, match="Invalid params"):
            await c.request("setPilot", {"state": True, "dimming": 5})      # below the bulb's floor
        assert c.sent == 1 and len(bulb.requests) == 1
    finally:
        await c.aclose()


async def test_a_late_reply_to_an_earlier_request_is_skipped(bulb):
    c = WizClient("127.0.0.1", bulb.port, timeout=0.05, retries=0)
    try:
        bulb.delay_s = 0.15
        with pytest.raises(WizError):
            await c.request("getPilot")                     # times out; the bulb answers later
        bulb.delay_s = 0.0
        await asyncio.sleep(0.2)                            # the stale getPilot reply is now sitting in the socket
        c.timeout = 0.5
        cfg = await c.request("getSystemConfig")            # must not be confused by the stale getPilot
        assert cfg["moduleName"] == "ESP01_SHRGB_03"
    finally:
        await c.aclose()


async def test_closed_port_fails_fast_with_a_readable_error():
    c = WizClient("127.0.0.1", 1, timeout=0.5, retries=1)
    try:
        with pytest.raises(WizError) as ei:
            await c.request("getPilot")
        assert "after 2 attempt" in str(ei.value)
    finally:
        await c.aclose()


async def test_discovery_finds_the_fake_bulb_and_reads_its_config(bulb):
    found = await discover_wiz(broadcast="127.0.0.1", port=bulb.port, timeout=0.3)
    assert found == [{"host": "127.0.0.1", "mac": bulb.mac, "moduleName": "ESP01_SHRGB_03", "fwVersion": "1.21.4"}]
    assert bulb.requests_for("registration") and bulb.requests_for("getSystemConfig")


async def test_discovery_with_nobody_home_returns_nothing():
    assert await discover_wiz(broadcast="127.0.0.1", port=1, timeout=0.2) == []
