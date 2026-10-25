"""KasaPlug: the connector seam with FakeKasaDevice, env-only credentials, the never-raise contract, and
python-kasa's REAL client against FakeKasaIotPlug (legacy IOT protocol on loopback)."""
from __future__ import annotations

import importlib.util
import time

import pytest

from unspoken.contracts import Device
from unspoken.devices.fakes import FakeKasaDevice, FakeKasaIotPlug, fake_kasa_connector, xor_decrypt, xor_encrypt
from unspoken.devices.kasa_plug import KasaPlug, credentials_from_env

HAVE_KASA = importlib.util.find_spec("kasa") is not None


def test_kasaplug_is_a_device():
    assert isinstance(KasaPlug("fan", "192.0.2.1", connect=fake_kasa_connector(FakeKasaDevice())), Device)


def test_credentials_only_from_env(monkeypatch):
    monkeypatch.delenv("KASA_USERNAME", raising=False)
    monkeypatch.delenv("KASA_PASSWORD", raising=False)
    assert credentials_from_env() == (None, None)
    assert credentials_from_env({"KASA_USERNAME": "me@example.com"}) == (None, None)       # half a pair is none
    assert credentials_from_env({"KASA_USERNAME": "me@example.com", "KASA_PASSWORD": "pw"}) == ("me@example.com", "pw")


async def test_env_credentials_reach_the_connector(monkeypatch):
    monkeypatch.setenv("KASA_USERNAME", "me@example.com")
    monkeypatch.setenv("KASA_PASSWORD", "secret")
    connects: list = []
    fan = KasaPlug("fan", "192.0.2.1", port=9999, timeout=0.7, connect=fake_kasa_connector(FakeKasaDevice(), connects=connects))
    await fan.apply("on")
    assert connects == [("192.0.2.1", "me@example.com", "secret", 0.7, 9999)]
    monkeypatch.delenv("KASA_USERNAME")
    monkeypatch.delenv("KASA_PASSWORD")
    connects.clear()
    fan2 = KasaPlug("fan", "192.0.2.1", connect=fake_kasa_connector(FakeKasaDevice(), connects=connects))
    await fan2.apply("on")
    assert connects[0][1:3] == (None, None)


async def test_on_off_state_through_the_seam():
    dev = FakeKasaDevice(alias="Desk fan", model="KP115(US)")
    fan = KasaPlug("fan", "192.0.2.1", connect=fake_kasa_connector(dev))
    st = await fan.apply("on")
    assert st["on"] is True and st["healthy"] is True and st["error"] == ""
    assert st["alias"] == "Desk fan" and st["model"] == "KP115(US)" and dev.is_on is True
    st = await fan.apply("off")
    assert st["on"] is False and dev.is_on is False
    assert dev.calls == ["turn_on", "update", "turn_off", "update"]
    assert (await fan.state())["on"] is False and await fan.healthy() is True
    st = await fan.apply("toggle")
    assert st["error"].startswith("bad command") and st["on"] is False
    await fan.aclose()
    assert dev.calls[-1] == "disconnect"


async def test_connect_failure_is_reported_not_raised():
    connects: list = []
    fan = KasaPlug("fan", "192.0.2.1", retries=2,
                   connect=fake_kasa_connector(FakeKasaDevice(), fail_connect=ConnectionError("no route"), connects=connects))
    st = await fan.apply("on")
    assert st["healthy"] is False and "no route" in st["error"] and "after 3 attempt" in st["error"] and st["on"] is False
    assert len(connects) == 3
    assert await fan.healthy() is False
    st = await fan.apply("off")                            # known down: a single attempt
    assert len(connects) == 4 and st["healthy"] is False


async def test_mid_session_failure_reconnects_then_reports_last_known_state():
    dev = FakeKasaDevice(fail_with=ConnectionResetError("reset"), fail_after=2)
    connects: list = []
    fan = KasaPlug("fan", "192.0.2.1", retries=1, connect=fake_kasa_connector(dev, connects=connects))
    assert (await fan.apply("on"))["on"] is True            # calls 1-2 succeed
    st = await fan.apply("off")                              # turn_off fails -> reconnect -> fails again
    assert st["healthy"] is False and "reset" in st["error"] and st["on"] is True
    assert len(connects) == 2
    dev.fail_with = None                                     # the plug is back
    st = await fan.apply("off")
    assert st["healthy"] is True and st["on"] is False


def test_xor_helpers_round_trip():
    assert xor_decrypt(xor_encrypt(b'{"system":{"get_sysinfo":{}}}')) == b'{"system":{"get_sysinfo":{}}}'
    assert xor_encrypt(b"\x00")[0] == 171


@pytest.fixture
async def fake_plug():
    plug = await FakeKasaIotPlug.start()
    yield plug
    await plug.stop()


@pytest.mark.skipif(not HAVE_KASA, reason="python-kasa not installed (requirements/devices.txt)")
async def test_real_python_kasa_client_drives_the_fake_legacy_plug(fake_plug):
    fan = KasaPlug("fan", "127.0.0.1", port=fake_plug.port, timeout=1.0, retries=1)
    try:
        t0 = time.perf_counter()
        st = await fan.state()
        assert st["healthy"] is True and st["on"] is False and st["model"] == "HS103" and st["alias"] == "fan"
        assert fake_plug.udp_requests >= 1                               # python-kasa discovered it over UDP
        assert time.perf_counter() - t0 < 3.0
        st = await fan.apply("on")
        assert st["on"] is True and fake_plug.on is True
        st = await fan.apply("off")
        assert st["on"] is False and fake_plug.on is False
        assert any("set_relay_state" in r.get("system", {}) for r in fake_plug.requests)

        fake_plug.fail_tcp = True                                        # the plug drops every connection
        st = await fan.apply("on")
        assert st["healthy"] is False and st["error"] and st["on"] is False
        fake_plug.fail_tcp = False
        st = await fan.apply("on")
        assert st["healthy"] is True and st["on"] is True and fake_plug.on is True
    finally:
        await fan.aclose()


@pytest.mark.skipif(not HAVE_KASA, reason="python-kasa not installed (requirements/devices.txt)")
async def test_real_python_kasa_client_with_nobody_home():
    fan = KasaPlug("fan", "127.0.0.1", port=1, timeout=1.0, retries=0)
    st = await fan.apply("on")
    assert st["healthy"] is False and "Timed out" in st["error"]
