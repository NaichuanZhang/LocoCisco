"""load_devices(config): real drivers from room.toml, fakes for driver="fake", unhealthy fakes for anything
missing or broken; load_room_config never raises; probe_devices reads every state."""
from __future__ import annotations

import tomllib


from unspoken.contracts import Device
from unspoken.devices import (REQUIRED_DEVICES, ROOM_CONFIG_PATH, FakeLight, FakePlug, KasaPlug, WizLight,
                              close_devices, load_devices, load_room_config, probe_devices)


def _toml(text: str) -> dict:
    return tomllib.loads(text)


async def test_repo_room_toml_loads_two_healthy_fakes():
    cfg = load_room_config(ROOM_CONFIG_PATH)
    devices = load_devices(cfg)
    assert set(devices) >= set(REQUIRED_DEVICES)
    assert isinstance(devices["lamp"], FakeLight) and isinstance(devices["fan"], FakePlug)
    states = await probe_devices(devices)
    assert all(s["healthy"] for s in states.values()), states
    assert (await devices["lamp"].apply("scene", {"scene": "off_record"}))["scene"] == "off_record"
    assert (await devices["fan"].apply("on"))["on"] is True


def test_real_drivers_are_built_from_the_config():
    devices = load_devices(_toml('''
        [devices.lamp]
        kind = "light"
        driver = "wiz"
        host = "192.168.8.21"
        port = 38900
        timeout = 0.5
        retries = 1
        debounce_s = 0.3
        [devices.fan]
        kind = "plug"
        driver = "kasa"
        host = "192.168.8.22"
        [devices.spare]
        kind = "plug"
        driver = "fake"
    '''))
    lamp, fan = devices["lamp"], devices["fan"]
    assert isinstance(lamp, WizLight) and (lamp.host, lamp.port, lamp.client.timeout, lamp.client.retries,
                                           lamp.debounce_s) == ("192.168.8.21", 38900, 0.5, 1, 0.3)
    assert isinstance(fan, KasaPlug) and fan.host == "192.168.8.22" and fan.port is None and fan.timeout == 1.0
    assert isinstance(devices["spare"], FakePlug) and devices["spare"].kind == "plug"
    assert all(isinstance(d, Device) for d in devices.values())


async def test_missing_devices_become_unhealthy_fakes():
    devices = load_devices(_toml('''
        [devices.lamp]
        kind = "light"
        driver = "fake"
    '''))
    assert isinstance(devices["fan"], FakePlug)
    st = await devices["fan"].state()
    assert st["healthy"] is False and "not configured" in st["error"]
    st = await devices["fan"].apply("on")                 # never crashes, never pretends
    assert st["on"] is False and st["healthy"] is False
    assert (await devices["lamp"].state())["healthy"] is True


async def test_empty_or_wrong_config_still_gives_a_lamp_and_a_fan():
    for cfg in (None, {}, {"room": {"name": "x"}}, {"devices": "nope"}, {"devices": {}}):
        devices = load_devices(cfg)   # type: ignore[arg-type]
        assert set(devices) == {"lamp", "fan"}
        states = await probe_devices(devices)
        assert all(s["healthy"] is False for s in states.values())
        assert devices["lamp"].kind == "light" and devices["fan"].kind == "plug"


def test_bare_devices_table_is_accepted_too():
    devices = load_devices({"lamp": {"kind": "light", "driver": "fake"}, "fan": {"kind": "plug", "driver": "fake"}})
    assert isinstance(devices["lamp"], FakeLight) and isinstance(devices["fan"], FakePlug)


async def test_broken_entries_become_unhealthy_fakes_with_the_reason():
    devices = load_devices(_toml('''
        [devices.lamp]
        kind = "light"
        driver = "hue"                 # unknown driver
        host = "192.168.8.21"
        [devices.fan]
        kind = "plug"
        driver = "kasa"                # no host
        [devices.strip]
        kind = "ledstrip"              # unknown kind
        [devices.heater]
        kind = "plug"
        driver = "kasa"
        host = "192.168.8.30"
        timeout = "soon"               # bad value
    '''))
    states = await probe_devices(devices)
    assert isinstance(devices["lamp"], FakeLight) and "unknown driver 'hue'" in states["lamp"]["error"]
    assert isinstance(devices["fan"], FakePlug) and "needs host" in states["fan"]["error"]
    assert "unknown kind 'ledstrip'" in states["strip"]["error"]
    assert isinstance(devices["heater"], FakePlug) and "bad config" in states["heater"]["error"]
    assert all(s["healthy"] is False for s in states.values())


def test_fake_entries_can_be_declared_unhealthy_for_demos():
    devices = load_devices({"devices": {"lamp": {"kind": "light", "driver": "fake", "healthy": False, "error": "unplugged"},
                                        "fan": {"kind": "plug", "driver": "fake"}}})
    assert isinstance(devices["lamp"], FakeLight) and devices["lamp"]._error == "unplugged"


def test_load_room_config_never_raises(tmp_path, monkeypatch):
    assert load_room_config(tmp_path / "missing.toml") == {}
    broken = tmp_path / "broken.toml"
    broken.write_text("[devices.lamp\nkind = ")
    assert load_room_config(broken) == {}
    good = tmp_path / "room.toml"
    good.write_text('[devices.lamp]\nkind = "light"\ndriver = "fake"\n')
    assert load_room_config(good)["devices"]["lamp"]["driver"] == "fake"
    monkeypatch.setenv("UNSPOKEN_ROOM_CONFIG", str(good))
    assert load_room_config()["devices"]["lamp"]["kind"] == "light"


async def test_probe_devices_guards_against_a_foreign_device_that_raises():
    class Grumpy:
        device_id, kind = "grumpy", "plug"

        async def apply(self, command, args=None):
            raise RuntimeError("no")

        async def state(self):
            raise RuntimeError("no")

        async def healthy(self):
            return False

    states = await probe_devices({"grumpy": Grumpy(), "fan": FakePlug("fan")})   # type: ignore[dict-item]
    assert states["grumpy"]["healthy"] is False and "RuntimeError" in states["grumpy"]["error"]
    assert states["fan"]["healthy"] is True


async def test_close_devices_is_safe_for_every_kind():
    devices = load_devices(_toml('''
        [devices.lamp]
        kind = "light"
        driver = "wiz"
        host = "127.0.0.1"
        [devices.fan]
        kind = "plug"
        driver = "fake"
    '''))
    await close_devices(devices)
