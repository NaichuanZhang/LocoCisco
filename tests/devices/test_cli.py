"""The CLI against a fake bulb running as its own process (`python -m unspoken.devices fake-bulb`):
set / list / discover / scenes, exit codes, and the room.toml block `discover` prints."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from unspoken.devices.__main__ import _room_toml_block, main, parse_set_args

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def fake_bulb_process():
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.Popen([sys.executable, "-m", "unspoken.devices", "fake-bulb", "--port", "0"],
                            cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        line = proc.stdout.readline()
        m = re.search(r"127\.0\.0\.1:(\d+)", line)
        assert m, f"fake-bulb did not announce a port: {line!r}"
        yield int(m.group(1))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def room_toml(tmp_path, fake_bulb_process):
    p = tmp_path / "room.toml"
    p.write_text(f'''
[devices.lamp]
kind = "light"
driver = "wiz"
host = "127.0.0.1"
port = {fake_bulb_process}
timeout = 0.5
[devices.fan]
kind = "plug"
driver = "fake"
''')
    return str(p)


def test_parse_set_args():
    assert parse_set_args("scene", ["off_record"]) == {"scene": "off_record"}
    assert parse_set_args("scene", ["scene=talk"]) == {"scene": "talk"}
    assert parse_set_args("set", ["brightness=40", "color=#ff8800"]) == {"brightness": 40, "color": "#ff8800"}
    assert parse_set_args("set", ["40", "#ff8800"]) == {"brightness": 40, "color": "#ff8800"}
    assert parse_set_args("off", []) == {}
    with pytest.raises(SystemExit):
        parse_set_args("off", ["now"])


def test_room_toml_block_is_pasteable():
    block = _room_toml_block(
        [{"host": "192.168.8.21", "mac": "a8bb50aabbcc", "moduleName": "ESP01_SHRGB_03", "fwVersion": "1.21.4"}],
        [{"host": "192.168.8.22", "model": "HS103(US)", "alias": "fan", "mac": "50:C7:BF:00:00:01", "needs_credentials": False},
         {"host": "192.168.8.23", "model": "KP125M(US)", "alias": "heater", "mac": "", "needs_credentials": True}])
    import tomllib
    parsed = tomllib.loads(block)
    assert parsed["devices"]["lamp"] == {"kind": "light", "driver": "wiz", "host": "192.168.8.21"}
    assert parsed["devices"]["fan"] == {"kind": "plug", "driver": "kasa", "host": "192.168.8.22"}
    assert parsed["devices"]["plug2"]["host"] == "192.168.8.23"
    assert "NEEDS KASA_USERNAME/KASA_PASSWORD" in block
    assert "nothing answered" in _room_toml_block([], [])


def test_set_scene_and_set_and_off_against_the_fake_bulb(room_toml, capsys):
    assert main(["--config", room_toml, "set", "lamp", "scene", "off_record"]) == 0
    out = capsys.readouterr().out
    assert "lamp" in out and "ok" in out and "scene=off_record" in out and "color=#ff0000" in out

    assert main(["--config", room_toml, "--json", "set", "lamp", "set", "brightness=40", "color=#ff8800"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert (st["on"], st["brightness"], st["color"], st["healthy"]) == (True, 40, "#ff8800", True)

    assert main(["--config", room_toml, "--json", "set", "lamp", "off"]) == 0
    assert json.loads(capsys.readouterr().out)["on"] is False

    assert main(["--config", room_toml, "--json", "set", "fan", "off"]) == 0
    assert json.loads(capsys.readouterr().out) == {"on": False, "healthy": True, "error": "", "driver": "fake", "host": ""}


def test_set_reports_bad_input_with_exit_1(room_toml, capsys):
    assert main(["--config", room_toml, "set", "lamp", "scene", "disco"]) == 1
    assert "bad command" in capsys.readouterr().out
    assert main(["--config", room_toml, "set", "toaster", "on"]) == 2


def test_list_shows_live_state_and_health(room_toml, capsys):
    main(["--config", room_toml, "set", "lamp", "scene", "privacy"])
    capsys.readouterr()
    assert main(["--config", room_toml, "list"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"lamp\s+light\s+wiz\s+ok.*scene=privacy", out) and re.search(r"fan\s+plug\s+fake\s+ok", out)


def test_list_exits_1_when_a_device_is_down(tmp_path, capsys):
    p = tmp_path / "room.toml"
    p.write_text('[devices.lamp]\nkind = "light"\ndriver = "wiz"\nhost = "127.0.0.1"\nport = 1\ntimeout = 0.2\n'
                 '[devices.fan]\nkind = "plug"\ndriver = "fake"\n')
    t0 = time.perf_counter()
    assert main(["--config", str(p), "--json", "list"]) == 1
    states = json.loads(capsys.readouterr().out)
    assert states["lamp"]["healthy"] is False and states["lamp"]["error"] and states["fan"]["healthy"] is True
    assert time.perf_counter() - t0 < 3.0


def test_discover_prints_a_room_toml_block_for_the_fake_bulb(fake_bulb_process, capsys):
    assert main(["discover", "--no-kasa", "--broadcast", "127.0.0.1", "--wiz-port", str(fake_bulb_process),
                 "--timeout", "0.5"]) == 0
    out = capsys.readouterr().out
    assert "WiZ bulbs: 1" in out and '[devices.lamp]' in out and 'host = "127.0.0.1"' in out and "ESP01_SHRGB_03" in out
    assert main(["--json", "discover", "--no-kasa", "--broadcast", "127.0.0.1", "--wiz-port", str(fake_bulb_process),
                 "--timeout", "0.5"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["wiz"][0]["mac"] == "a8bb50aabbcc" and data["kasa"] == []


def test_scenes_command(capsys):
    assert main(["--json", "scenes"]) == 0
    assert set(json.loads(capsys.readouterr().out)) >= {"talk", "privacy", "off_record"}
    assert main(["scenes"]) == 0
    assert "off_record" in capsys.readouterr().out
