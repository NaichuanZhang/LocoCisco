"""python -m unspoken.models show|check: table and JSON output, exit codes, clear config errors."""
from __future__ import annotations

import json

from unspoken.contracts import ROLES
from unspoken.models.__main__ import main


def test_show_prints_a_table_with_every_role(capsys):
    assert main(["show", "--profile", "gb10"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("profile gb10  local_only=yes")
    assert all(role in out for role in ROLES) and "nvidia/Qwen3.6-35B-A3B-NVFP4" in out and "python:unspoken.sensors.vad:SileroVAD" in out


def test_show_json_and_role_subset(capsys, monkeypatch):
    monkeypatch.setenv("UNSPOKEN_LLM_URL", "http://127.0.0.1:9999/v1")
    assert main(["show", "--profile", "dev-local", "--roles", "llm,decider", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert list(doc["roles"]) == ["llm", "decider"] and doc["roles"]["llm"]["url"] == "http://127.0.0.1:9999/v1"
    assert doc["applied"] == ["env: llm.url = 'http://127.0.0.1:9999/v1'"] and doc["local_only"] is True


def test_check_fake_profile_is_green(capsys):
    assert main(["check", "--profile", "fake"]) == 0
    out = capsys.readouterr().out
    assert "12/12 enabled roles ok" in out and "FAIL" not in out and "P(true)=" in out


def test_check_json_and_exit_1_on_a_dead_role(tmp_path, capsys):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.pii]\nprovider='regex'\n[profiles.p.decider]\nprovider='strands-decider'\n"
                 "url='http://127.0.0.1:9'\noptions={timeout=2}\n")
    assert main(["check", "--profile", "p", "--path", str(p), "--roles", "pii,decider,asr", "--json", "--timeout", "5"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["ok"] is False and [r["ok"] for r in doc["roles"]] == [True, False, True]
    assert doc["roles"][2]["detail"] == "disabled"
    assert main(["check", "--profile", "p", "--path", str(p), "--roles", "pii,decider"]) == 1
    assert "FAILED: decider" in capsys.readouterr().out


def test_config_errors_exit_1_with_the_message(tmp_path, capsys):
    assert main(["show", "--profile", "nope"]) == 1
    assert "ConfigError: " in capsys.readouterr().err
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\nlocal_only=true\n[profiles.p.llm]\nprovider='bedrock'\nmodel='x'\n")
    assert main(["check", "--profile", "p", "--path", str(p)]) == 1
    assert "RemoteNotAllowed" in capsys.readouterr().err
