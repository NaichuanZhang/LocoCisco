"""config/models.toml resolution: profile and path precedence, `extends`, env and dict overrides, and the
errors a typo must produce."""
from __future__ import annotations

import pytest

from unspoken.contracts import ROLES
from unspoken.models import ConfigError, DEFAULT_PATH, load_registry
from unspoken.models.config import env_overrides, resolve

TOML = """
[profiles.base]
local_only = false

[profiles.base.decider]
provider = "strands-decider"
url = "http://127.0.0.1:8765"
label = "small"
options = { name = "decider-2b" }

[profiles.base.llm]
provider = "openai"
url = "http://127.0.0.1:8010/v1"
model = "qwen"
options = { max_tokens = 100 }

[profiles.base.pii]
provider = "regex"

[profiles.child]
extends = "base"
local_only = true

[profiles.child.llm]
provider = "scripted"
label = "scripted"
"""


@pytest.fixture
def toml(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text(TOML)
    return str(p)


def test_every_role_gets_a_spec_and_missing_roles_are_disabled(toml):
    r = resolve("base", toml, environ={})
    assert set(r.specs) == set(ROLES) and r.profile == "base" and r.local_only is False
    assert r.specs["decider"].url == "http://127.0.0.1:8765" and r.specs["decider"].options == {"name": "decider-2b"}
    assert r.specs["llm"].model == "qwen" and r.specs["llm"].options["max_tokens"] == 100
    assert r.specs["asr"].provider == "none" and r.specs["vad"].provider == "none"


def test_extends_merges_whole_role_blocks(toml):
    r = resolve("child", toml, environ={})
    assert r.local_only is True
    assert r.specs["decider"].label == "small"                  # inherited
    assert r.specs["llm"].provider == "scripted" and r.specs["llm"].url is None   # replaced, not merged
    assert r.specs["pii"].provider == "regex"


def test_profile_and_path_come_from_env_when_not_given(toml):
    r = resolve(None, None, environ={"UNSPOKEN_PROFILE": "child", "UNSPOKEN_MODELS": toml})
    assert r.profile == "child" and r.path == toml


def test_profile_defaults_to_dev_and_path_to_repo_config():
    r = resolve(None, None, environ={})
    assert r.profile == "dev" and r.path == str(DEFAULT_PATH)
    assert r.specs["decider"].url == "http://127.0.0.1:8765"


def test_env_overrides_only_match_exact_role_names():
    env = {"UNSPOKEN_LLM_URL": "http://10.0.0.5:8000/v1", "UNSPOKEN_DECIDER_BIG_PROVIDER": "none",
           "UNSPOKEN_PROFILE": "gb10", "UNSPOKEN_MCP_URL": "https://not-a-role", "UNSPOKEN_LLM_URI": "typo"}
    assert env_overrides(env) == {"llm.url": "http://10.0.0.5:8000/v1", "decider_big.provider": "none"}


def test_env_then_dict_overrides_win_in_that_order(toml):
    env = {"UNSPOKEN_LLM_URL": "http://127.0.0.1:9000/v1", "UNSPOKEN_LLM_MODEL": "from-env"}
    r = resolve("base", toml, overrides={"llm.model": "from-dict", "decider.provider": "fake",
                                         "llm.options.max_tokens": "7", "llm.options.stream": "false"},
                environ=env)
    llm = r.specs["llm"]
    assert llm.url == "http://127.0.0.1:9000/v1" and llm.model == "from-dict"
    assert llm.options == {"max_tokens": 7, "stream": False}          # coerced like the TOML would be
    assert r.specs["decider"].provider == "fake"
    assert [a.split(":")[0] for a in r.applied] == ["env", "env", "overrides", "overrides", "overrides", "overrides"]


def test_override_on_a_role_the_profile_does_not_name_enables_it(toml):
    r = resolve("base", toml, overrides={"asr.provider": "fake"}, environ={})
    assert r.specs["asr"].provider == "fake"


@pytest.mark.parametrize("key", ["nope.url", "llm.uri", "llm", "llm.options", "llm.options.a.b"])
def test_bad_override_keys_are_clear_errors(toml, key):
    with pytest.raises(ConfigError) as e:
        resolve("base", toml, overrides={key: "x"}, environ={})
    assert key in str(e.value)


def test_unknown_profile_lists_the_known_ones(toml):
    with pytest.raises(ConfigError, match="no profile 'nope'; have \\['base', 'child'\\]"):
        resolve("nope", toml, environ={})


def test_missing_file_names_the_env_var(tmp_path):
    with pytest.raises(ConfigError, match="UNSPOKEN_MODELS"):
        resolve("dev", str(tmp_path / "missing.toml"), environ={})


@pytest.mark.parametrize("body,needle", [
    ("[profiles.p]\n[profiles.p.lllm]\nprovider='fake'\n", "unknown role"),
    ("[profiles.p]\n[profiles.p.llm]\nprovider='fake'\nurll='x'\n", "unknown key"),
    ("[profiles.p]\n[profiles.p.llm]\nurl='x'\n", "needs a provider"),
    ("[profiles.p]\n[profiles.p.llm]\nprovider=3\n", "must be a string"),
    ("[profiles.p]\nlocal_only='yes'\n", "must be true or false"),
    ("[profiles.p]\nextends='p'\n", "extends itself"),
    ("[profiles.p]\nextends='q'\n[profiles.q]\nextends='p'\n", "extends itself"),
    ("nothing = 1\n", "no \\[profiles"),
    ("[profiles.p\n", "not valid TOML"),
])
def test_bad_files_are_clear_errors(tmp_path, body, needle):
    p = tmp_path / "m.toml"
    p.write_text(body)
    with pytest.raises(ConfigError, match=needle):
        resolve("p", str(p), environ={})


def test_load_registry_threads_profile_path_and_overrides(toml, monkeypatch):
    monkeypatch.delenv("UNSPOKEN_PROFILE", raising=False)
    reg = load_registry("child", toml, overrides={"pii.provider": "fake"})
    assert reg.profile == "child" and reg.local_only and reg.spec("pii").provider == "fake"
    assert reg.path == toml and any("pii.provider" in a for a in reg.applied)


def test_repo_config_has_the_profiles_and_they_all_resolve():
    for profile in ("dev", "dev-local", "dev-sensors", "dev-best", "gb10", "fake"):
        r = resolve(profile, None, environ={})
        assert r.specs["decider"].provider in ("strands-decider", "fake"), profile
    gb10 = resolve("gb10", None, environ={})
    assert gb10.local_only and gb10.specs["llm"].url == "http://127.0.0.1:8000/v1"
    assert gb10.specs["vad"].provider == "python:unspoken.sensors.vad:SileroVAD"
    assert gb10.specs["asd"].provider == "python:unspoken.transcribe.asd_live:LightASDLive"     # transcriber test winner
    assert gb10.specs["diarization"].provider == "python:unspoken.transcribe.diar:OnlineWindowDiarizer"
    best = resolve("dev-best", None, environ={})
    assert best.local_only and best.specs["asr"].provider == "python:unspoken.transcribe.asr_parakeet:ParakeetASR"
    dev_local = resolve("dev-local", None, environ={})
    assert dev_local.local_only and dev_local.specs["llm"].provider == "openai"
    assert dev_local.specs["decider"].url == "http://127.0.0.1:8765"      # extends dev
    sensors = resolve("dev-sensors", None, environ={})
    assert sensors.specs["llm"].url == dev_local.specs["llm"].url            # extends dev-local
    assert sensors.specs["asr"].provider == "python:unspoken.sensors.asr_local:FasterWhisperASR"
    assert sensors.specs["asr"].model == "Systran/faster-whisper-base.en"
    assert sensors.specs["asr"].options == {"device": "cpu", "compute_type": "int8", "language": "en", "threads": 4}
    assert resolve("dev", None, environ={}).specs["llm"].provider == "bedrock"
