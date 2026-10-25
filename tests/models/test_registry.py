"""ModelRegistry behaviour with no servers: lazy build + cache, disabled roles, python: factories, protocol
checks, local_only refusals, check() and aclose()."""
from __future__ import annotations

import sys
import types

import pytest

from unspoken.contracts import ROLES, RemoteNotAllowed, RoleDisabled, RoleSpec
from unspoken.core.decider import FakeDecider, HttpDecider
from unspoken.models import load_registry
from unspoken.models.fakes import FakeASR, FakeEmbedder
from unspoken.models.netpolicy import host_is_local, url_is_local
from unspoken.models.providers import providers_for

FAKE = {"profile": "fake"}


def test_fake_profile_builds_every_role_lazily_and_caches():
    reg = load_registry("fake")
    assert reg._built == {}                                      # nothing built at load time
    d = reg.decider()
    assert isinstance(d, FakeDecider) and d.name == "decider-2b" and reg.decider() is d
    assert isinstance(reg.decider("decider_big"), FakeDecider)
    assert reg.strands_model("llm").get_config() == {"model_id": "scripted"}
    assert isinstance(reg.asr(), FakeASR) and isinstance(reg.embedder(), FakeEmbedder)
    for getter in (reg.vad, reg.diarizer, reg.faces, reg.asd, reg.tts, reg.pii):
        assert getter() is getter()
    assert all(reg.enabled(r) for r in ROLES) and reg.label("llm") == "scripted model"
    assert "profile='fake'" in repr(reg)


def test_disabled_role_raises_role_disabled_and_label_says_off():
    reg = load_registry("dev")
    assert not reg.enabled("vlm") and reg.label("vlm") == "off"
    with pytest.raises(RoleDisabled, match="vlm"):
        reg.strands_model("vlm")
    assert reg.enabled("decider") and isinstance(reg.decider(), HttpDecider)
    assert reg.decider().label == "Strands Decider 2B (CPU)" and reg.decider().name == "decider-2b"


def test_unknown_role_and_wrong_getter_are_clear_errors():
    reg = load_registry("fake")
    with pytest.raises(ValueError, match="unknown role 'nope'"):
        reg.spec("nope")
    with pytest.raises(ValueError, match="decider\\(\\) takes role"):
        reg.decider("llm")
    with pytest.raises(ValueError, match="strands_model\\(\\) takes role"):
        reg.strands_model("asr")


def test_unknown_provider_lists_the_choices_for_the_role(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.asr]\nprovider='whisper-cpp'\n")
    reg = load_registry("p", str(p))
    with pytest.raises(ValueError) as e:
        reg.asr()
    assert "no provider 'whisper-cpp'" in str(e.value) and "openai-audio" in str(e.value)
    assert providers_for("asr") == ["fake", "openai-audio", "python:<module>:<Factory>", "none"]


def test_python_factory_gets_the_spec_and_must_implement_the_protocol(tmp_path, monkeypatch):
    mod = types.ModuleType("fake_sensors_for_test")
    seen: list[RoleSpec] = []

    class GoodASR:
        def __init__(self, spec: RoleSpec) -> None:
            seen.append(spec)

        async def transcribe(self, pcm16, sample_rate, lang=None):
            return None

        async def healthy(self):
            return True

    class NotAnASR:
        def __init__(self, spec: RoleSpec) -> None:
            pass

    mod.GoodASR, mod.NotAnASR = GoodASR, NotAnASR
    monkeypatch.setitem(sys.modules, "fake_sensors_for_test", mod)
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.asr]\nprovider='python:fake_sensors_for_test:GoodASR'\n"
                 "options={device='cpu'}\n[profiles.p.tts]\nprovider='python:fake_sensors_for_test:NotAnASR'\n"
                 "[profiles.p.vad]\nprovider='python:fake_sensors_for_test:Missing'\n"
                 "[profiles.p.pii]\nprovider='python:no_such_module_xyz:X'\n[profiles.p.embed]\nprovider='python:bad'\n")
    reg = load_registry("p", str(p))
    asr = reg.asr()
    assert isinstance(asr, GoodASR) and seen[0].role == "asr" and seen[0].options == {"device": "cpu"}
    with pytest.raises(TypeError, match="does not implement TTS"):
        reg.tts()
    with pytest.raises(AttributeError, match="no attribute 'Missing'"):
        reg.vad()
    with pytest.raises(ImportError, match="no_such_module_xyz"):
        reg.pii()
    with pytest.raises(ValueError, match="python:<module>:<Factory>"):
        reg.embedder()


# --------------------------------------------------------------------------- local_only
@pytest.mark.parametrize("host,ok", [
    ("127.0.0.1", True), ("localhost", True), ("::1", True), ("10.1.2.3", True), ("172.16.0.9", True),
    ("172.31.255.1", True), ("192.168.1.20", True), ("gb10.local", True), ("inference.local", True),
    ("fd00::1", True), ("::ffff:192.168.0.2", True),
    ("172.32.0.1", False), ("8.8.8.8", False), ("api.openai.com", False), ("bedrock-runtime.us-west-2.amazonaws.com", False),
    ("169.254.1.1", False), ("100.64.0.1", False), ("", False), (None, False),
])
def test_host_is_local(host, ok):
    assert host_is_local(host) is ok


def test_url_is_local_reads_the_host_only():
    assert url_is_local("http://127.0.0.1:8000/v1") and url_is_local("https://[::1]:8443/x")
    assert url_is_local("http://user:pw@192.168.0.4/v1") and not url_is_local("http://example.com:127.0.0.1/")
    assert not url_is_local("not a url") and not url_is_local("http://")


def _toml(tmp_path, body):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\nlocal_only = true\n" + body)
    return str(p)


def test_local_only_refuses_bedrock_at_load_time(tmp_path):
    path = _toml(tmp_path, "[profiles.p.llm]\nprovider='bedrock'\nmodel='x'\n")
    with pytest.raises(RemoteNotAllowed, match="'bedrock'"):
        load_registry("p", path)


def test_local_only_refuses_a_public_url_and_names_the_role(tmp_path):
    path = _toml(tmp_path, "[profiles.p.asr]\nprovider='openai-audio'\nurl='https://api.openai.com/v1'\n")
    with pytest.raises(RemoteNotAllowed, match="role 'asr' url 'https://api.openai.com/v1'"):
        load_registry("p", path)


def test_local_only_refuses_a_remote_override_too(tmp_path):
    path = _toml(tmp_path, "[profiles.p.llm]\nprovider='openai'\nurl='http://127.0.0.1:8000/v1'\nmodel='m'\n")
    assert load_registry("p", path).local_only
    with pytest.raises(RemoteNotAllowed):
        load_registry("p", path, overrides={"llm.url": "http://203.0.113.9:8000/v1"})


def test_local_only_allows_lan_loopback_and_mdns_and_disabled_roles(tmp_path):
    path = _toml(tmp_path, "[profiles.p.llm]\nprovider='openai'\nurl='http://inference.local:8000/v1'\nmodel='m'\n"
                           "[profiles.p.asr]\nprovider='openai-audio'\nurl='http://192.168.7.7:8770/v1'\n"
                           "[profiles.p.tts]\nprovider='none'\n[profiles.p.vlm]\nprovider='openai'\nurl='http://[::1]:8000/v1'\nmodel='m'\n")
    reg = load_registry("p", path)
    assert reg.local_only and not reg.enabled("tts")


def test_gb10_profile_is_local_only_and_dev_is_not():
    assert load_registry("gb10").local_only is True
    assert load_registry("dev").local_only is False
    assert load_registry("dev-local").local_only is True


# --------------------------------------------------------------------------- check / aclose
async def test_check_probes_every_role_of_the_fake_profile():
    reg = load_registry("fake")
    health = await reg.check()
    assert [h.role for h in health] == list(ROLES) and all(h.ok for h in health)
    by = {h.role: h for h in health}
    assert by["decider"].detail.startswith("P(true)=") and by["decider"].target == "in-process"
    assert by["llm"].detail.startswith("stop=tool_use") and by["tts"].detail.endswith("wav")
    assert by["embed"].detail == "dim=64" and by["pii"].detail == "0 span(s)"
    assert all(h.latency_ms is not None and h.latency_ms >= 0 for h in health)
    await reg.aclose()
    assert reg._built == {}


async def test_check_reports_disabled_roles_and_a_subset(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.pii]\nprovider='regex'\n")
    reg = load_registry("p", str(p))
    health = await reg.check(["pii", "asr"])
    assert [(h.role, h.ok, h.detail) for h in health] == [("pii", True, "1 span(s)"), ("asr", True, "disabled")]
    with pytest.raises(ValueError, match="unknown role"):
        await reg.check(["nope"])


async def test_check_marks_a_dead_server_failed_without_raising(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.decider]\nprovider='strands-decider'\nurl='http://127.0.0.1:9'\n"
                 "options={timeout=2}\n[profiles.p.embed]\nprovider='openai'\nurl='http://127.0.0.1:9/v1'\nmodel='e'\n"
                 "options={timeout=2}\n")
    reg = load_registry("p", str(p))
    health = await reg.check(["decider", "embed"], timeout=5)
    assert [h.ok for h in health] == [False, False]
    assert "healthy() is false" in health[0].detail and health[0].target == "http://127.0.0.1:9"
    await reg.aclose()


async def test_check_catches_a_broken_python_factory(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.vad]\nprovider='python:no_such_module_abc:VAD'\n")
    reg = load_registry("p", str(p))
    (h,) = await reg.check(["vad"])
    assert not h.ok and h.detail.startswith("ImportError") and h.target == "in-process"


async def test_check_accepts_sync_healthy_and_a_describe_on_in_process_providers(tmp_path, monkeypatch):
    mod = types.ModuleType("sync_health_for_test")

    class SyncVAD:
        def __init__(self, spec: RoleSpec) -> None:
            self.ok = spec.options.get("ok", True)

        def healthy(self) -> bool:          # sync, like unspoken.sensors.vad.SileroVAD
            return bool(self.ok)

        def describe(self) -> str:
            return "silero v5, cpu"

        def stream(self, sample_rate: int = 16000):
            return FakeVADStreamLike()

    class FakeVADStreamLike:
        def feed(self, frame: bytes):
            return []

        def flush(self):
            return []

    mod.SyncVAD = SyncVAD
    monkeypatch.setitem(sys.modules, "sync_health_for_test", mod)
    p = tmp_path / "m.toml"
    p.write_text("[profiles.p]\n[profiles.p.vad]\nprovider='python:sync_health_for_test:SyncVAD'\n"
                 "[profiles.p.asd]\nprovider='python:sync_health_for_test:SyncVAD'\noptions={ok=false}\n")
    reg = load_registry("p", str(p))
    vad, asd = await reg.check(["vad", "asd"])
    assert vad.ok and vad.detail == "silero v5, cpu" and vad.latency_ms is not None
    assert not asd.ok and "healthy() is false" in asd.detail


def test_probe_model_for_openai_is_uncached_and_capped_at_one_token():
    reg = load_registry("dev-local")
    probe = reg._probe_model("llm")
    assert probe is not reg.strands_model("llm") and reg._probe_model("llm") is not probe
    assert probe.get_config()["params"]["max_tokens"] == 1
    assert reg.strands_model("llm").get_config()["params"]["max_tokens"] == 300
