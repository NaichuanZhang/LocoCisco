"""deploy/gb10 and deploy/usb: every script parses, shellchecks (when shellcheck is installed) and dry-runs
cleanly with no GPU, no docker daemon, no model servers and no network. Live checks need UNSPOKEN_LIVE=1."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
GB10 = ROOT / "deploy" / "gb10"
USB = ROOT / "deploy" / "usb"
SCRIPTS = sorted(GB10.glob("*.sh")) + sorted(USB.glob("*.sh"))
LIVE = os.environ.get("UNSPOKEN_LIVE") == "1"


def sh(args: list[str], env: dict[str, str] | None = None, cwd: Path = ROOT, ok: bool = True) -> subprocess.CompletedProcess:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("UNSPOKEN_", "DECIDER_", "VLLM_", "SPEECH_", "DRY_RUN"))}
    e.update(env or {})
    r = subprocess.run(args, cwd=cwd, env=e, capture_output=True, text=True, timeout=300)
    if ok and r.returncode != 0:
        raise AssertionError(f"{' '.join(str(a) for a in args)} -> {r.returncode}\n{r.stdout}\n{r.stderr}")
    return r


@pytest.fixture
def box(tmp_path: Path) -> dict[str, str]:
    """Environment that points every path the scripts touch into tmp_path (never the real home)."""
    return {
        "UNSPOKEN_HOME": str(tmp_path / "unspoken"), "UNSPOKEN_HF_HOME": str(tmp_path / "hf"),
        "UNSPOKEN_MODELS": str(tmp_path / "models"), "UNSPOKEN_STATE": str(tmp_path / "state"),
        "UNSPOKEN_LOGS": str(tmp_path / "logs"), "HOME": str(tmp_path),
    }


# ----------------------------------------------------------------------------- static
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_bash_syntax(script: Path) -> None:
    sh(["bash", "-n", str(script)])
    assert os.access(script, os.X_OK) or script.name == "lib.sh", f"{script.name} is not executable"


@pytest.mark.skipif(not (shutil.which("shellcheck") or os.environ.get("SHELLCHECK")), reason="shellcheck not installed")
def test_shellcheck() -> None:
    sc = os.environ.get("SHELLCHECK") or shutil.which("shellcheck")
    sh([sc, "-x", "-S", "warning", *map(str, SCRIPTS)])


def test_everything_binds_loopback() -> None:
    for script in SCRIPTS:
        text = script.read_text()
        assert "0.0.0.0" not in text, f"{script.name} mentions 0.0.0.0"
        assert "hackathon" not in text.lower(), f"{script.name} uses the forbidden infrastructure word"


def test_no_secrets_in_tree() -> None:
    for p in list(GB10.rglob("*")) + list(USB.rglob("*")):
        if p.is_file() and p.suffix in {".sh", ".py", ".txt", ".json"}:
            assert not re.search(r"hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|xox[bp]-", p.read_text()), p


# ----------------------------------------------------------------------------- gb10 dry runs
def test_decider_12b_docker_dry_run(box: dict[str, str]) -> None:
    r = sh([str(GB10 / "decider.sh"), "12b", "--dry-run"], env=box)
    out = r.stdout + r.stderr
    for needle in ("docker run", "--gpus all", "--network host", "nvcr.io/nvidia/vllm:26.05.post1-py3",
                   "strands-decider serve", "--device cuda", "--port 8765", "--host 127.0.0.1",
                   "HF_HUB_OFFLINE=1", "models--StrandsAgents--strands-decider-12B-gemma4-v1-2610",
                   "pip install --user --break-system-packages --no-index --no-deps --find-links /wheels"):
        assert needle in out, f"missing {needle!r} in\n{out}"
    assert "cuda=" in out  # the torch sanity line in the entry script


def test_decider_2b_native_cpu_dry_run(box: dict[str, str]) -> None:
    env = dict(box, DECIDER_RUNTIME="native", DECIDER_DEVICE="cpu", DECIDER_PORT="18765", DECIDER_BIN="/nonexistent/strands-decider")
    r = sh([str(GB10 / "decider.sh"), "2b", "--dry-run"], env=env)
    out = r.stdout + r.stderr
    assert "--device cpu" in out and "--port 18765" in out and "nohup" in out
    assert "docker" not in out.split("+ nohup", 1)[1]


def test_decider_rejects_bad_size(box: dict[str, str]) -> None:
    r = sh([str(GB10 / "decider.sh"), "7b", "--dry-run"], env=box, ok=False)
    assert r.returncode == 2


def test_speech_dry_run(box: dict[str, str]) -> None:
    r = sh([str(GB10 / "speech.sh"), "all", "--dry-run"], env=box)
    out = r.stdout + r.stderr
    for needle in ("nvcr.io/nvidia/nemo-speech.cpp:0.2.0", "--port 8770", "--port 8771", "--host 127.0.0.1",
                   "parakeet-tdt-0.6b-v3.q8_0.gguf", "--diar-model", "Nemotron-3-Diarization.q8_0.gguf",
                   "--tts-model", "magpie_tts_multilingual_357m.v2607.q8_0.gguf", "--codec-model", "--tokenizer-dir", ":/models:ro"):
        assert needle in out, f"missing {needle!r} in\n{out}"
    r = sh([str(GB10 / "speech.sh"), "asr", "--dry-run"], env=dict(box, ASR_MODEL="nemotron35", SPEECH_DIAR="0"))
    out = r.stdout + r.stderr
    assert "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf" in out and "--diar-model" not in out


def test_vllm_dry_run(box: dict[str, str]) -> None:
    r = sh([str(GB10 / "vllm.sh"), "--dry-run"], env=box)
    out = r.stdout + r.stderr
    for needle in ("nvcr.io/nvidia/vllm:26.05.post1-py3", "vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4", "--host 127.0.0.1 --port 8000",
                   "--tool-call-parser qwen3_xml", "--reasoning-parser qwen3", "--enable-auto-tool-choice",
                   "--gpu-memory-utilization 0.40", "--max-num-seqs 4", "--kv-cache-dtype fp8", "--speculative-config", "--restart always"):
        assert needle in out, f"missing {needle!r} in\n{out}"
    r = sh([str(GB10 / "vllm.sh"), "--dry-run"], env=dict(box, VLLM_MTP="0", VLLM_GPU_UTIL="0.35"))
    out = r.stdout + r.stderr
    assert "--speculative-config" not in out and "--gpu-memory-utilization 0.35" in out


@pytest.mark.parametrize("args", [["up.sh", "--with-2b"], ["down.sh"], ["status.sh"], ["check.sh"], ["00-preflight.sh"], ["down.sh", "--all"]],
                         ids=lambda a: " ".join(a))
def test_orchestration_dry_runs(box: dict[str, str], args: list[str]) -> None:
    r = sh([str(GB10 / args[0]), *args[1:], "--dry-run"], env=box)
    out = r.stdout + r.stderr
    if args[0] == "up.sh":
        for step in ("1/6 vLLM", "2/6 decider 12B", "3/6 ASR", "4/6 TTS", "5/6 decider 2B", "6/6 check"):
            assert step in out
        assert "unspoken.models check" in out and "UNSPOKEN_PROFILE=gb10" in out
        assert "unspoken.sensors.check" in out and "HF_HUB_OFFLINE=1" in out
    if args[0] == "status.sh":
        assert "unspoken-decider-12b" in out and ":8765" in out


def test_up_creates_no_files_in_dry_run(box: dict[str, str], tmp_path: Path) -> None:
    sh([str(GB10 / "up.sh"), "--dry-run"], env=box)
    assert not (tmp_path / "logs").exists() and not (tmp_path / "state").exists()


# ----------------------------------------------------------------------------- usb
def test_usbkit_plan_matches_lock() -> None:
    plan = sh([sys.executable, str(USB / "usbkit.py"), "plan"]).stdout.strip().splitlines()
    rows = [line.split("\t") for line in plan]
    keys = [r[0] for r in rows]
    dests = [r[3] for r in rows]
    assert len(set(keys)) == len(keys) and len(set(dests)) == len(dests)
    lock = json.loads((USB / "sizes.lock.json").read_text())
    assert set(lock["items"]) == set(keys), "sizes.lock.json is stale: python3 deploy/usb/usbkit.py sizes --write-lock"
    for r in rows:
        assert r[2] in {"image", "hf", "hf-files", "url", "git", "pip", "apt"}
        assert r[4] in {"0", "1"}
    # the pins the GB10 scripts use must be on the USB
    assert "nvcr.io/nvidia/vllm:26.05.post1-py3" in plan[0] or any("vllm:26.05.post1-py3" in line for line in plan)
    assert any("nvidia/Qwen3.6-35B-A3B-NVFP4" in line for line in plan)
    assert any("google/gemma-4-12B-it" in line for line in plan)


def test_usbkit_sizes_offline() -> None:
    r = sh([sys.executable, str(USB / "usbkit.py"), "sizes", "--offline"])
    assert "TOTAL" in r.stdout and "drive: 256 GB" in r.stdout
    m = re.search(r"TOTAL\s+([\d.]+) GiB\s+([\d.]+) GiB", r.stdout)
    assert m, r.stdout
    download, on_usb = float(m.group(1)), float(m.group(2))
    assert 60 < download < 120 and download <= on_usb < 200
    r = sh([sys.executable, str(USB / "usbkit.py"), "sizes", "--offline", "--group", "sensors"])
    assert "wespeaker-campp" in r.stdout and "apt-audio" in r.stdout and "image-vllm" not in r.stdout


def test_prepare_dry_run_offline(tmp_path: Path) -> None:
    env = {"UNSPOKEN_STAGE": str(tmp_path / "stage"), "HOME": str(tmp_path)}
    r = sh([str(USB / "prepare.sh"), str(tmp_path / "usb"), "--dry-run", "--offline"], env=env)
    out = r.stdout + r.stderr
    for needle in ("TOTAL", "docker pull --platform linux/arm64 nvcr.io/nvidia/vllm:26.05.post1-py3", "hf download nvidia/Qwen3.6-35B-A3B-NVFP4",
                   "hf download google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
                   "pip download --no-deps", "strands-decider @ git+https://github.com/strands-labs/strands-decider@930e6d5",
                   "git clone --depth 1 --branch lkg https://github.com/NVIDIA/NemoClaw.git", "bundle create", "finished: 28 done, 0 skipped, 0 failed"):
        assert needle in out, f"missing {needle!r}"
    assert not (tmp_path / "usb").exists(), "dry run must not create the USB dir"
    r = sh([str(USB / "prepare.sh"), str(tmp_path / "usb"), "--dry-run", "--offline", "--group=sensors", "--skip-optional"], env=env)
    out = r.stdout + r.stderr
    assert "docker pull" not in out and "talkset" not in out.lower() and "yunet" not in out
    for needle in ("wespeaker_en_voxceleb_CAM++.onnx", "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef",
                   "hf download Systran/faster-whisper-base.en --revision 3d3d5dee", "hf download Systran/faster-whisper-small.en",
                   "apt-resolve libportaudio2 alsa-utils", "lightasd_pretrain_AVA_CVPR.model",
                   "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317",
                   "pip wheel --no-deps", "python_speech_features==0.6", "finished: 7 done"):
        assert needle in out, f"missing {needle!r}"


def test_restore_dry_run(tmp_path: Path, box: dict[str, str]) -> None:
    usb = tmp_path / "usb"
    for rel in ("images/vllm-26.05.post1-py3.tar.zst", "images/nemo-speech.cpp-0.2.0.tar", "hf/nvidia--Qwen3.6-35B-A3B-NVFP4.tar",
                "speech/nvidia/magpie_tts_multilingual_357m/magpie_tts_multilingual_357m.nemo", "sensors/wespeaker_en_voxceleb_CAM++.onnx",
                "apt/libportaudio2_19.6.0-1.2build3_arm64.deb",
                "wheels/unspoken/x.whl", "tools/uv-0.13.5-aarch64-unknown-linux-gnu.tar.gz",
                "tools/nemo-speech-0.2.0-linux-aarch64-cuda13.tar.gz", "tools/nemoclaw.sh", "repo/nemoclaw-lkg.tar", "repo/unspoken.bundle", "MANIFEST.sha256"):
        (usb / rel).parent.mkdir(parents=True, exist_ok=True)
        (usb / rel).write_bytes(b"")
    (usb / "images/vllm-26.05.post1-py3.ref").write_text("nvcr.io/nvidia/vllm:26.05.post1-py3\n")
    (usb / "apt/PACKAGES").write_text("libportaudio2 alsa-utils\n")
    r = sh([str(USB / "restore.sh"), str(usb), "--dry-run", "--native-decider"], env=box)
    out = r.stdout + r.stderr
    assert ".cache/unspoken/models/wespeaker_en_voxceleb_CAM++.onnx" in out
    assert "libportaudio2" in out and ("sudo apt-get install" in out or "skipping libportaudio2 alsa-utils" in out or "already installed" in out)
    for needle in ("zstd -dc", "docker load", "tar -C", "nvidia--Qwen3.6-35B-A3B-NVFP4.tar", "magpie_tts_multilingual_357m.nemo -C",
                   "uv-0.13.5-aarch64", "nemo-speech-0.2.0-linux-aarch64-cuda13", "nemoclaw-lkg.tar", "git clone --branch hack/main",
                   "--offline --no-index --find-links", "decider-venv", "torch-cu130"):
        assert needle in out, f"missing {needle!r} in\n{out}"
    assert not Path(box["UNSPOKEN_HOME"]).exists()


def test_restore_runs_from_usb_bin_copy(tmp_path: Path, box: dict[str, str]) -> None:
    """prepare.sh copies restore.sh + gb10/ into USB/bin so the box can run it before the repo exists."""
    usb = tmp_path / "usb"
    (usb / "bin" / "gb10").mkdir(parents=True)
    shutil.copy(USB / "restore.sh", usb / "bin" / "restore.sh")
    for s in GB10.glob("*.sh"):
        shutil.copy(s, usb / "bin" / "gb10" / s.name)
    (usb / "MANIFEST.sha256").write_text("")
    r = sh(["bash", str(usb / "bin" / "restore.sh"), str(usb), "--dry-run"], env=box)
    assert "restore done" in r.stdout + r.stderr


def test_restore_writes_refs_main_and_installs_sensors_model(tmp_path: Path, box: dict[str, str]) -> None:
    """Real (not dry) restore of an HF tar without refs/ (pinned download) + the CAM++ file: refs/main must point at
    the snapshot (offline loaders resolve "main") and the ONNX must land where unspoken.sensors.assets looks."""
    import hashlib
    import tarfile
    stage = tmp_path / "stage" / "hub"
    snap = stage / "models--Systran--faster-whisper-base.en" / "snapshots" / "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    usb = tmp_path / "usb"
    (usb / "hf").mkdir(parents=True)
    with tarfile.open(usb / "hf" / "Systran--faster-whisper-base.en.tar", "w") as t:
        t.add(stage / "models--Systran--faster-whisper-base.en", arcname="models--Systran--faster-whisper-base.en")
    (usb / "sensors").mkdir()
    onnx = usb / "sensors" / "wespeaker_en_voxceleb_CAM++.onnx"
    onnx.write_bytes(b"not the real model")
    (usb / "MANIFEST.sha256").write_text("")
    env = dict(box, UNSPOKEN_MODELS_DIR=str(tmp_path / "sensor-models"))
    r = sh([str(USB / "restore.sh"), str(usb), "--skip-images", "--skip-apt"], env=env)
    refs = Path(box["UNSPOKEN_HF_HOME"]) / "hub" / "models--Systran--faster-whisper-base.en" / "refs" / "main"
    assert refs.read_text() == "3d3d5dee26484f91867d81cb899cfcf72b96be6c"
    # wrong bytes -> sha256 mismatch -> refused and removed, with the fetch_models hint
    assert not (tmp_path / "sensor-models" / onnx.name).exists()
    assert "sha256 mismatch" in r.stderr and "fetch_models" in r.stderr
    assert hashlib.sha256(b"not the real model").hexdigest() not in (USB / "usbkit.py").read_text()


def test_usb_sensor_pins_match_the_sensors_code() -> None:
    """The USB must carry exactly what unspoken.sensors loads: the CAM++ URL + sha256 from assets.PINNED, the
    faster-whisper DEFAULT_MODEL, the Light-ASD AVA weights the transcriber loads (URL + sha256 from
    unspoken.transcribe.asd_lightasd.WEIGHTS), and nothing the repo already vendors (Silero, YuNet)."""
    assets_py = ROOT / "unspoken" / "sensors" / "assets.py"
    if not assets_py.exists():
        pytest.skip("unspoken.sensors not in this checkout")
    from unspoken.sensors import assets
    plan = [line.split("\t") for line in sh([sys.executable, str(USB / "usbkit.py"), "plan"]).stdout.splitlines()]
    urls = {r[3].rsplit("/", 1)[-1]: (r[5], r[6]) for r in plan if r[2] == "url" and r[3].startswith("sensors/")}
    url, sha = assets.PINNED["wespeaker_en_voxceleb_CAM++.onnx"]
    assert urls["wespeaker_en_voxceleb_CAM++.onnx"] == (url, sha)
    for vendored in ("silero_vad.onnx", "face_detection_yunet_2023mar.onnx"):
        assert (assets.VENDORED_DIR / vendored).exists() and vendored not in urls
    from unspoken.transcribe.asd_lightasd import DEFAULT_WEIGHTS, WEIGHTS
    lurl, lsha = WEIGHTS[DEFAULT_WEIGHTS]
    assert urls[DEFAULT_WEIGHTS] == (lurl, lsha)
    assert not any("TalkSet" in r[5] for r in plan)          # research-licence weights never ship
    default = re.search(r'DEFAULT_MODEL = "([^"]+)"', (ROOT / "unspoken" / "sensors" / "asr_local.py").read_text()).group(1)
    assert any(r[2] == "hf" and r[5] == default for r in plan), default
    apt = next(r for r in plan if r[2] == "apt")
    assert set(apt[5].split(",")) == {"libportaudio2", "alsa-utils"}


def test_apt_resolver_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """apt_resolve on a tiny fake noble index: first satisfiable alternative, virtual packages via Provides,
    Priority required/important deps skipped, sha256/size/url taken from the stanza."""
    import importlib.util
    import lzma
    spec = importlib.util.spec_from_file_location("usbkit", USB / "usbkit.py")
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    stanzas = [
        ("libportaudio2", "optional", "libasound2t64 (>= 1.0.16), libc6 (>= 2.34), libjack-jackd2-0 (>= 1.9) | libjack-0.125", ""),
        ("libasound2t64", "optional", "libasound2-data", ""),
        ("libasound2-data", "optional", "", ""),
        ("libjack-jackd2-0", "optional", "", ""),
        ("libc6", "required", "", ""),
        ("alsa-utils", "optional", "kmod (>= 17), libasound2t64, fake-virtual", ""),
        ("kmod", "important", "", ""),
        ("real-impl", "optional", "", "fake-virtual"),
    ]
    text = "\n\n".join(
        f"Package: {n}\nVersion: 1.0\nPriority: {pr}\n" + (f"Depends: {dep}\n" if dep else "") + (f"Provides: {pv}\n" if pv else "")
        + f"Filename: pool/x/{n}_1.0_arm64.deb\nSize: 10\nSHA256: {'0' * 63}{len(n) % 10}" for n, pr, dep, pv in stanzas) + "\n"
    blob = lzma.compress(text.encode())
    monkeypatch.setattr(kit, "_get", lambda url, headers=None, timeout=60: blob)
    got = {d["package"]: d for d in kit.apt_resolve(["libportaudio2", "alsa-utils"])}
    assert set(got) == {"libportaudio2", "libasound2t64", "libasound2-data", "libjack-jackd2-0", "alsa-utils", "real-impl"}
    assert got["libportaudio2"]["url"] == "http://ports.ubuntu.com/ubuntu-ports/pool/x/libportaudio2_1.0_arm64.deb"
    assert got["alsa-utils"]["size"] == 10 and len(got["alsa-utils"]["sha256"]) == 64


# ----------------------------------------------------------------------------- docs
def test_models_toml_gb10_matches_the_deploy_ports() -> None:
    """The gb10 profile in config/models.toml (the source of truth HACKDAY.md points at) uses the ports the
    deploy scripts serve, loopback only, and names sensors factories that exist."""
    import tomllib
    gb10 = tomllib.loads((ROOT / "config" / "models.toml").read_text())["profiles"]["gb10"]
    assert gb10.get("local_only") is True
    assert gb10["decider"]["url"] == "http://127.0.0.1:8765" and gb10["decider"]["provider"] == "strands-decider"
    assert gb10["llm"]["url"] == "http://127.0.0.1:8000/v1" and gb10["llm"]["model"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
    assert gb10["asr"]["url"] == "http://127.0.0.1:8770/v1" and gb10["asr"]["provider"] == "openai-audio"
    assert gb10["tts"]["url"] == "http://127.0.0.1:8771/v1" and gb10["tts"]["provider"] == "openai-speech"
    for role, block in gb10.items():
        if isinstance(block, dict) and "url" in block:
            assert block["url"].startswith("http://127.0.0.1:"), role
    providers = {r: gb10[r]["provider"] for r in ("vad", "diarization", "faces", "asd")}
    # the winners of the per-person transcriber test (docs/TRANSCRIBE.md); Light-ASD on the GPU needs wheels/torch-cu130
    assert providers == {"vad": "python:unspoken.sensors.vad:SileroVAD",
                         "diarization": "python:unspoken.transcribe.diar:OnlineWindowDiarizer",
                         "faces": "python:unspoken.sensors.faces:YuNetFaces",
                         "asd": "python:unspoken.transcribe.asd_live:LightASDLive"}
    assert gb10["asd"]["options"]["device"] == "cuda"
    fallback = "python:unspoken.sensors.asr_local:FasterWhisperASR"
    assert fallback in (ROOT / "docs" / "HACKDAY.md").read_text()      # the CPU ASR fallback is documented
    for prov in list(providers.values()) + [fallback]:
        mod, _, cls = prov.removeprefix("python:").partition(":")
        assert f"class {cls}" in (ROOT / (mod.replace(".", "/") + ".py")).read_text(), prov


def test_hackday_has_the_runbook_parts() -> None:
    text = (ROOT / "docs" / "HACKDAY.md").read_text()
    for needle in ("00-preflight.sh", "restore.sh", "up.sh", "check.sh", "Memory budget", "USB", "3-minute demo", "nemoclaw",
                   "unspoken.sensors.check", "libportaudio2", "wespeaker_en_voxceleb_CAM++.onnx"):
        assert needle in text, needle


# ----------------------------------------------------------------------------- live
@pytest.mark.skipif(not LIVE, reason="UNSPOKEN_LIVE=1 for network checks")
def test_prepare_dry_run_online_sizes(tmp_path: Path) -> None:
    r = sh([sys.executable, str(USB / "usbkit.py"), "sizes", "--group", "core", "--group", "speech"])
    assert "[online]" in r.stdout and "UNKNOWN" not in r.stdout


def test_download_md_lists_every_usb_item() -> None:
    """docs/DOWNLOAD.md mirrors deploy/usb/usbkit.py: every item key, every repo / URL / image, and every pin."""
    sys.path.insert(0, str(USB))
    import usbkit
    text = (ROOT / "docs" / "DOWNLOAD.md").read_text()
    for it in usbkit.ITEMS:
        assert it["key"] in text, it["key"]
        for field in ("repo", "url", "rev", "sha256"):
            if it.get(field):
                assert it[field] in text, (it["key"], field)
        if it["kind"] == "image":
            assert it["ref"] in text, it["key"]
        for f in it.get("files", []):
            assert f.rsplit(".", 2)[0] in text, (it["key"], f)
