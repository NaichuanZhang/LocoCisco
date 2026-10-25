#!/usr/bin/env python3
"""USB packing list for the LocoCisco GB10 deploy, and its size planner. Standard library only (runs on the
laptop's stock python3, macOS or Linux, 3.9+).

    python3 usbkit.py plan [--group G]...        TSV: key, group, kind, dest, optional, spec...   (prepare.sh reads it)
    python3 usbkit.py sizes [--offline] [--lock sizes.lock.json] [--write-lock] [--pip PYTHON] [--group G]...
                                                 one line per item with its size, group subtotals, the total,
                                                 and the drive to buy. Online: HF API, registry manifests, PyPI,
                                                 HEAD requests; --offline (or no network) reads the lock.

Kinds
  image     docker image, saved with `docker save` (size here = compressed registry layers for linux/arm64;
            the .tar on the USB is ~2.2x bigger unless zstd is available, then ~1.0x)
  hf        a whole Hugging Face repo (optionally pinned revision) -> hf/<repo>.tar of its HF-cache directory
  hf-files  named files from a Hugging Face repo -> speech/<repo>/<file>
  url       one file -> tools/...
  pip       a wheelhouse for aarch64 / cp312 -> wheels/<group>/
  git       a git bundle -> repo/<name>.bundle
  apt       Ubuntu 24.04 (noble) arm64 .deb files for named packages + their dependencies, resolved from
            ports.ubuntu.com (no docker, no emulation) -> apt/ ; restore.sh installs only what the box lacks
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_LOCK = os.path.join(HERE, "sizes.lock.json")

# Pinned sources (see docs/HACKDAY.md for why each one).
STRANDS_DECIDER_GIT = "https://github.com/strands-labs/strands-decider"
STRANDS_DECIDER_REF = "930e6d5de16309be32579f5631a7369a3322140c"   # v0.1.0-8-g930e6d5, the dev-desktop commit
NEMOCLAW_GIT = "https://github.com/NVIDIA/NemoClaw.git"
NEMOCLAW_REF = "lkg"                                                 # NemoClaw's "last known good" tag
UV_VERSION = "0.13.5"
NEMO_SPEECH_VERSION = "0.2.0"
TORCH_CU130 = "torch==2.14.1+cu130"                                  # https://download.pytorch.org/whl/cu130 (aarch64 wheel: 454 MB)
# pip's --platform takes exact tags (no glibc fallback), so list every manylinux the box (glibc 2.39) accepts.
PIP_PLATFORMS = [f"manylinux_2_{v}_aarch64" for v in range(39, 16, -1)] + ["manylinux2014_aarch64", "linux_aarch64"]
PIP_TARGET_FLAGS = sum((["--platform", t] for t in PIP_PLATFORMS), []) + ["--python-version", "3.12", "--implementation", "cp", "--abi", "cp312", "--only-binary=:all:"]

# Each item: key, group, kind, dest (relative to the USB root), optional, spec (kind-specific).
# Groups: core (must), speech, sensors, fallback (the no-docker / second-torso plan B), extras.
ITEMS: list[dict] = [
    # ---- docker images (linux/arm64; digests verified against nvcr.io on 2026-10-24)
    {"key": "image-vllm", "group": "core", "kind": "image", "optional": False,
     "ref": "nvcr.io/nvidia/vllm:26.05.post1-py3", "dest": "images/vllm-26.05.post1-py3.tar",
     "note": "NemoClaw's managed vLLM image; also the decider's torch"},
    {"key": "image-speech", "group": "speech", "kind": "image", "optional": False,
     "ref": "nvcr.io/nvidia/nemo-speech.cpp:0.2.0", "dest": "images/nemo-speech.cpp-0.2.0.tar",
     "note": "ASR + diarization + TTS runtime (C++, CUDA 13 aarch64)"},
    {"key": "image-pytorch", "group": "fallback", "kind": "image", "optional": True,
     "ref": "nvcr.io/nvidia/pytorch:26.09-py3", "dest": "images/pytorch-26.09-py3.tar",
     "note": "second torso for the decider if the vLLM image's torch misbehaves"},
    # ---- Hugging Face repos, whole cache directories
    {"key": "hf-qwen36", "group": "core", "kind": "hf", "optional": False,
     "repo": "nvidia/Qwen3.6-35B-A3B-NVFP4", "rev": None, "dest": "hf/nvidia--Qwen3.6-35B-A3B-NVFP4.tar",
     "note": "main LLM (vLLM :8000)"},
    {"key": "hf-decider-12b", "group": "core", "kind": "hf", "optional": False,
     "repo": "StrandsAgents/strands-decider-12B-gemma4-v1-2610", "rev": None,
     "dest": "hf/StrandsAgents--strands-decider-12B-gemma4-v1-2610.tar", "note": "decider adapter (:8765)"},
    {"key": "hf-gemma4-12b", "group": "core", "kind": "hf", "optional": False,
     "repo": "google/gemma-4-12B-it", "rev": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
     "dest": "hf/google--gemma-4-12B-it.tar", "note": "base of the 12B decider (GATED: needs HF_TOKEN with Gemma access)"},
    {"key": "hf-decider-2b", "group": "core", "kind": "hf", "optional": False,
     "repo": "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610", "rev": "216d1ccb46a9e9b036eb9187b248db98480f77d4",
     "dest": "hf/StrandsAgents--strands-decider-2B-qwen3.5-v1-2610.tar", "note": "decider 2B adapter (:8766, optional server)"},
    {"key": "hf-qwen35-2b", "group": "core", "kind": "hf", "optional": False,
     "repo": "Qwen/Qwen3.5-2B-Base", "rev": "b1485b2fa6dfa1287294f269f5fb618e03d52d7c",
     "dest": "hf/Qwen--Qwen3.5-2B-Base.tar", "note": "base of the 2B decider"},
    # ---- speech models (GGUF files the NeMo-Speech.cpp runtime loads by path)
    {"key": "speech-parakeet", "group": "speech", "kind": "hf-files", "optional": False,
     "repo": "nvidia/parakeet-tdt-0.6b-v3", "files": ["parakeet-tdt-0.6b-v3.q8_0.gguf"],
     "dest": "speech/nvidia/parakeet-tdt-0.6b-v3", "note": "ASR (offline, 25 languages)"},
    {"key": "speech-diar", "group": "speech", "kind": "hf-files", "optional": False,
     "repo": "nvidia/Nemotron-3-Diarization", "files": ["Nemotron-3-Diarization.q8_0.gguf"],
     "dest": "speech/nvidia/Nemotron-3-Diarization", "note": "speaker tags, <= 8 speakers"},
    {"key": "speech-nemotron35", "group": "speech", "kind": "hf-files", "optional": True,
     "repo": "nvidia/nemotron-3.5-asr-streaming-0.6b", "files": ["nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"],
     "dest": "speech/nvidia/nemotron-3.5-asr-streaming-0.6b", "note": "streaming multilingual ASR alternative"},
    {"key": "speech-magpie", "group": "speech", "kind": "hf-files", "optional": True,
     "repo": "nvidia/magpie_tts_multilingual_357m",
     "files": ["magpie_tts_multilingual_357m.v2607.q8_0.gguf", "magpie_tts_multilingual_357m.nemo"],
     "dest": "speech/nvidia/magpie_tts_multilingual_357m", "note": "TTS (:8771); the .nemo holds the tokenizer (restore.sh extracts it)"},
    {"key": "speech-nanocodec", "group": "speech", "kind": "hf-files", "optional": True,
     "repo": "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps",
     "files": ["nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"],
     "dest": "speech/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps", "note": "TTS codec"},
    {"key": "hf-kokoro", "group": "extras", "kind": "hf", "optional": True,
     "repo": "hexgrad/Kokoro-82M", "rev": None, "dest": "hf/hexgrad--Kokoro-82M.tar",
     "note": "alternative TTS voice (Kokoro-FastAPI), not scripted"},
    # ---- sensors + transcriber (unspoken/sensors, unspoken/transcribe). Silero VAD + YuNet ONNX are vendored in the
    # repo. Pins below must match unspoken/sensors/assets.py and unspoken/transcribe/asd_lightasd.py (tests check it).
    {"key": "wespeaker-campp", "group": "sensors", "kind": "url", "optional": False,
     "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++.onnx",
     "sha256": "c46fad10b5f81e1aa4a60c162714208577093655076c5450f8c469e522ec54ef",
     "dest": "sensors/wespeaker_en_voxceleb_CAM++.onnx",
     "note": "speaker embeddings for diarization; restore.sh puts it in ~/.cache/unspoken/models"},
    {"key": "hf-fw-base-en", "group": "sensors", "kind": "hf", "optional": False,
     "repo": "Systran/faster-whisper-base.en", "rev": "3d3d5dee26484f91867d81cb899cfcf72b96be6c",
     "dest": "hf/Systran--faster-whisper-base.en.tar", "note": "CPU ASR default (FasterWhisperASR DEFAULT_MODEL)"},
    {"key": "hf-fw-small-en", "group": "sensors", "kind": "hf", "optional": False,
     "repo": "Systran/faster-whisper-small.en", "rev": "d1d751a5f8271d482d14ca55d9e2deeebbae577f",
     "dest": "hf/Systran--faster-whisper-small.en.tar", "note": "CPU ASR, more robust on real room audio; ~3x base.en compute"},
    {"key": "lightasd-ava", "group": "sensors", "kind": "url", "optional": False,
     "url": "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/ed38c232de5efe0261dbd68627c0ade7cdfe14eb/weight/pretrain_AVA_CVPR.model",
     "sha256": "d44bc3ea7baa8e0946fa3921311714a630ed8b90a1928fab0dbe30d918909317",
     "dest": "sensors/lightasd_pretrain_AVA_CVPR.model",
     "note": "Light-ASD AVA weights (MIT, 4.2 MB): who is talking on camera; restore.sh puts it in ~/.cache/unspoken/models"},
    {"key": "apt-audio", "group": "sensors", "kind": "apt", "optional": False,
     "packages": ["libportaudio2", "alsa-utils"], "dest": "apt",
     "note": "PortAudio for sounddevice + arecord fallback; Ubuntu noble arm64 .debs with dependencies"},
    # ---- tools
    {"key": "uv", "group": "core", "kind": "url", "optional": False,
     "url": f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz",
     "dest": f"tools/uv-{UV_VERSION}-aarch64-unknown-linux-gnu.tar.gz", "note": "venvs offline on the box"},
    {"key": "nemo-speech-native", "group": "fallback", "kind": "url", "optional": True,
     "url": f"https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v{NEMO_SPEECH_VERSION}/nemo-speech-{NEMO_SPEECH_VERSION}-linux-aarch64-cuda13.tar.gz",
     "dest": f"tools/nemo-speech-{NEMO_SPEECH_VERSION}-linux-aarch64-cuda13.tar.gz", "note": "speech runtime without docker"},
    {"key": "nemoclaw-installer", "group": "core", "kind": "url", "optional": False,
     "url": "https://www.nvidia.com/nemoclaw.sh", "dest": "tools/nemoclaw.sh", "note": "bootstrap installer (clones the ref below)"},
    {"key": "nemoclaw-repo", "group": "core", "kind": "git", "optional": False, "shallow": True,
     "git": NEMOCLAW_GIT, "ref": NEMOCLAW_REF, "dest": "repo/nemoclaw-lkg.tar",
     "note": "shallow clone of NemoClaw at the lkg tag, tarred; its install.sh runs scripts/install.sh from the checkout"},
    {"key": "unspoken-repo", "group": "core", "kind": "git", "optional": False,
     "git": REPO_ROOT, "ref": "HEAD", "dest": "repo/unspoken.bundle", "note": "this repository, every branch"},
    # ---- wheels (aarch64, cp312)
    {"key": "wheels-decider", "group": "core", "kind": "pip", "optional": False, "deps": False,
     "requirements": [os.path.join(HERE, "requirements-decider.txt")], "index": None,
     "extra": [f"strands-decider @ git+{STRANDS_DECIDER_GIT}@{STRANDS_DECIDER_REF}"],
     "dest": "wheels/decider", "note": "strands-decider + deps, no torch (the image's torch is used)"},
    {"key": "wheels-unspoken", "group": "core", "kind": "pip", "optional": False, "deps": True,
     "requirements": "repo-requirements", "index": None, "extra": [],
     "dest": "wheels/unspoken", "note": "requirements-dev.txt + requirements/*.txt of this repo, resolved for aarch64"},
    {"key": "wheels-torch-cu130", "group": "sensors", "kind": "pip", "optional": False, "deps": True,
     "requirements": [], "index": "https://download.pytorch.org/whl/cu130", "extra": [TORCH_CU130],
     "dest": "wheels/torch-cu130",
     "note": "torch for the repo venv (Light-ASD on the GPU) and the no-docker decider fallback, ~3-4 GB with CUDA libs"},
    {"key": "wheels-sdist", "group": "sensors", "kind": "pip", "optional": False, "deps": False,
     "requirements": [], "index": None, "extra": ["python_speech_features==0.6"],
     "dest": "wheels/sdist", "note": "pure-python wheels built from sdist-only pins (Light-ASD's MFCC front end)"},
]

APT_MIRROR = "http://ports.ubuntu.com/ubuntu-ports"
APT_SUITES = ("noble-security", "noble-updates", "noble")      # first hit wins (newest)
APT_COMPONENTS = ("main", "universe")
APT_SKIP_PRIORITIES = {"required", "important"}               # on every Ubuntu install; never shipped

IMAGE_TAR_FACTOR = 2.2       # docker save (uncompressed) vs registry compressed layers, NGC CUDA images
GiB = 1024 ** 3


def repo_requirement_files() -> list[str]:
    files = [os.path.join(REPO_ROOT, "requirements-dev.txt")]
    rdir = os.path.join(REPO_ROOT, "requirements")
    if os.path.isdir(rdir):
        files += sorted(os.path.join(rdir, f) for f in os.listdir(rdir) if f.endswith(".txt"))
    return [f for f in files if os.path.isfile(f)]


def items(groups: list[str] | None) -> list[dict]:
    return [i for i in ITEMS if not groups or i["group"] in groups]


# ----------------------------------------------------------------------------- network helpers
def _get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _json(url: str, headers: dict | None = None) -> object:
    return json.loads(_get(url, headers).decode())


def hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def hf_tree(repo: str, rev: str | None) -> list[dict]:
    rev = rev or "main"
    url = f"https://huggingface.co/api/models/{repo}/tree/{urllib.parse.quote(rev, safe='')}?recursive=true"
    data = _json(url, hf_headers())
    return [x for x in data if x.get("type") == "file"]


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            if n:
                return int(n)
    except urllib.error.HTTPError:
        pass
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        m = re.search(r"/(\d+)$", cr)
        if m:
            return int(m.group(1))
        n = r.headers.get("Content-Length")
        return int(n) if n else 0


def registry_arm64_size(ref: str) -> tuple[int, str]:
    """Compressed size of the linux/arm64 manifest of REF on nvcr.io / docker.io (anonymous pull token)."""
    host, _, rest = ref.partition("/")
    repo, _, tag = rest.partition(":")
    tag = tag or "latest"
    if host == "nvcr.io":
        tok = _json(f"https://nvcr.io/proxy_auth?scope=repository:{repo}:pull&service=registry")["token"]
    elif host in ("docker.io", "registry-1.docker.io"):
        host = "registry-1.docker.io"
        tok = _json(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull")["token"]
    else:
        raise RuntimeError(f"unsupported registry {host}")
    acc = ("application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.index.v1+json, "
           "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")
    hdr = {"Authorization": f"Bearer {tok}", "Accept": acc}
    top = _json(f"https://{host}/v2/{repo}/manifests/{tag}", hdr)
    digest = tag
    if "manifests" in top:
        arm = [m for m in top["manifests"] if m.get("platform", {}).get("architecture") == "arm64"]
        if not arm:
            raise RuntimeError(f"{ref}: no linux/arm64 manifest")
        digest = arm[0]["digest"]
        top = _json(f"https://{host}/v2/{repo}/manifests/{digest}", hdr)
    return sum(layer["size"] for layer in top["layers"]), digest


def pypi_wheel_size(name: str, version: str) -> int:
    d = _json(f"https://pypi.org/pypi/{name}/{version}/json")
    for f in d["urls"]:
        fn = f["filename"]
        if fn.endswith("py3-none-any.whl") or (re.search(r"(cp312|abi3)", fn) and "aarch64" in fn and "manylinux" in fn):
            return f["size"]
    raise RuntimeError(f"{name}=={version}: no aarch64/pure wheel on PyPI")


def pip_report_size(python: str, args: list[str], index: str | None) -> tuple[int, int]:
    """Resolve ARGS for linux/aarch64 cp312 with pip's --dry-run --report, then HEAD every wheel."""
    cmd = [python, "-m", "pip", "install", "--dry-run", "--quiet", "--report", "-", "--ignore-installed",
           "--target", os.path.join(os.environ.get("TMPDIR", "/tmp"), "usbkit-dry")] + PIP_TARGET_FLAGS
    if index:
        cmd += ["--index-url", index, "--extra-index-url", "https://pypi.org/simple"]
    out = subprocess.run(cmd + args, check=True, capture_output=True, text=True).stdout
    report = json.loads(out)
    total = 0
    for inst in report["install"]:
        total += head_size(inst["download_info"]["url"])
    return total, len(report["install"])


# ----------------------------------------------------------------------------- apt (Ubuntu noble arm64)
def _apt_index() -> tuple[dict[str, dict], dict[str, str]]:
    """Package name -> stanza (newest suite first), and virtual name -> providing package."""
    import lzma
    pkgs: dict[str, dict] = {}
    provides: dict[str, str] = {}
    for suite in APT_SUITES:
        for comp in APT_COMPONENTS:
            raw = lzma.decompress(_get(f"{APT_MIRROR}/dists/{suite}/{comp}/binary-arm64/Packages.xz", timeout=180)).decode("utf-8", "replace")
            for block in raw.split("\n\n"):
                st: dict[str, str] = {}
                key = None
                for line in block.splitlines():
                    if line.startswith(" ") and key:
                        st[key] += "\n" + line
                    elif ":" in line:
                        key, _, val = line.partition(":")
                        st[key] = val.strip()
                name = st.get("Package")
                if not name or name in pkgs:
                    continue
                pkgs[name] = st
                for prov in st.get("Provides", "").split(","):
                    pv = prov.strip().split(" ")[0]
                    if pv:
                        provides.setdefault(pv, name)
    return pkgs, provides


def apt_resolve(packages: list[str]) -> list[dict]:
    """Named packages + recursive Depends/Pre-Depends (first satisfiable alternative), skipping Priority
    required/important. Returns [{package, version, url, size, sha256}] sorted by name."""
    pkgs, provides = _apt_index()
    out: dict[str, dict] = {}
    todo = list(packages)
    while todo:
        name = todo.pop()
        name = name.split(":")[0]
        if name in out:
            continue
        st = pkgs.get(name) or pkgs.get(provides.get(name, ""))
        if st is None:
            raise RuntimeError(f"apt: {name} not in noble arm64 {'/'.join(APT_COMPONENTS)}")
        if st.get("Priority") in APT_SKIP_PRIORITIES and name not in packages:
            continue
        out[st["Package"]] = {"package": st["Package"], "version": st["Version"], "size": int(st["Size"]),
                              "sha256": st["SHA256"], "url": f"{APT_MIRROR}/{st['Filename']}"}
        for field in ("Pre-Depends", "Depends"):
            for dep in filter(None, (d.strip() for d in st.get(field, "").split(","))):
                alts = [a.strip().split(" ")[0].split(":")[0] for a in dep.split("|")]
                pick = next((a for a in alts if a in pkgs), None) or next((provides[a] for a in alts if a in provides), None)
                if pick is None:
                    raise RuntimeError(f"apt: cannot satisfy {dep!r} (needed by {name})")
                todo.append(pick)
    return [out[k] for k in sorted(out)]


def cmd_apt_resolve(a: argparse.Namespace) -> int:
    """TSV for prepare.sh: package, version, size, sha256, url."""
    for d in apt_resolve(a.packages):
        print("\t".join([d["package"], d["version"], str(d["size"]), d["sha256"], d["url"]]))
    return 0


# ----------------------------------------------------------------------------- sizes
def size_of(item: dict, pip_python: str | None) -> dict:
    """Returns {"bytes": int, "files": int, "detail": str} for one item, online."""
    k = item["kind"]
    if k == "image":
        n, digest = registry_arm64_size(item["ref"])
        return {"bytes": n, "files": 1, "detail": f"arm64 {digest[:19]} compressed; .tar ~{n * IMAGE_TAR_FACTOR / GiB:.1f} GiB plain"}
    if k == "hf":
        tree = hf_tree(item["repo"], item.get("rev"))
        return {"bytes": sum(x.get("size", 0) for x in tree), "files": len(tree), "detail": f"{len(tree)} files @ {item.get('rev') or 'main'}"}
    if k == "hf-files":
        tree = {x["path"]: x.get("size", 0) for x in hf_tree(item["repo"], None)}
        missing = [f for f in item["files"] if f not in tree]
        if missing:
            raise RuntimeError(f"{item['repo']}: files not in repo: {missing}")
        return {"bytes": sum(tree[f] for f in item["files"]), "files": len(item["files"]), "detail": ", ".join(item["files"])}
    if k == "url":
        pin = " (sha256 pinned)" if item.get("sha256") else ""
        return {"bytes": head_size(item["url"]), "files": 1, "detail": urllib.parse.unquote(item["url"].rsplit("/", 1)[-1]) + pin}
    if k == "apt":
        debs = apt_resolve(item["packages"])
        return {"bytes": sum(d["size"] for d in debs), "files": len(debs),
                "detail": f"{len(debs)} debs: " + " ".join(d["package"] for d in debs)}
    if k == "git":
        if item["git"].startswith("http"):
            # a bundle is roughly the packed repository; GitHub tells us the repo size in KB
            m = re.match(r"https://github.com/([^/]+)/([^/.]+)", item["git"])
            kb = _json(f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}")["size"] if m else 0
            if item.get("shallow"):
                return {"bytes": kb * 1024 // 4, "files": 1, "detail": f"{item['ref']} shallow clone (<= 1/4 of GitHub's {kb // 1024} MiB packed size)"}
            return {"bytes": kb * 1024, "files": 1, "detail": f"{item['ref']} (GitHub packed size)"}
        out = subprocess.run(["git", "-C", item["git"], "count-objects", "-v"], capture_output=True, text=True, check=True).stdout
        kb = sum(int(v) for key, v in (line.split(": ") for line in out.splitlines()) if key in ("size", "size-pack"))
        return {"bytes": kb * 1024, "files": 1, "detail": "local repo packed size"}
    if k == "pip":
        if not item["deps"]:
            total = 0
            n = 0
            for rf in item["requirements"]:
                for line in open(rf, encoding="utf-8"):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    name, ver = line.split("==")
                    total += pypi_wheel_size(name, ver)
                    n += 1
            total += 150_000  # the strands-decider wheel built from git
            return {"bytes": total, "files": n + len(item["extra"]), "detail": f"{n} pinned wheels + {len(item['extra'])} built from source ({', '.join(e.split(' @')[0] for e in item['extra'])})"}
        if not pip_python:
            raise RuntimeError("needs --pip PYTHON (a venv with pip) to resolve dependencies")
        args = []
        reqs = repo_requirement_files() if item["requirements"] == "repo-requirements" else item["requirements"]
        for rf in reqs:
            args += ["-r", rf]
        args += item["extra"]
        total, n = pip_report_size(pip_python, args, item["index"])
        return {"bytes": total, "files": n, "detail": f"{n} wheels resolved by pip for linux aarch64 cp312"}
    raise RuntimeError(f"unknown kind {k}")


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} TiB"


def cmd_sizes(a: argparse.Namespace) -> int:
    lock: dict = {}
    if os.path.isfile(a.lock):
        with open(a.lock, encoding="utf-8") as fh:
            lock = json.load(fh)
    rows = []
    unknown = 0
    for it in items(a.group):
        key = it["key"]
        entry = None
        if not a.offline:
            try:
                entry = size_of(it, a.pip)
                entry["source"] = "online"
            except Exception as e:  # noqa: BLE001 -- report and fall back to the lock
                print(f"  ! {key}: {type(e).__name__}: {e}", file=sys.stderr)
        if entry is None and key in lock.get("items", {}):
            entry = dict(lock["items"][key])
            entry["source"] = f"lock {lock.get('generated', '?')[:10]}"
        if entry is None:
            entry = {"bytes": 0, "files": 0, "detail": "UNKNOWN", "source": "-"}
            unknown += 1
        rows.append((it, entry))

    if not rows:
        print("no items match", file=sys.stderr)
        return 1
    width = max(len(r[0]["key"]) for r in rows) + 2
    print(f"{'item':<{width}} {'group':<9} {'kind':<9} {'size':>10}  {'on USB':>10}  detail")
    totals: dict[str, float] = {}
    usb_totals: dict[str, float] = {}
    for it, e in rows:
        on_usb = e["bytes"] * (IMAGE_TAR_FACTOR if (it["kind"] == "image" and not a.zstd) else 1.0)
        totals[it["group"]] = totals.get(it["group"], 0) + e["bytes"]
        usb_totals[it["group"]] = usb_totals.get(it["group"], 0) + on_usb
        opt = " (optional)" if it["optional"] else ""
        print(f"{it['key']:<{width}} {it['group']:<9} {it['kind']:<9} {human(e['bytes']):>10}  {human(on_usb):>10}  {e['detail']}{opt} [{e['source']}]")
    print()
    print(f"{'group':<12} {'download':>10}  {'on USB':>10}")
    for g in totals:
        print(f"{g:<12} {human(totals[g]):>10}  {human(usb_totals[g]):>10}")
    dl = sum(totals.values())
    usb = sum(usb_totals.values())
    must = sum(v for (it, e), v in zip(rows, [usb_totals_item(it, e, a.zstd) for it, e in rows]) if not it["optional"])
    print(f"{'TOTAL':<12} {human(dl):>10}  {human(usb):>10}   (required items only: {human(must)})")
    drive = next(d for d in (64, 128, 256, 512, 1024, 2048) if d * 1e9 * 0.93 > usb * 1.15)
    print(f"drive: {drive} GB (usable ~{drive * 0.93:.0f} GB; +15% slack). Images stored {'zstd-compressed' if a.zstd else 'as plain docker-save tars'}.")
    if a.write_lock:
        out = {"generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
               "image_tar_factor": IMAGE_TAR_FACTOR,
               "items": {it["key"]: {k: v for k, v in e.items() if k != "source"} for it, e in rows}}
        with open(a.lock, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"lock written: {a.lock}")
    if unknown:
        print(f"{unknown} item(s) of unknown size", file=sys.stderr)
        return 1
    return 0


def usb_totals_item(it: dict, e: dict, zstd: bool) -> float:
    return e["bytes"] * (IMAGE_TAR_FACTOR if (it["kind"] == "image" and not zstd) else 1.0)


def cmd_plan(a: argparse.Namespace) -> int:
    for it in items(a.group):
        k = it["kind"]
        if k == "image":
            spec = [it["ref"]]
        elif k == "hf":
            spec = [it["repo"], it.get("rev") or ""]
        elif k == "hf-files":
            spec = [it["repo"], ",".join(it["files"])]
        elif k == "url":
            spec = [it["url"], it.get("sha256", "")]
        elif k == "apt":
            spec = [",".join(it["packages"])]
        elif k == "git":
            spec = [it["git"], it["ref"], "shallow" if it.get("shallow") else "bundle"]
        elif k == "pip":
            reqs = repo_requirement_files() if it["requirements"] == "repo-requirements" else it["requirements"]
            spec = ["deps" if it["deps"] else "nodeps", it["index"] or "", ",".join(reqs), "|".join(it["extra"])]
        else:
            raise RuntimeError(k)
        # bash `read` with IFS=tab collapses empty fields, so empty spec fields are printed as "-"
        print("\t".join([it["key"], it["group"], k, it["dest"], "1" if it["optional"] else "0"] + [x or "-" for x in spec]))
    return 0


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan"); s.add_argument("--group", action="append"); s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("apt-resolve", help="noble arm64 .debs for packages + deps (TSV)")
    s.add_argument("packages", nargs="+"); s.set_defaults(fn=cmd_apt_resolve)
    s = sub.add_parser("pip-flags", help="the pip target flags for aarch64/cp312 wheels"); s.set_defaults(fn=lambda a: print(" ".join(PIP_TARGET_FLAGS)) or 0)
    s = sub.add_parser("sizes")
    s.add_argument("--offline", action="store_true", help="never touch the network; read the lock")
    s.add_argument("--lock", default=DEFAULT_LOCK)
    s.add_argument("--write-lock", action="store_true")
    s.add_argument("--pip", help="python with pip, used to resolve dependency wheel sets")
    s.add_argument("--zstd", action="store_true", help="images will be zstd-compressed on the USB")
    s.add_argument("--group", action="append")
    s.set_defaults(fn=cmd_sizes)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
