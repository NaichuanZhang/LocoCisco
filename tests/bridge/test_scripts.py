"""deploy/nemoclaw scripts and nemoclaw/ assets: syntax-clean, shellcheck-clean, dry-run prints every command, the
policy preset and skill follow NemoClaw's documented shapes, and nothing secret or Amazon-internal is in the tree."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DEPLOY = REPO / "deploy" / "nemoclaw"
SCRIPTS = sorted(DEPLOY.glob("*.sh"))
OWNED = [REPO / "unspoken" / "bridge", REPO / "deploy" / "nemoclaw", REPO / "nemoclaw", REPO / "tests" / "bridge",
         REPO / "config" / "bridge.toml", REPO / "requirements" / "bridge.txt", REPO / "docs" / "NEMOCLAW.md"]


def owned_files() -> list[Path]:
    out: list[Path] = []
    for p in OWNED:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out += [f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts]
    return out


def dry_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DISCORD_", "UNSPOKEN_", "NEMOCLAW_", "OPENCLAW_"))}
    env.update({"UNSPOKEN_RUN_DIR": str(tmp_path / "run"), "UNSPOKEN_TLS_DIR": str(tmp_path / "tls"),
                "UNSPOKEN_BRIDGE_IP": "172.18.0.1", "PATH": env.get("PATH", "/usr/bin:/bin")})
    env.update(extra)
    return env


def sh(script: str, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(DEPLOY / script), *args], capture_output=True, text=True, env=env, timeout=120)


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_scripts_exist_and_bash_n_clean():
    assert {s.name for s in SCRIPTS} == {"lib.sh", "make-ca.sh", "onboard.sh", "connect.sh", "verify.sh"}
    for s in SCRIPTS:
        r = subprocess.run(["bash", "-n", str(s)], capture_output=True, text=True)
        assert r.returncode == 0, f"{s.name}: {r.stderr}"
        if s.name != "lib.sh":
            assert os.access(s, os.X_OK), f"{s.name} is not executable"
            assert "--dry-run" in s.read_text(), f"{s.name} has no --dry-run"


def test_shellcheck_clean():
    sc = shutil.which("shellcheck") or str(Path(sys.executable).with_name("shellcheck"))
    if not Path(sc).exists():
        pytest.skip("shellcheck not installed (pip install shellcheck-py)")
    r = subprocess.run([sc, "-x", "-P", "SCRIPTDIR", "-s", "bash", *map(str, SCRIPTS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_onboard_dry_run_prints_the_nemoclaw_commands(tmp_path: Path):
    r = sh("onboard.sh", "--dry-run", env=dry_env(tmp_path, DISCORD_BOT_TOKEN="x", DISCORD_SERVER_ID="1"))
    out = plain(r.stdout + r.stderr)
    assert r.returncode == 0, out
    assert "+ nemoclaw onboard --name unspoken --non-interactive --yes --yes-i-accept-third-party-software" in out
    assert "NEMOCLAW_PROVIDER=install-vllm NEMOCLAW_VLLM_MODEL=qwen3.6-35b-a3b-nvfp4" in out
    assert "NEMOCLAW_POLICY_TIER=restricted" in out and "NEMOCLAW_CORPORATE_CA_BUNDLE=" in out
    assert "DISCORD_REQUIRE_MENTION=1" in out and "Message Content Intent" in out
    assert "+ openssl req -x509" in out and "IP:172.18.0.1" in out
    assert "x" not in re.findall(r"DISCORD_BOT_TOKEN=(\S+)", out)   # never echoes the token value
    r = sh("onboard.sh", "--dry-run", "--profile", "vllm.dgx-spark-gb10.single.qwen", env=dry_env(tmp_path, DISCORD_BOT_TOKEN="x", DISCORD_SERVER_ID="1"))
    assert r.returncode == 0 and "--profile vllm.dgx-spark-gb10.single.qwen" in plain(r.stdout + r.stderr)
    bad = sh("onboard.sh", "--dry-run", "--tier", "personal", env=dry_env(tmp_path))
    assert bad.returncode != 0 and "personal" in plain(bad.stderr)


def test_onboard_refuses_placeholder_discord_env_when_live(tmp_path: Path):
    """Without --dry-run the Discord variables are mandatory (nemoclaw is not here, so it must fail before it)."""
    r = sh("onboard.sh", env=dry_env(tmp_path, DISCORD_BOT_TOKEN="<your-discord-bot-token>", DISCORD_SERVER_ID="1"))
    assert r.returncode != 0
    text = plain(r.stderr)
    assert "nemoclaw is not on PATH" in text or "DISCORD_BOT_TOKEN" in text


def test_connect_dry_run_covers_proxy_mcp_add_policy_persona_and_hooks(tmp_path: Path):
    env = dry_env(tmp_path, UNSPOKEN_MCP_TOKEN="0123456789abcdef0123456789abcdef", OPENCLAW_HOOKS_TOKEN="hooks-0123456789abcdef")
    r = sh("connect.sh", "--dry-run", "--firewall", "--hooks", env=env)
    out = plain(r.stdout + r.stderr)
    assert r.returncode == 0, out
    assert "-m unspoken.bridge.mcp_server --runtime fake --host 127.0.0.1 --port 8791" in out
    assert "-m unspoken.bridge.tls_proxy --bind-ip 172.18.0.1 --port 8443 --upstream http://127.0.0.1:8791" in out
    assert "+ sudo ufw allow proto tcp from 172.18.0.0/16 to 172.18.0.1 port 8443" in out
    assert "+ nemoclaw unspoken mcp add unspoken --url https://172.18.0.1:8443/mcp --env UNSPOKEN_MCP_TOKEN --trusted-private-host 172.18.0.1" in out
    assert "+ nemoclaw unspoken mcp status unspoken" in out
    assert "+ nemoclaw unspoken policy add discord --yes" in out
    for f in ("SOUL.md", "AGENTS.md", "TOOLS.md", "IDENTITY.md", "USER.md"):
        assert f"/sandbox/.openclaw/workspace/{f}" in out
    assert "+ nemoclaw unspoken skill install" in out and "nemoclaw/skills/unspoken-room/" in out
    assert "+ nemoclaw unspoken exec -- openclaw config patch --file /sandbox/hooks.patch.json5" in out
    assert "hooks-0123456789abcdef" not in out                      # the hooks token value is never printed
    assert "0123456789abcdef0123456789abcdef" not in out            # nor the MCP token
    assert "0.0.0.0" not in out
    # fallback path
    r = sh("connect.sh", "--dry-run", "--fallback-rest", "--skip-persona", env=env)
    out = plain(r.stdout + r.stderr)
    assert r.returncode == 0 and "policy add --from-file" in out and "--trusted-private-host 172.18.0.1 --dry-run" in out
    assert "mcp add unspoken" not in out and "mcp.servers.unspoken.url https://172.18.0.1:8443/mcp" in out
    # the token is mandatory even for a dry run's live sibling
    r = sh("connect.sh", env=dry_env(tmp_path))
    assert r.returncode != 0 and ("UNSPOKEN_MCP_TOKEN" in plain(r.stderr) or "not on PATH" in plain(r.stderr))


def test_verify_dry_run_lists_every_check(tmp_path: Path):
    r = sh("verify.sh", "--dry-run", "--post", "--agent", env=dry_env(tmp_path, UNSPOKEN_MCP_TOKEN="0123456789abcdef0123456789abcdef"))
    out = plain(r.stdout + r.stderr)
    assert r.returncode == 0, out
    assert "http://127.0.0.1:8791/health" in out and "https://172.18.0.1:8443/health" in out
    assert "+ deny401" in out and "+ notpublic" in out
    assert "-m unspoken.bridge.probe --url https://172.18.0.1:8443/mcp" in out and " room_status" in out
    assert "+ mcpmatch" in out and "channels status --channel discord" in out
    assert "-m unspoken.bridge.send --channel notes" in out
    assert "nemoclaw unspoken agent --agent main" in out


def test_make_ca_refuses_public_and_reserved_ips(tmp_path: Path):
    for ip in ("8.8.8.8", "240.10.0.1", "169.254.1.1", "0.0.0.0"):
        r = sh("make-ca.sh", "--dry-run", "--ip", ip, "--dir", str(tmp_path / "x"), env=dry_env(tmp_path))
        assert r.returncode != 0 and "RFC1918" in plain(r.stderr), ip


# --------------------------------------------------------------------------- nemoclaw/ assets
def test_policy_preset_follows_the_documented_shape():
    text = (REPO / "nemoclaw" / "policy" / "unspoken-mcp-host.yaml").read_text()
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None
    if yaml is not None:
        doc = yaml.safe_load(text)
        assert doc["preset"]["name"] == "unspoken-mcp-host"
        assert re.fullmatch(r"[a-z0-9]([a-z0-9-]*[a-z0-9])?", doc["preset"]["name"])        # RFC 1123 label
        pol = doc["network_policies"]["unspoken_mcp_host"]
        ep = pol["endpoints"][0]
        assert ep["port"] == 8443 and ep["protocol"] == "rest" and ep["enforcement"] == "enforce"
        assert [r["allow"]["method"] for r in ep["rules"]] == ["GET", "POST", "DELETE"]
        assert all(r["allow"]["path"] == "/mcp" for r in ep["rules"])
        assert pol["binaries"] == [{"path": "/usr/local/bin/node"}]
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    assert "allowed_ips" not in body                                # NemoClaw rejects user-authored pins
    assert "BRIDGE_IP_PLACEHOLDER" in text and "npm_yarn" not in text and "personal_open_internet" not in text
    assert "/**" not in text


def test_skill_frontmatter_and_persona_files():
    skill = (REPO / "nemoclaw" / "skills" / "unspoken-room" / "SKILL.md").read_text()
    m = re.match(r"---\n(.*?)\n---\n", skill, re.S)
    assert m, "SKILL.md needs YAML frontmatter"
    front = m.group(1)
    name = re.search(r"^name:\s*(\S+)$", front, re.M).group(1)
    assert name == "unspoken-room" and re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name)
    assert re.search(r"^description:\s*\S", front, re.M)
    for f in ("SOUL.md", "AGENTS.md", "TOOLS.md", "IDENTITY.md", "USER.md"):
        assert (REPO / "nemoclaw" / "workspace" / f).exists()
    soul = (REPO / "nemoclaw" / "workspace" / "SOUL.md").read_text()
    agents = (REPO / "nemoclaw" / "workspace" / "AGENTS.md").read_text()
    assert "off the record" in soul.lower() and "never exists" in soul.lower()
    for phrase in ("I'm Priya", "off the record", "undo", "what did we decide"):
        assert phrase in agents, phrase
    for tool in ("room_status", "transcript", "list_items", "search_docs", "recap", "checkin", "pause_notes",
                 "resume_notes", "create_task", "set_light", "undo_item", "decide_request"):
        assert tool in (REPO / "nemoclaw" / "workspace" / "TOOLS.md").read_text(), tool


def test_no_secrets_internal_urls_or_forbidden_names_in_owned_files():
    secretish = re.compile(r"(?i)(discord_bot_token|openclaw_hooks_token|unspoken_mcp_token)\s*[=:]\s*['\"]?[A-Za-z0-9._-]{20,}")
    internal = re.compile(r"(?i)(amazon\.com|a2z\.com|aws\.dev|amazon\.dev|amazon\.work|midway|brazil|crux|isengard|conduit)")
    for f in owned_files():
        if f.suffix in (".pyc",) or f == Path(__file__).resolve():   # this scanner names the patterns it bans
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        if "tests" not in f.parts:                                  # tests carry obviously fake tokens
            assert not secretish.search(text), f"secret-looking assignment in {f}"
        assert not internal.search(text), f"Amazon-internal reference in {f}"
        if f.suffix in (".sh", ".toml", ".yaml"):
            assert "hackathon" not in text.lower(), f"infrastructure name must not say hackathon: {f}"
            assert not re.search(r'(--host|--bind-ip|--bind|host\s*=|bind\s*=)\s*["\']?0\.0\.0\.0', text), f"all-interfaces bind in {f}"
