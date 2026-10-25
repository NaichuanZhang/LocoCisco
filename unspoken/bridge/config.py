"""config/bridge.toml loading. Secrets never live in the file: they are read from the environment by the
component that needs them (DISCORD_BOT_TOKEN, OPENCLAW_HOOKS_TOKEN, UNSPOKEN_MCP_TOKEN)."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "config" / "bridge.toml"

# Secrets: environment variable names (values are never written to disk by this package).
ENV_DISCORD_TOKEN = "DISCORD_BOT_TOKEN"
ENV_HOOKS_TOKEN = "OPENCLAW_HOOKS_TOKEN"
ENV_MCP_TOKEN = "UNSPOKEN_MCP_TOKEN"


class BridgeConfigError(ValueError):
    """Bad or incomplete bridge configuration (raised at startup, never from a post)."""


def load_bridge_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parse config/bridge.toml (or `path`). Returns the whole document as a dict."""
    p = Path(path) if path else DEFAULT_PATH
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BridgeConfigError(f"bridge config not found: {p}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise BridgeConfigError(f"bridge config {p} is not valid TOML: {exc}") from exc


def secret(name: str, *, required: bool = False) -> str | None:
    """Read a secret from the environment. Empty counts as unset."""
    value = os.environ.get(name, "").strip()
    if not value:
        if required:
            raise BridgeConfigError(f"{name} is not set (secrets come only from the environment)")
        return None
    return value


def mcp_settings(config: dict[str, Any]) -> dict[str, Any]:
    """The [mcp] block with env overrides (UNSPOKEN_MCP_HOST / UNSPOKEN_MCP_PORT)."""
    block = dict(config.get("mcp", {}))
    block.setdefault("host", "127.0.0.1")
    block.setdefault("port", 8791)
    block.setdefault("path", "/mcp")
    block.setdefault("verdict_wait", 3.0)
    block.setdefault("stateless", True)
    block.setdefault("json_response", True)
    if os.environ.get("UNSPOKEN_MCP_HOST"):
        block["host"] = os.environ["UNSPOKEN_MCP_HOST"]
    if os.environ.get("UNSPOKEN_MCP_PORT"):
        block["port"] = int(os.environ["UNSPOKEN_MCP_PORT"])
    return block


def tls_proxy_settings(config: dict[str, Any]) -> dict[str, Any]:
    """The [tls_proxy] block with env overrides (UNSPOKEN_BRIDGE_IP, UNSPOKEN_TLS_PORT, UNSPOKEN_TLS_DIR)."""
    block = dict(config.get("tls_proxy", {}))
    block.setdefault("bind_ip", "")
    block.setdefault("port", 8443)
    block.setdefault("upstream", "http://127.0.0.1:8791")
    block.setdefault("cert_dir", "~/.unspoken/tls")
    if os.environ.get("UNSPOKEN_BRIDGE_IP"):
        block["bind_ip"] = os.environ["UNSPOKEN_BRIDGE_IP"]
    if os.environ.get("UNSPOKEN_TLS_PORT"):
        block["port"] = int(os.environ["UNSPOKEN_TLS_PORT"])
    if os.environ.get("UNSPOKEN_TLS_DIR"):
        block["cert_dir"] = os.environ["UNSPOKEN_TLS_DIR"]
    block["cert_dir"] = str(Path(block["cert_dir"]).expanduser())
    return block
