"""The bridge: LocoCisco <-> NemoClaw/OpenClaw/OpenShell <-> Discord.

Outbound (room -> people): a Notifier (unspoken.bridge.notifiers). Every byte of text that leaves the box
passes through `on_egress(dest, "text", nbytes)` so the room screen can prove nothing else left.
Inbound (people -> room): the OpenClaw agent in the NemoClaw sandbox calls our MCP server
(unspoken.bridge.mcp_server) through the TLS proxy (unspoken.bridge.tls_proxy). See docs/NEMOCLAW.md.
"""
from __future__ import annotations

from .config import BridgeConfigError, load_bridge_config
from .notifiers import ConsoleNotifier, DiscordRestNotifier, OpenClawNotifier, make_notifier

__all__ = [
    "BridgeConfigError",
    "ConsoleNotifier",
    "DiscordRestNotifier",
    "OpenClawNotifier",
    "load_bridge_config",
    "make_notifier",
]
