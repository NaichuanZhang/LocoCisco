"""Probe client: call the room's MCP server the way the sandbox will (bearer + TLS via the proxy), from the host.

    UNSPOKEN_MCP_TOKEN=... python -m unspoken.bridge.probe --url https://172.17.0.1:8443/mcp \\
        --ca ~/.unspoken/tls/ca.pem tools
    ... room_status
    ... create_task text="Send the deck" owner=Sam due=Friday

Exit 0 and JSON on stdout when the call succeeds; non-zero with the error otherwise. verify.sh uses it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import ssl
import sys
from typing import Any

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import ENV_MCP_TOKEN


def parse_args_kv(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"tool argument must be key=value, not {p!r}")
        k, v = p.split("=", 1)
        if v in ("true", "false"):
            out[k] = v == "true"
        else:
            try:
                out[k] = float(v) if "." in v else int(v)
            except ValueError:
                out[k] = v
    return out


async def probe(url: str, token: str, ca: str | None, tool: str, args: dict[str, Any], timeout: float) -> dict[str, Any]:
    verify: ssl.SSLContext | bool = True
    if ca:
        verify = ssl.create_default_context(cafile=os.path.expanduser(ca))
    async with httpx2.AsyncClient(headers={"Authorization": f"Bearer {token}"}, verify=verify,
                                  timeout=httpx2.Timeout(timeout, read=max(timeout, 60.0)), follow_redirects=False) as http:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                if tool == "tools":
                    tools = await session.list_tools()
                    return {"server": init.server_info.name, "tools": [t.name for t in tools.tools]}
                result = await session.call_tool(tool, args)
                if result.is_error:
                    raise SystemExit(f"tool error: {[getattr(c, 'text', '') for c in result.content]}")
                if result.structured_content is not None:
                    return {"server": init.server_info.name, "tool": tool, "result": result.structured_content}
                return {"server": init.server_info.name, "tool": tool,
                        "result": "".join(getattr(c, "text", "") for c in result.content)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Call the LocoCisco MCP server like the sandbox does")
    p.add_argument("--url", required=True, help="e.g. https://<bridge-ip>:8443/mcp or http://127.0.0.1:8791/mcp")
    p.add_argument("--ca", default=None, help="CA bundle for the TLS proxy (make-ca.sh ca.pem)")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("tool", help="'tools' to list, or a tool name")
    p.add_argument("args", nargs="*", help="key=value tool arguments")
    a = p.parse_args(argv)
    token = os.environ.get(ENV_MCP_TOKEN, "").strip()
    if not token:
        print(f"probe: {ENV_MCP_TOKEN} is not set", file=sys.stderr)
        return 2
    try:
        out = asyncio.run(probe(a.url, token, a.ca, a.tool, parse_args_kv(a.args), a.timeout))
    except SystemExit:
        raise
    except Exception as exc:  # one line, no traceback: this runs inside shell scripts
        print(f"probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
