"""TLS front for the MCP server, for the NemoClaw sandbox only.

NemoClaw's managed MCP requires HTTPS on a routed private (RFC1918) address and rejects loopback, so this
proxy terminates TLS on the private IPv4 gateway of the `openshell-docker` bridge (the one host address the
sandbox can route to) and forwards to the loopback MCP server. It is the single network-facing listener in
LocoCisco, so it is deliberately small:

- binds ONLY to the given private IPv4 literal (never 0.0.0.0 / :: / a public address);
- requires the same bearer token the MCP server requires, checked in constant time BEFORE anything is
  forwarded (401 otherwise); the token is forwarded, so the loopback server checks it again;
- forwards only the MCP path (GET/POST/DELETE, Streamable HTTP) and GET /health; everything else is 404/405;
- certificate and key come from deploy/nemoclaw/make-ca.sh (0700 dir outside the repo); the host firewall
  limits :8443 to the sandbox subnet (docs/NEMOCLAW.md).

    python -m unspoken.bridge.tls_proxy --bind-ip 172.18.0.1 --port 8443 --upstream http://127.0.0.1:8791 \\
        --cert ~/.unspoken/tls/server.crt --key ~/.unspoken/tls/server.key      # token: UNSPOKEN_MCP_TOKEN
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import logging
import os
import ssl
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import httpx

from .config import ENV_MCP_TOKEN, BridgeConfigError

log = logging.getLogger("unspoken.bridge.tls_proxy")

HOP_BY_HOP = {b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te", b"trailer",
              b"transfer-encoding", b"upgrade", b"host", b"content-length"}   # request side (httpx recomputes length)
RESPONSE_DROP = {b"connection", b"keep-alive", b"transfer-encoding", b"upgrade", b"server", b"date"}  # uvicorn re-frames
MAX_BODY = 1 << 20          # 1 MiB; NemoClaw caps MCP request bodies at 128 KiB anyway
ALLOWED_METHODS = ("GET", "POST", "DELETE")
CGNAT = ipaddress.ip_network("100.64.0.0/10")   # carrier-grade NAT: NemoClaw admits it like RFC1918
MIN_TOKEN_CHARS = 16

Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]


def require_private_bind(ip: str, *, allow_loopback: bool = False) -> str:
    """Accept one private IPv4 literal (RFC1918 / CGNAT). Loopback only when the caller says so (tests)."""
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError as exc:
        raise ValueError(f"bind address must be an IP literal, not {ip!r}") from exc
    if addr.version != 4:
        raise ValueError(f"refusing {ip!r}: NemoClaw accepts IPv4 private literals for MCP; IPv6 literals are rejected")
    if addr.is_unspecified or addr.is_multicast or addr.is_link_local or addr.is_reserved:
        raise ValueError(f"refusing to bind {ip!r}: never all interfaces, multicast, link-local or reserved")
    if addr.is_loopback:
        if allow_loopback:
            return str(addr)
        raise ValueError(f"refusing {ip!r}: the sandbox cannot reach loopback; bind the docker bridge IP")
    if not (addr.is_private or addr in CGNAT):
        raise ValueError(f"refusing to bind public address {ip!r}: the proxy is for the sandbox network only")
    return str(addr)


def require_token(token: str | None) -> str:
    if not token or len(token) < MIN_TOKEN_CHARS:
        raise BridgeConfigError(f"bearer token missing or shorter than {MIN_TOKEN_CHARS} chars (set {ENV_MCP_TOKEN})")
    return token


class TlsProxyApp:
    """ASGI app. `client` is injectable for tests; otherwise one is opened on lifespan startup."""

    def __init__(self, upstream: str, token: str, *, mcp_path: str = "/mcp", health_path: str = "/health",
                 client: httpx.AsyncClient | None = None, read_timeout: float = 300.0) -> None:
        parts = urlsplit(upstream)
        host = parts.hostname or ""
        if parts.scheme != "http" or not host:
            raise ValueError(f"upstream must be an http://host:port URL on this machine, not {upstream!r}")
        if not (host == "localhost" or (_is_ip(host) and ipaddress.ip_address(host).is_loopback)):
            raise ValueError(f"upstream must be loopback (the MCP server binds 127.0.0.1), not {upstream!r}")
        self.upstream = upstream.rstrip("/")
        self.upstream_host = parts.netloc.encode("latin-1")
        self.token = require_token(token).encode("utf-8")
        self.mcp_path = mcp_path
        self.health_path = health_path
        self._client = client
        self._owns_client = client is None
        self._read_timeout = read_timeout
        self.rejected = 0
        self.forwarded = 0

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._lifespan(receive, send)
            return
        if scope["type"] != "http":
            return
        method = scope.get("method", "")
        path = scope.get("path", "")
        if path == self.health_path and method in ("GET", "HEAD"):
            await _json(send, 200, {"ok": True, "proxy": "unspoken-tls"})
            return
        if path != self.mcp_path:
            self.rejected += 1
            await _json(send, 404, {"error": "not found"})
            return
        if method not in ALLOWED_METHODS:
            self.rejected += 1
            await _json(send, 405, {"error": "method not allowed"}, [(b"allow", ", ".join(ALLOWED_METHODS).encode())])
            return
        if not self._authorized(scope):
            self.rejected += 1
            await _json(send, 401, {"error": "unauthorized"}, [(b"www-authenticate", b'Bearer realm="unspoken"')])
            return
        await self._forward(scope, receive, send)

    def _authorized(self, scope: dict[str, Any]) -> bool:
        for k, v in scope.get("headers", []):
            if k == b"authorization":
                return v[:7].lower() == b"bearer " and hmac.compare_digest(v[7:].strip(), self.token)
        return False

    async def _lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            msg = await receive()
            if msg["type"] == "lifespan.startup":
                if self._client is None:
                    self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=self._read_timeout))
                await send({"type": "lifespan.startup.complete"})
            elif msg["type"] == "lifespan.shutdown":
                if self._owns_client and self._client is not None:
                    await self._client.aclose()
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def _forward(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if self._client is None:   # no lifespan ran (bare ASGI test harness): open lazily
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=self._read_timeout))
        body = bytearray()
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            if len(body) > MAX_BODY:
                self.rejected += 1
                await _json(send, 413, {"error": "body too large"})
                return
            if not msg.get("more_body"):
                break
        headers = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in scope.get("headers", [])
                   if k.lower() not in HOP_BY_HOP]
        headers.append(("host", self.upstream_host.decode("latin-1")))
        url = self.upstream + scope["path"]
        if scope.get("query_string"):
            url += "?" + scope["query_string"].decode("latin-1")
        request = self._client.build_request(scope["method"], url, headers=headers, content=bytes(body))
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as exc:
            log.warning("upstream %s unreachable: %s", self.upstream, exc)
            await _json(send, 502, {"error": "upstream unreachable"})
            return
        self.forwarded += 1
        try:
            out_headers = [(k.encode("latin-1"), v.encode("latin-1")) for k, v in response.headers.multi_items()
                           if k.lower().encode("latin-1") not in RESPONSE_DROP]
            await send({"type": "http.response.start", "status": response.status_code, "headers": out_headers})
            if response.is_stream_consumed:          # body already in memory (small responses, test transports)
                await send({"type": "http.response.body", "body": response.content, "more_body": False})
                return
            async for chunk in response.aiter_raw():
                if chunk:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            await response.aclose()


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


async def _json(send: Send, status: int, body: dict[str, Any], extra: list[tuple[bytes, bytes]] | None = None) -> None:
    raw = json.dumps(body).encode("utf-8")
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode()),
                            (b"cache-control", b"no-store")] + (extra or [])})
    await send({"type": "http.response.body", "body": raw})


def create_app(upstream: str, token: str, **kw: Any) -> TlsProxyApp:
    return TlsProxyApp(upstream, token, **kw)


def uvicorn_config(app: Any, bind_ip: str, port: int, certfile: str, keyfile: str, *,
                   allow_loopback: bool = False, log_level: str = "info") -> Any:
    """TLS 1.2+ with the server cert from make-ca.sh. Refuses non-private binds."""
    import uvicorn

    host = require_private_bind(bind_ip, allow_loopback=allow_loopback)
    for f in (certfile, keyfile):
        if not Path(f).expanduser().is_file():
            raise FileNotFoundError(f"{f}: run deploy/nemoclaw/make-ca.sh first")
    return uvicorn.Config(app, host=host, port=port, log_level=log_level, lifespan="on",
                          ssl_certfile=str(Path(certfile).expanduser()), ssl_keyfile=str(Path(keyfile).expanduser()),
                          ssl_version=ssl.PROTOCOL_TLS_SERVER, ssl_ciphers="HIGH:!aNULL:!MD5",
                          proxy_headers=False, server_header=False, date_header=False)


async def serve(bind_ip: str, port: int, upstream: str, token: str, certfile: str, keyfile: str, *,
                allow_loopback: bool = False, mcp_path: str = "/mcp") -> None:
    import uvicorn

    app = create_app(upstream, token, mcp_path=mcp_path)
    await uvicorn.Server(uvicorn_config(app, bind_ip, port, certfile, keyfile, allow_loopback=allow_loopback)).serve()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TLS front for the LocoCisco MCP server (sandbox network only)")
    p.add_argument("--bind-ip", required=True, help="private IPv4 of the openshell-docker bridge (never 0.0.0.0)")
    p.add_argument("--port", type=int, default=8443)
    p.add_argument("--upstream", default="http://127.0.0.1:8791")
    p.add_argument("--mcp-path", default="/mcp")
    p.add_argument("--cert", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--allow-loopback", action="store_true", help="dev only: let --bind-ip be 127.0.0.1")
    args = p.parse_args(argv)
    token = os.environ.get(ENV_MCP_TOKEN, "")
    try:
        require_token(token)
        asyncio.run(serve(args.bind_ip, args.port, args.upstream, token, args.cert, args.key,
                          allow_loopback=args.allow_loopback, mcp_path=args.mcp_path))
    except (ValueError, FileNotFoundError, BridgeConfigError) as exc:
        print(f"tls_proxy: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
