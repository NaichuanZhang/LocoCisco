"""TLS proxy tests. The upstream is an httpx.MockTransport (no MCP server needed); the TLS test runs the proxy on
127.0.0.1 with certificates from deploy/nemoclaw/make-ca.sh and connects with the generated CA."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn

from unspoken.bridge import tls_proxy
from unspoken.bridge.config import BridgeConfigError
from unspoken.bridge.tls_proxy import TlsProxyApp, create_app, require_private_bind

TOKEN = "proxy-token-0123456789abcdef"
REPO = Path(__file__).resolve().parents[2]
MAKE_CA = REPO / "deploy" / "nemoclaw" / "make-ca.sh"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Upstream:
    """Pretends to be the loopback MCP server: records what the proxy forwarded."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/mcp" and request.method == "POST":
            body = json.loads(request.content)
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body.get("id"), "result": {"echo": body}},
                                  headers={"mcp-session-id": "abc", "x-upstream": "yes"})
        if request.url.path == "/mcp" and request.method == "GET":   # the SSE receive stream, chunked
            return httpx.Response(200, content=b"event: message\ndata: {\"n\":1}\n\nevent: message\ndata: {\"n\":2}\n\n",
                                  headers={"content-type": "text/event-stream", "transfer-encoding": "chunked"})
        if request.url.path == "/mcp" and request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404, json={"error": "upstream 404"})


def proxy_pair(**kw: Any) -> tuple[TlsProxyApp, Upstream]:
    up = Upstream()
    client = httpx.AsyncClient(transport=httpx.MockTransport(up.handler))
    return create_app("http://127.0.0.1:8791", TOKEN, client=client, **kw), up


@pytest.fixture
async def client():
    app, up = proxy_pair()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://172.18.0.1:8443") as c:
        c.up = up  # type: ignore[attr-defined]
        c.app = app  # type: ignore[attr-defined]
        yield c


# --------------------------------------------------------------------------- guards
def test_bind_guard_accepts_only_private_ipv4():
    for ok in ("172.18.0.1", "10.0.0.5", "192.168.1.20", "100.64.0.1"):
        assert require_private_bind(ok) == ok
    for bad in ("0.0.0.0", "::", "::1", "fd00::1", "8.8.8.8", "169.254.1.1", "224.0.0.1", "docker0", "", "localhost"):
        with pytest.raises(ValueError):
            require_private_bind(bad)
    with pytest.raises(ValueError):
        require_private_bind("127.0.0.1")
    assert require_private_bind("127.0.0.1", allow_loopback=True) == "127.0.0.1"


def test_upstream_and_token_guards():
    for bad in ("https://127.0.0.1:8791", "http://10.0.0.5:8791", "http://example.com", "127.0.0.1:8791"):
        with pytest.raises(ValueError):
            create_app(bad, TOKEN)
    for bad_token in ("", "short", None):
        with pytest.raises(BridgeConfigError):
            create_app("http://127.0.0.1:8791", bad_token)  # type: ignore[arg-type]
    create_app("http://localhost:8791", TOKEN)


def test_cli_refuses_bad_bind_without_starting(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("UNSPOKEN_MCP_TOKEN", TOKEN)
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x"), key.write_text("y")
    assert tls_proxy.main(["--bind-ip", "0.0.0.0", "--cert", str(cert), "--key", str(key)]) == 2
    assert tls_proxy.main(["--bind-ip", "172.18.0.1", "--cert", str(tmp_path / "missing"), "--key", str(key)]) == 2
    monkeypatch.setenv("UNSPOKEN_MCP_TOKEN", "")
    assert tls_proxy.main(["--bind-ip", "172.18.0.1", "--cert", str(cert), "--key", str(key)]) == 2


# --------------------------------------------------------------------------- gates
async def test_health_open_everything_else_gated(client: httpx.AsyncClient):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json() == {"ok": True, "proxy": "unspoken-tls"}
    r = await client.post("/mcp", json={"id": 1})
    assert r.status_code == 401 and r.headers["www-authenticate"] == 'Bearer realm="unspoken"'
    r = await client.post("/mcp", json={"id": 1}, headers={"Authorization": "Bearer nope-nope-nope-nope"})
    assert r.status_code == 401
    r = await client.post("/mcp", json={"id": 1}, headers={"Authorization": f"Basic {TOKEN}"})
    assert r.status_code == 401
    r = await client.post("/admin", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 404
    r = await client.put("/mcp", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert r.status_code == 405 and "POST" in r.headers["allow"]
    assert client.up.requests == []            # type: ignore[attr-defined]  nothing reached the upstream
    assert client.app.rejected == 5           # type: ignore[attr-defined]


async def test_forwards_with_host_rewrite_and_streams_back(client: httpx.AsyncClient):
    up: Upstream = client.up  # type: ignore[attr-defined]
    r = await client.post("/mcp?x=1", json={"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
                          headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json, text/event-stream",
                                   "X-Trace": "t1", "Connection": "keep-alive"})
    assert r.status_code == 200 and r.json()["result"]["echo"]["id"] == 7
    assert r.headers["mcp-session-id"] == "abc" and r.headers["x-upstream"] == "yes"
    fwd = up.requests[-1]
    assert str(fwd.url) == "http://127.0.0.1:8791/mcp?x=1"
    assert fwd.headers["host"] == "127.0.0.1:8791"                   # Host rewritten for the loopback server
    assert fwd.headers["authorization"] == f"Bearer {TOKEN}"        # the loopback server checks it again
    assert fwd.headers["x-trace"] == "t1" and fwd.headers["accept"] == "application/json, text/event-stream"
    r = await client.get("/mcp", headers={"Authorization": f"Bearer {TOKEN}", "Accept": "text/event-stream"})
    assert r.status_code == 200 and r.headers["content-type"] == "text/event-stream"
    assert r.text.count("event: message") == 2 and "transfer-encoding" not in r.headers
    r = await client.delete("/mcp", headers={"Authorization": f"Bearer {TOKEN}", "mcp-session-id": "abc"})
    assert r.status_code == 204 and up.requests[-1].headers["mcp-session-id"] == "abc"
    assert client.app.forwarded == 3  # type: ignore[attr-defined]


async def test_upstream_down_is_502_and_big_bodies_are_413():
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")
    app = create_app("http://127.0.0.1:8791", TOKEN, client=httpx.AsyncClient(transport=httpx.MockTransport(down)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="https://172.18.0.1:8443") as c:
        r = await c.post("/mcp", json={}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 502
        r = await c.post("/mcp", content=b"x" * (tls_proxy.MAX_BODY + 1), headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 413


# --------------------------------------------------------------------------- real TLS with make-ca.sh certs
@pytest.mark.skipif(not MAKE_CA.exists(), reason="make-ca.sh missing")
async def test_make_ca_and_tls_round_trip(tmp_path: Path):
    if subprocess.run(["bash", "-c", "command -v openssl"], capture_output=True).returncode != 0:
        pytest.skip("openssl not installed")
    certdir = tmp_path / "tls"
    env = {**os.environ, "UNSPOKEN_TLS_DIR": str(certdir)}
    res = subprocess.run(["bash", str(MAKE_CA), "--ip", "172.18.0.1", "--extra-ip", "127.0.0.1", "--dir", str(certdir)],
                         capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0, res.stderr
    assert (certdir.stat().st_mode & 0o777) == 0o700
    for name, mode in (("ca.key", 0o600), ("server.key", 0o600), ("ca.pem", 0o644), ("server.crt", 0o644)):
        assert (certdir / name).exists(), name
        assert ((certdir / name).stat().st_mode & 0o777) == mode, name
    assert not (certdir / "server.csr").exists()
    san = subprocess.run(["openssl", "x509", "-in", str(certdir / "server.crt"), "-noout", "-ext", "subjectAltName"],
                         capture_output=True, text=True).stdout
    assert "IP Address:172.18.0.1" in san and "IP Address:127.0.0.1" in san
    ca_text = (certdir / "ca.pem").read_text()
    assert ca_text.count("BEGIN CERTIFICATE") == 1                         # CA-only bundle, as NemoClaw requires
    # idempotent: a second run keeps the files
    before = (certdir / "server.crt").read_bytes()
    res2 = subprocess.run(["bash", str(MAKE_CA), "--ip", "172.18.0.1", "--dir", str(certdir)], capture_output=True,
                          text=True, env=env, timeout=60)
    assert res2.returncode == 0 and (certdir / "server.crt").read_bytes() == before and "already exist" in res2.stderr
    # a public IP is refused
    bad = subprocess.run(["bash", str(MAKE_CA), "--ip", "8.8.8.8", "--dir", str(tmp_path / "x")], capture_output=True,
                         text=True, env=env, timeout=60)
    assert bad.returncode != 0 and "RFC1918" in bad.stderr

    # run the proxy over TLS on loopback (dev exception) and talk to it with the CA
    up = Upstream()
    app = create_app("http://127.0.0.1:8791", TOKEN, client=httpx.AsyncClient(transport=httpx.MockTransport(up.handler)))
    port = free_port()
    config = tls_proxy.uvicorn_config(app, "127.0.0.1", port, str(certdir / "server.crt"), str(certdir / "server.key"),
                                      allow_loopback=True, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(300):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    try:
        ctx = ssl.create_default_context(cafile=str(certdir / "ca.pem"))
        async with httpx.AsyncClient(verify=ctx) as c:
            r = await c.get(f"https://127.0.0.1:{port}/health")
            assert r.status_code == 200 and r.json()["ok"] is True
            r = await c.post(f"https://127.0.0.1:{port}/mcp", json={"id": 1})
            assert r.status_code == 401                                     # denial verified before anything else
            r = await c.post(f"https://127.0.0.1:{port}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
                             headers={"Authorization": f"Bearer {TOKEN}"})
            assert r.status_code == 200 and r.json()["result"]["echo"]["method"] == "ping"
        # without the CA the handshake fails: the proxy is not trusted by default anywhere else
        async with httpx.AsyncClient(verify=True) as c:
            with pytest.raises(httpx.ConnectError):
                await c.get(f"https://127.0.0.1:{port}/health")
        # the proxy refuses to come up on all interfaces
        with pytest.raises(ValueError):
            tls_proxy.uvicorn_config(app, "0.0.0.0", port, str(certdir / "server.crt"), str(certdir / "server.key"))
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)
