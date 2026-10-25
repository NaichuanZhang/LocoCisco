# Bridge workstream report (ws/bridge) — NemoClaw / OpenClaw / OpenShell / Discord

Branch `ws/bridge`, worktree `unspoken-wt/bridge`. STOP condition met: notifiers + MCP server pass tests with
fakes, the TLS proxy passes a real TLS round trip with `make-ca.sh` certificates, the scripts dry-run cleanly and
are `bash -n` + shellcheck clean. No NemoClaw/OpenShell/OpenClaw was installed here; see §4 for what that leaves open.

## 1. What I built

| Path | One line |
|---|---|
| `unspoken/bridge/__init__.py` | exports `make_notifier`, `load_bridge_config`, the notifier classes |
| `unspoken/bridge/config.py` | `config/bridge.toml` loader; secrets only from env (`DISCORD_BOT_TOKEN`, `OPENCLAW_HOOKS_TOKEN`, `UNSPOKEN_MCP_TOKEN`); env overrides for ports/bridge IP |
| `unspoken/bridge/notifiers.py` | `make_notifier(config, on_egress)`; `OpenClawNotifier` (primary; mode `exec` = `nemoclaw <sb> exec -- openclaw message send …` verbatim with message ids + task threads; mode `hooks` = `/hooks/agent`), `DiscordRestNotifier` (fallback, bot token, threads, DMs, 429/5xx retries), `ConsoleNotifier` (dev); 1900-char chunking; every sent chunk → `on_egress(dest, "text", nbytes)`; retries with backoff + timeouts; `post()`/`dm()` never raise |
| `unspoken/bridge/mcp_server.py` | `async serve(runtime, host, port, token)` (fixed interface), `create_app()`, `run()`, CLI; Streamable HTTP via `mcp` 2.1.1, constant-time bearer gate, loopback-only bind, open `GET /health`; tools `room_status transcript(since_minutes) list_items(kind) search_docs(query) recap checkin(name) pause_notes(reason) resume_notes create_task(owner,text,due) set_light(scene)` + `undo_item(item_id)` `decide_request(request_id, approve)`; `create_task`/`set_light` publish `MT.PROPOSAL` with `by="user", route="user"` and wait up to `verdict_wait` s for the policy's `MT.VERDICT`; every read tool's description says off-record content never exists |
| `unspoken/bridge/tls_proxy.py` | TLS reverse proxy: binds one private IPv4 literal only (RFC1918/CGNAT; loopback only with `--allow-loopback`), bearer verified before forwarding, `/mcp` GET/POST/DELETE + `/health` only, Host rewritten to the loopback upstream, streaming passthrough, 413/502 handling; CLI |
| `unspoken/bridge/probe.py` | host-side MCP client (bearer + CA) used by `verify.sh`: `tools`, `<tool> key=value…` |
| `unspoken/bridge/send.py` | one test post through the configured notifier (`verify.sh --post`) |
| `unspoken/bridge/dev_fake.py` | `FakeMeetingRuntime` (real Bus, stand-in policy) for tests and `--runtime fake` |
| `config/bridge.toml` | transport, channel ids (placeholders to fill), threads, retries, `[mcp]` 8791/8792, `[tls_proxy]` 8443 |
| `requirements/bridge.txt` | pins (see §5) |
| `deploy/nemoclaw/lib.sh` | shared: `run` (prints under `--dry-run`), `need`, `need_env`, `bridge_ip` (openshell-docker gateway), pidfile services |
| `deploy/nemoclaw/make-ca.sh` | EC P-256 local CA + leaf with `IP:<bridge-ip>` SAN; 0700 dir outside the repo (`~/.unspoken/tls`); CA-only PEM for `NEMOCLAW_CORPORATE_CA_BUNDLE`; refuses non-RFC1918 IPs; idempotent |
| `deploy/nemoclaw/onboard.sh` | host checks, Discord env, CA, `nemoclaw onboard --name unspoken --non-interactive …` with `NEMOCLAW_PROVIDER=install-vllm NEMOCLAW_VLLM_MODEL=qwen3.6-35b-a3b-nvfp4`, tier `restricted` (or `--profile <id>`) |
| `deploy/nemoclaw/connect.sh` | MCP server (fake/sim) if nothing on :8791, TLS proxy on the bridge IP, anonymous-POST→401 check, firewall rule (printed / `--firewall`), `mcp add … --trusted-private-host`, `policy add discord`, persona upload + `skill install`, `--hooks`, `--fallback-rest` |
| `deploy/nemoclaw/verify.sh` | health on both listeners, 401 denial, no `0.0.0.0` listeners, `tools/list` + `room_status` + `list_items` through the proxy, `mcp status … state == match`, Discord channel status, `--post`, `--agent` |
| `nemoclaw/workspace/{SOUL,AGENTS,TOOLS,IDENTITY,USER}.md` | the Discord persona: short replies, off-record never exists, phrase → tool table ("I'm Priya", "off the record", "undo", ✅ ❌, "what did we decide?") |
| `nemoclaw/skills/unspoken-room/SKILL.md` | skill with `name`/`description` frontmatter (NemoClaw-valid name) |
| `nemoclaw/policy/unspoken-mcp-host.yaml` | custom preset for Fallback A (`/mcp` GET/POST/DELETE from `/usr/local/bin/node`, no `allowed_ips`) |
| `docs/NEMOCLAW.md` | architecture, cited research, portal steps, day-of commands + expected output, fallbacks, troubleshooting, the `[DAY-OF CHECK]` list |
| `tests/bridge/` | 44 tests: notifiers, MCP server (real `mcp` client over loopback), TLS proxy (+ real TLS with make-ca.sh), scripts/assets |

## 2. How to run it

```bash
cd unspoken-wt/bridge && UV_CACHE_DIR=... scripts/dev-venv.sh && uv pip install --python .venv/bin/python -r requirements/bridge.txt
.venv/bin/python -m pytest -q                                   # whole suite
export UNSPOKEN_MCP_TOKEN="$(openssl rand -hex 32)"
.venv/bin/python -m unspoken.bridge.mcp_server --runtime fake --port 8792          # loopback MCP server (dev)
deploy/nemoclaw/make-ca.sh --ip 172.18.0.1 --extra-ip 127.0.0.1                    # certs in ~/.unspoken/tls
.venv/bin/python -m unspoken.bridge.tls_proxy --bind-ip 127.0.0.1 --allow-loopback --port 8443 \
    --upstream http://127.0.0.1:8792 --cert ~/.unspoken/tls/server.crt --key ~/.unspoken/tls/server.key
.venv/bin/python -m unspoken.bridge.probe --url https://127.0.0.1:8443/mcp --ca ~/.unspoken/tls/ca.pem room_status
UNSPOKEN_NOTIFIER=console .venv/bin/python -m unspoken.bridge.send --text "hello" --health
# GB10 (in order): deploy/nemoclaw/onboard.sh ; deploy/nemoclaw/connect.sh --firewall ; deploy/nemoclaw/verify.sh --post --agent
# every script: --dry-run prints each command
```

In code: `notifier = make_notifier(load_bridge_config(), on_egress)`; `await serve(runtime, "127.0.0.1", 8791, token)`
from the meeting process.

## 3. Verification (this desktop, x86_64, no GPU, no NemoClaw)

- `.venv/bin/python -m pytest -q` → **427 passed, 2 skipped** (baseline 383 passed + 2 skipped; +44 bridge tests), 21 s.
- Live loopback chain (not in pytest): `mcp_server --runtime fake` on 127.0.0.1:8792 + `tls_proxy --bind-ip 127.0.0.1
  --allow-loopback` on :8443 with make-ca.sh certs → `ss -ltnp` shows both on `127.0.0.1` only; `GET /health` 200
  on both; anonymous and wrong-bearer `POST /mcp` through TLS → **401**; probe through TLS: `tools` lists the 12
  tools (~1.0 s incl. interpreter start), `room_status` returns the fake snapshot, `create_task` returns
  `verdict: act, risk: medium` from the stand-in policy in ~0.7 s, `set_light scene=disco` → readable error,
  `pause_notes` then `transcript` → `count: 0`; `send --health` on console transport counts 25 egress bytes; the
  bearer appears 0 times in either log.
- `bash -n` clean; `shellcheck -x -P SCRIPTDIR -s bash deploy/nemoclaw/*.sh` → exit 0 (shellcheck-py 0.11.0).
- `onboard.sh --dry-run`, `connect.sh --dry-run --firewall --hooks`, `connect.sh --dry-run --fallback-rest`,
  `verify.sh --dry-run --post --agent` → exit 0 and print every `nemoclaw …` command (asserted in tests).
- Security sweep over my paths: no `0.0.0.0`/`[::]` binds; MCP server refuses non-loopback hosts; proxy refuses
  non-private/IPv6/public binds and non-loopback upstreams; tokens < 16 chars refused.

## 4. Not verifiable here → how to verify on the GB10

Everything that touches `nemoclaw`, `openshell`, `openclaw` or Discord. In order, after `onboard.sh`:
1. `docker network inspect openshell-docker -f '{{(index .IPAM.Config 0).Gateway}}'` prints a 172.x/10.x address
   (else set `UNSPOKEN_BRIDGE_IP`; `connect.sh` computes the subnet from it).
2. `connect.sh` → `nemoclaw unspoken mcp status unspoken --json` shows `trustedPrivateTarget.state: match` and a
   verified probe; if the cert is rejected: `openssl x509 -in ~/.unspoken/tls/server.crt -noout -ext subjectAltName`
   must show the bridge IP, and the CA must have been exported before `nemoclaw onboard` (else `rebuild`).
3. `verify.sh --post` → a message appears in `#room-notes` posted by the bot, JSON shows a Discord message id
   (confirms the `openclaw message send --json` shape; if `id` is empty, read the raw output in
   `~/.unspoken/run/` and adjust `find_id()` keys).
4. `verify.sh --agent` → the agent replies with the meeting title (proves tools/list + tools/call through
   OpenShell policy). If OpenClaw rejects JSON responses, set `[mcp] stateless=false json_response=false`.
5. In Discord: `@LocoCisco I'm Priya`, `@LocoCisco off the record`, ❌ on a task post. Check `nemoclaw unspoken logs`
   for the tool calls.
6. Time `nemoclaw unspoken exec -- openclaw message send …` (expect 1–5 s); if slower than the notifier timeout,
   raise `[notifier] timeout` or use `mode = "hooks"` (`connect.sh --hooks`).
7. `sudo ufw status` shows the 8443 allow-from-subnet + deny rules; from another LAN host `curl -k https://<gb10-lan-ip>:8443/health` must fail (the proxy is not bound there anyway).

## 5. Contract changes, deps, config for other owners

Contract changes requested: **none** (coded against `Notifier`, `MessageRef`, `MeetingRuntimeAPI`, `MT.PROPOSAL`/`MT.VERDICT`,
`ChatCommandData`, `ProposalData` as published). Two notes for the integrator/meeting owner, not changes:
- The MCP server submits `create_task`/`set_light` as `Event(MT.PROPOSAL, ProposalData{by:"user", route:"user",
  mode:"act", confidence:1.0, model:"discord", …}, source="bridge.mcp:<user>")` and expects the policy to publish
  `MT.VERDICT` with `proposal_id == event.id` (it answers "pending" after `verdict_wait` otherwise).
- `pause_notes`/`resume_notes`/`checkin`/`undo_item`/`decide_request` go through `runtime.chat()` with commands
  `offrecord` (`args.reason`), `resume`, `here` (`args.name`), `undo` (`args.item_id`), `approve`/`deny`
  (`args.request_id`) — exactly the `ChatCommandData` vocabulary.

Deps added (`requirements/bridge.txt`, all `py3-none-any` wheels → fine on aarch64): `mcp==2.1.1`, `mcp-types==2.1.1`,
`httpx2==2.16.0`, `sse-starlette==3.5.1`, `starlette==1.10.0`, `uvicorn==0.54.0`, `httpx==0.28.1`. Transitive
`cryptography` (via `pyjwt[crypto]`) ships manylinux aarch64 wheels. `mcp` 2.3.0 exists but `strands-agents`
requires `mcp<2.2`, hence 2.1.1. Dev-only: `shellcheck-py==0.11.0.1` (not in the file; tests skip without it).

Config other owners must add / do:
- **meeting**: start the MCP server from the meeting process: `asyncio.create_task(unspoken.bridge.mcp_server.serve(
  runtime, cfg["host"], cfg["port"], os.environ["UNSPOKEN_MCP_TOKEN"]))` with `cfg = mcp_settings(load_bridge_config())`;
  build the notifier with `make_notifier(load_bridge_config(), egress_meter)` and route every outbound post
  through it; include `(task <item_id>)` in task posts so ✅ ❌ / "undo" can name the item.
- **deploy / day-of**: fill `config/bridge.toml [notifier.channels]` with the three Discord channel ids; export
  `DISCORD_BOT_TOKEN`, `DISCORD_SERVER_ID`, `UNSPOKEN_MCP_TOKEN` (and `OPENCLAW_HOOKS_TOKEN` only for hooks mode);
  add `docs/NEMOCLAW.md §4` to `docs/HACKDAY.md` step 3 (onboard) and step 6 (connect/verify).
- **integrator**: merge `requirements/bridge.txt`; the `openclaw`/`discord-rest` transports are selected by
  `UNSPOKEN_NOTIFIER` or `[notifier].transport`.

## 6. Open risks

- The whole NemoClaw side is from docs only (no install here). Highest-risk assumptions: the `openshell-docker`
  bridge gateway is the routable host address; `mcp add` accepts our EC CA/IP-SAN leaf; OpenClaw's MCP client
  accepts stateless JSON responses; `openclaw message send --json` shape; `nemoclaw exec` latency.
- `nemoclaw exec` per post is heavier than an HTTP call; if it is > 5 s, hooks mode (not verbatim) or the REST
  fallback (bypasses the sandbox, token on the host) are the escape hatches.
- Reactions reach the agent only for bot-authored messages (`own`); if the room ever posts via REST while the
  bot is also running, both share one bot identity — do not run both.
- Fallback A puts the MCP bearer inside the sandbox config (readable by the agent); acceptable for the demo,
  documented.
- NemoClaw release drift: command flags were taken from the "latest" docs on 2026-10-24; `--dry-run` output is
  the checklist to diff against `nemoclaw --help` on the day.
