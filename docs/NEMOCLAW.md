# LocoCisco on NemoClaw + OpenClaw + OpenShell, with Discord

How the room brain (host process `unspoken`) plugs into the NemoClaw stack on the GB10, what each piece is
allowed to reach, the exact day-of commands, and what to check when something does not come up. Everything
here was built from the public docs cited inline; **nothing was run against a real NemoClaw install** (none on
the dev desktop). Every step marked `[DAY-OF CHECK]` needs confirming on the box; §9 lists them all.

Sources used: NemoClaw docs index https://docs.nvidia.com/nemoclaw/latest/llms.txt (append `.md` to any page
for Markdown); OpenClaw docs https://docs.openclaw.ai (source: https://github.com/openclaw/openclaw/tree/main/docs).

## 1. Architecture

```
 GB10 host (DGX OS 7, Docker)                                  │ NemoClaw sandbox "unspoken" (OpenShell)
                                                               │
 unspoken (meeting runtime, 127.0.0.1 only)                    │  OpenClaw gateway + agent "main"
   ├─ Notifier = OpenClawNotifier ──► nemoclaw unspoken exec ──┼──► openclaw message send --channel discord ──► Discord
   │      (text, verbatim; message ids back)                   │        (bot token lives in OpenShell, never on our side)
   ├─ MCP server 127.0.0.1:8791  ◄── TLS proxy <bridge-ip>:8443 ◄┼──── OpenShell proxy ◄── agent tool call (mcp.servers.unspoken)
   │      bearer UNSPOKEN_MCP_TOKEN      bearer + private CA   │        policy: pinned host/IP, /mcp, MCP methods only
   └─ decider / vLLM: https://inference.local ◄────────────────┼──── nemoclaw-vllm :8000 (Qwen3.6-35B-A3B-NVFP4), bearer managed by NemoClaw
                                                               │
 Discord (channel messages, reactions, @mentions) ─────────────┼──► Discord gateway (native OpenClaw channel; mention-gated)
```

| Direction | Mechanism | Why this one |
|---|---|---|
| Room → Discord (notes, tasks, recaps) | `nemoclaw <sb> exec -- openclaw message send --channel discord --target channel:<id> --message=<text> --json` ([exec](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/reference/commands.md#nemoclaw-name-exec), [message send](https://github.com/openclaw/openclaw/blob/main/docs/cli/message.md)) | Verbatim delivery, message id back, threads via `message thread create`, reactions land on bot-authored messages. The OpenClaw webhook (`/hooks/agent`) is an **agent turn**, not a verbatim post ([webhooks](https://github.com/openclaw/openclaw/blob/main/docs/automation/cron-jobs/webhooks.md)): kept as mode `hooks` for a room that only needs "tell Discord roughly this". |
| Discord → Room (questions, check-ins, undo, approvals) | OpenClaw agent calls our tools over **managed MCP** (`nemoclaw <sb> mcp add`) ([add-an-mcp-server](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/manage-sandboxes/mcp-servers/add-an-mcp-server.md)) | NemoClaw stores the bearer in OpenShell and injects it at egress; policy pins host, IPs, path and MCP methods. |
| Reactions ✅ ❌ | `channels.discord.guilds.<id>.reactionNotifications` (default `own`): reactions on the bot's own messages become system events in the routed Discord session ([events](https://github.com/openclaw/openclaw/blob/main/docs/channels/discord/events.md)) | The room posts tasks AS the bot (exec path), so ✅ ❌ on a task reach the agent, which calls `undo_item` / `decide_request`. |
| Persona + skill | `nemoclaw <sb> upload <file> /sandbox/.openclaw/workspace/<FILE>.md` and `nemoclaw <sb> skill install <dir>` ([sandbox state](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/manage-sandboxes/state-and-backups/understand-sandbox-state.md), [skill install](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/reference/commands.md#nemoclaw-name-skill-install-path)) | Workspace files `SOUL.md AGENTS.md TOOLS.md IDENTITY.md USER.md` are loaded every session; skills need `SKILL.md` with `name` + `description` frontmatter ([skills](https://github.com/openclaw/openclaw/blob/main/docs/tools/skills.md)). |
| Inference | NemoClaw-managed vLLM, `NEMOCLAW_PROVIDER=install-vllm NEMOCLAW_VLLM_MODEL=qwen3.6-35b-a3b-nvfp4` ([set-up-vllm](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/inference/local-inference/set-up-vllm.md)) | The GB10 is the DGX Spark / N1x host profile whose default is `nvidia/Qwen3.6-35B-A3B-NVFP4`. The container publishes :8000 on `127.0.0.1` **and** the `openshell-docker` bridge gateway only; inside the sandbox the route is `https://inference.local`. |

### Research results (what the docs say, verbatim where it matters)

1. **Host → Discord verbatim.** `openclaw message send` is "the single outbound command for sending messages";
   Discord targets are `channel:<id>`, `user:<id>` or a bare numeric id; `--reply-to`, `--thread-id`, `--json`
   are supported; `message thread create --target <channel id> --thread-name ... --message-id ...` and
   `thread reply --target <thread id>` exist for Discord. From the host it runs through
   `nemoclaw <sb> exec -- <cmd>`, which "runs as the sandbox user with `HOME=/sandbox`" so the agent's
   `openclaw.json` (Discord token placeholder, resolved by OpenShell at egress) is used. `--no-stdin` keeps one-shot
   runs from waiting on a pipe. The message text is one argv element: no shell is involved on either side.
   `[DAY-OF CHECK]` exact JSON shape of `--json` (we read any `messageId`/`id` key recursively) and whether
   NemoClaw's exec cleanup adds latency (docs: it restores the config permission contract after each one-shot).
2. **Hooks.** `POST /hooks/agent` on the gateway (`hooks.enabled`, `hooks.token`, bearer header) submits an agent
   turn; with `channel` + `to` the agent's reply is delivered to that Discord channel. HTTP 200 = admitted, not
   delivered. `/hooks/wake` queues a system event for the main session. NemoClaw forwards the OpenClaw gateway to
   `127.0.0.1:18789` on the host (its "dashboard port"; `nemoclaw <sb> gateway-token` authenticates the Control
   UI there), so hooks would be reachable at `http://127.0.0.1:18789/hooks/agent` **if** the same forward carries
   `/hooks` `[DAY-OF CHECK]`. Enabling hooks means patching `openclaw.json` inside the sandbox
   (`openclaw config patch --file`), which `connect.sh --hooks` does with a token from `OPENCLAW_HOOKS_TOKEN`.
3. **How the agent sees Discord.** Guild channels are isolated sessions `agent:main:discord:channel:<id>`; with
   `DISCORD_SERVER_ID` set NemoClaw defaults `DISCORD_REQUIRE_MENTION=1` (reply only when @mentioned); reactions
   per `reactionNotifications` (default `own`); message ids are surfaced in context so the agent can target them;
   `[[reply_to:<id>]]` tags make native replies. Threads: the bot needs **Send Messages in Threads**.
4. **Where instructions and skills live.** `/sandbox/.openclaw/workspace/` (`~` is `/sandbox`, not the workspace);
   NemoClaw seeds six template files unless `NEMOCLAW_MINIMAL_BOOTSTRAP=1` (we set it and upload ours).
   `nemoclaw <sb> skill install ./dir/` runs `openclaw skills install <staged> --agent main --force`; names:
   `[A-Za-z0-9._-]{1,64}`, dotfiles skipped, symlinks rejected. Workspace `skills/` has the highest precedence.
5. **Which host address the sandbox can reach.** NemoClaw's default Docker-driver topology is one Docker
   container, not k3s. `host.docker.internal` / `host.openshell.internal` are "not a general-purpose host-service
   path" and are **rejected** for authenticated MCP. Supported path: "Expose the service on a host IP address that
   the OpenShell gateway can reach, create a custom NemoClaw policy preset for that IP and port" — the managed
   vLLM docs name that address: "the private IPv4 gateway of the `openshell-docker` bridge". `connect.sh` reads it
   with `docker network inspect openshell-docker -f '{{(index .IPAM.Config 0).Gateway}}'` (falls back to
   `bridge`/`docker0`; override `UNSPOKEN_BRIDGE_IP`). `[DAY-OF CHECK]` the network name and that the gateway
   routes to it (a `policy_denied → upstream_unreachable` change in `openshell term` means policy matched but the
   IP is wrong).
6. **mcp add rules.** HTTPS only; loopback, `::1`, link-local, metadata, IPv6 literals and `host.*.internal`
   rejected; a private IPv4 literal URL needs a certificate with a matching **IP SAN**; the chain must reach a CA
   the sandbox trusts via `NEMOCLAW_CORPORATE_CA_BUNDLE` **before onboarding** (rebuild after changing it); the
   bundle must be a regular non-symlink PEM with only `CA:TRUE` certs, not group/world writable; exactly one
   `--env <TOKEN_VAR>` bearer per server (dedicated name, not `OPENCLAW_*`/`NEMOCLAW_*`); `--trusted-private-host`
   must equal the URL host; `nemoclaw <sb> mcp status <name> --json` must show `trustedPrivateTarget.state: match`.
   The generated policy allows `initialize`, `tools/list`, `tools/call`, resources, prompts, ping, and a
   131,072-byte request body — only from the agent's runtime binaries, so **`curl` inside the sandbox failing is
   not evidence** of a broken route; use `mcp status`.
7. **Minimal policy.** Tier `restricted` (no tier presets) + the `discord` preset (applied automatically for an
   enabled channel, re-applied idempotently by `connect.sh`) + the policy `mcp add` generates for our host.
   Nothing else. For the REST fallback the custom preset `nemoclaw/policy/unspoken-mcp-host.yaml` allows
   GET/POST/DELETE `/mcp` on `<bridge-ip>:8443` from `/usr/local/bin/node` only (confirm the binary with
   `nemoclaw unspoken exec -- which node`); user presets must not contain `allowed_ips` — NemoClaw pins them from
   `--trusted-private-host`.
8. **Channels are frozen into the image.** `openclaw channels add` inside the sandbox is intercepted; use
   `nemoclaw <sb> channels add discord` on the host (it rebuilds). Export `DISCORD_BOT_TOKEN`,
   `DISCORD_SERVER_ID`, `DISCORD_REQUIRE_MENTION` before onboarding; NemoClaw rejects the literal placeholder.

## 2. What the bridge workstream ships

| Path | What |
|---|---|
| `unspoken/bridge/notifiers.py` | `make_notifier(config, on_egress)`; `OpenClawNotifier` (exec / hooks), `DiscordRestNotifier`, `ConsoleNotifier`; chunking at 1900 chars; threads for `kind="task"`; retries with backoff; never raises |
| `unspoken/bridge/mcp_server.py` | `async serve(runtime, host, port, token)`, `create_app()`; Streamable HTTP (mcp 2.1.1), bearer gate, loopback only; 12 tools over `MeetingRuntimeAPI` |
| `unspoken/bridge/tls_proxy.py` | TLS front on the bridge IP only → `127.0.0.1:8791`; same bearer; `/mcp` GET/POST/DELETE + `/health` only |
| `unspoken/bridge/probe.py`, `send.py`, `dev_fake.py` | host-side MCP client for checks; one-shot test post; the fake runtime (`--runtime fake`) |
| `config/bridge.toml` | channel ids, transport, retries, MCP/TLS ports. **No secrets** (env: `DISCORD_BOT_TOKEN`, `OPENCLAW_HOOKS_TOKEN`, `UNSPOKEN_MCP_TOKEN`) |
| `deploy/nemoclaw/{make-ca,onboard,connect,verify}.sh` + `lib.sh` | the scripted day: all idempotent, `--dry-run` prints every command |
| `nemoclaw/workspace/*.md` | SOUL / AGENTS / TOOLS / IDENTITY / USER for the Discord persona |
| `nemoclaw/skills/unspoken-room/SKILL.md` | the skill (triggers → tools) |
| `nemoclaw/policy/unspoken-mcp-host.yaml` | custom preset for Fallback A |

## 3. Discord Developer Portal (once, before the day)

Source: [OpenClaw Discord setup](https://github.com/openclaw/openclaw/blob/main/docs/channels/discord/setup.md),
[NemoClaw Set Up Discord](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/manage-sandboxes/messaging-channels/set-up-discord.md).

1. https://discord.com/developers/applications → **New Application** "LocoCisco" → **Bot** → Username `LocoCisco`.
2. **Bot → Privileged Gateway Intents**: **Message Content Intent ON** (required for guild messages),
   **Server Members Intent ON** (recommended: name→id matching). Presence off.
3. **Bot → Reset Token** → copy → this is `DISCORD_BOT_TOKEN` (store in a password manager, never in the repo).
4. **OAuth2 → URL Generator**: scopes `bot` + `applications.commands`; Bot Permissions: View Channels, Send
   Messages, **Send Messages in Threads**, Create Public Threads, Read Message History, Embed Links, Attach Files,
   Add Reactions. Open the URL, pick the server, **Continue**.
5. Discord app → User Settings → Advanced → **Developer Mode ON**. Right-click the server → **Copy Server ID**
   (`DISCORD_SERVER_ID`). Create `#room-notes`, `#room-tasks`, `#room-recap`; right-click each → **Copy Channel ID**
   → put them in `config/bridge.toml [notifier.channels]`.
6. Server → Privacy Settings → Direct Messages ON (needed only if we DM the owner the recap).
7. Optional lockdown: `DISCORD_USER_ID=<your id>` restricts the bot to one user (NemoClaw env).

## 4. Day-of commands, in order (GB10)

Prerequisites: NemoClaw installed (`curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash`, Node ≥ 22.19, user in
the `docker` group, OpenShell 0.0.116), repo checked out, `scripts/dev-venv.sh` run, `uv pip install -r requirements/bridge.txt`.

```bash
cd ~/unspoken
export NEMOCLAW_DISABLE_TELEMETRY=1                       # optional
export DISCORD_BOT_TOKEN='<from the portal>'               # never echoed by our scripts
export DISCORD_SERVER_ID='<server id>' DISCORD_REQUIRE_MENTION=1
export UNSPOKEN_MCP_TOKEN="$(openssl rand -hex 32)"        # one bearer: MCP server, TLS proxy, sandbox

# 1. sandbox + Discord + managed vLLM (10-40 min: image 2.4 GB + model download unless restored from USB)
deploy/nemoclaw/onboard.sh --dry-run     # read it
deploy/nemoclaw/onboard.sh               # == 1/5 .. 5/5; ends with `nemoclaw unspoken status`

# 2. room ↔ sandbox (MCP server here is the fake unless the meeting process already serves 127.0.0.1:8791)
deploy/nemoclaw/connect.sh --dry-run
deploy/nemoclaw/connect.sh --runtime fake --firewall     # a) .. f); --hooks only if bridge.toml mode = "hooks"

# 3. prove it
deploy/nemoclaw/verify.sh --post --agent
```

Expected output (abridged):

```
[unspoken] == 4/5 NemoClaw onboarding: agent openclaw, sandbox unspoken, tier restricted, managed local vLLM
+ nemoclaw onboard --name unspoken --non-interactive --yes --yes-i-accept-third-party-software
   ... baking corporate proxy CA from /home/<you>/.unspoken/tls/ca.pem ...      <- the CA made it into the image
   ... Sandbox 'unspoken' is Ready ...
[unspoken] bridge ip 172.18.0.1 (subnet 172.18.0.0/16); MCP url for the sandbox: https://172.18.0.1:8443/mcp
[unspoken] proxy up; anonymous POST https://172.18.0.1:8443/mcp -> 401 (good)
+ nemoclaw unspoken mcp add unspoken --url https://172.18.0.1:8443/mcp --env UNSPOKEN_MCP_TOKEN --trusted-private-host 172.18.0.1
   ... probe: verified ...                                                        <- credential resolution probe
+ nemoclaw unspoken mcp status unspoken        -> trustedPrivateTarget.state: match, provider/policy/adapter ready
+ nemoclaw unspoken policy add discord --yes   -> "already effective" or the Discord egress scope
+ nemoclaw unspoken skill install .../nemoclaw/skills/unspoken-room/
[unspoken] PASS  MCP server /health on 127.0.0.1:8791
[unspoken] PASS  TLS proxy /health on 172.18.0.1:8443
[unspoken] PASS  anonymous POST https://172.18.0.1:8443/mcp is denied with 401
[unspoken] PASS  tools/list via https://172.18.0.1:8443/mcp      {"tools": ["room_status", "transcript", ...]}
[unspoken] PASS  nemoclaw unspoken mcp status unspoken: trustedPrivateTarget.state == match
[unspoken] PASS  test post through UNSPOKEN_NOTIFIER=openclaw     {"message": {"id": "1431…", ...}, "egress": [{"dest": "openclaw", "bytes": 61}]}
[unspoken] PASS  openclaw agent calls room_status                 Pricing sync
[unspoken] all checks passed
```

Then in Discord: `@LocoCisco I'm Priya` → "Welcome, Priya." · `@LocoCisco what did we decide?` → the decisions ·
`@LocoCisco off the record` → "Notes paused." · ❌ on a task post → "Undone."

Run the room with the real notifier: `UNSPOKEN_NOTIFIER=openclaw python -m unspoken.meeting --profile gb10 --mode live`
(the meeting process must call `unspoken.bridge.mcp_server.serve(runtime, "127.0.0.1", 8791, token)` so the
tools answer from the live room; `connect.sh` then finds `/health` already up and does not start the fake).

## 5. Ports and addresses

| Where | Address | Who connects |
|---|---|---|
| MCP server | `127.0.0.1:8791` (dev desktop `8792`) | TLS proxy only |
| TLS proxy | `<openshell-docker gateway>:8443`, e.g. `172.18.0.1:8443` | the sandbox (through the OpenShell proxy); firewall: that subnet only |
| OpenClaw gateway forward | `127.0.0.1:18789` (host) | `nemoclaw <sb> dashboard-url`, hooks mode |
| vLLM | `127.0.0.1:8000` + bridge gateway `:8000` (bearer) | the room (OpenAI API) and the sandbox (`https://inference.local`) |

Firewall (ufw, DGX OS): `sudo ufw allow proto tcp from <subnet> to <bridge-ip> port 8443 comment unspoken-mcp-proxy`
then `sudo ufw deny 8443/tcp` (`connect.sh --firewall` runs both; without the flag it prints them). nftables:
`nft add rule inet filter input ip saddr <subnet> tcp dport 8443 accept; nft add rule inet filter input tcp dport 8443 drop`.

## 6. Fallbacks

**A. Managed MCP will not converge** (`mcp add` rejects the host, pins drift, probe never verifies):
`deploy/nemoclaw/connect.sh --fallback-rest`. It applies `nemoclaw/policy/unspoken-mcp-host.yaml` with
`--trusted-private-host <bridge-ip>` (dry-run first, review the generated `allowed_ips`), then registers the
server natively: `nemoclaw unspoken exec -- openclaw config set mcp.servers.unspoken.url https://<ip>:8443/mcp`
and `... mcp.servers.unspoken.headers.Authorization "Bearer <token>"` (type the token inside `nemoclaw unspoken
connect` so it never appears in host logs), `nemoclaw unspoken gateway restart`. The token is then readable
inside the sandbox; acceptable for the demo, say so. `[DAY-OF CHECK]` the exact `mcp.servers.<name>` key shape
OpenClaw expects for URL servers with headers ("OpenClaw stores MCP servers in `openclaw.json` under
`mcp.servers`" — add-an-mcp-server.md).

**B. NemoClaw itself is down**: `UNSPOKEN_NOTIFIER=discord-rest` with `DISCORD_BOT_TOKEN` exported on the host.
The room posts straight to `https://discord.com/api/v10` with the same bot identity (do not run both at once);
inbound Discord commands are then unavailable (no agent), the room screen and voice still work. The egress meter
still counts every byte (`dest: discord`).

**C. exec too slow or unavailable**: `[notifier.openclaw] mode = "hooks"` + `connect.sh --hooks`. The agent relays
the text (not guaranteed verbatim; no thread ids come back).

## 7. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `onboard.sh` dies "refusing to issue a cert for 240.x / 8.8.8.8" | `bridge_ip` picked a non-RFC1918 docker gateway. `docker network ls`; set `UNSPOKEN_BRIDGE_IP` to the `openshell-docker` gateway. |
| `nemoclaw onboard` fails on the CA bundle | Bundle must be a regular file, CA-only, mode 0644 in a 0700 dir: `make-ca.sh --force`. Merged OS trust stores are rejected. |
| `mcp add` → "loopback … rejected" / "host.docker.internal rejected" | URL must be the bridge IP literal with `--trusted-private-host <same ip>`; never 127.0.0.1 or a `.internal` name. |
| `mcp add` → certificate / TLS error | Leaf SAN must be `IP:<bridge-ip>` (`openssl x509 -in ~/.unspoken/tls/server.crt -noout -ext subjectAltName`); CA baked in **before** onboarding, else `nemoclaw unspoken rebuild`. |
| `mcp status` → provider not ready / probe skipped | Rerun `nemoclaw unspoken mcp status unspoken --json`; `--env` var must have been exported in the same shell as `mcp add`; token name must not be `OPENCLAW_*`/`NEMOCLAW_*`. |
| `mcp status` → `trustedPrivateTarget.state != match` | Host in `--url` and `--trusted-private-host` differ, or the IP changed (docker restart). `mcp remove unspoken` then `connect.sh` again. |
| agent says the tool failed; `openshell term` shows `policy_denied` | Policy did not match: wrong path (`/mcp`), method, or binary; for Fallback A confirm `which node`. |
| `openshell term` shows `upstream_unreachable` | Policy matched but the gateway cannot reach `<ip>:8443`: proxy not running (`~/.unspoken/run/unspoken-tls.log`), firewall rule missing/wrong subnet, wrong bridge IP. |
| `curl https://<ip>:8443/mcp` from inside the sandbox fails | Expected: the generated policy allows the agent runtime binaries only. Use `nemoclaw unspoken mcp status unspoken`. |
| proxy answers 200 to an anonymous POST | **Stop.** Token not set in the proxy's env; restart with `UNSPOKEN_MCP_TOKEN` (`connect.sh` dies on this on purpose). |
| `openclaw message send` → "channel discord is not configured" | Discord not enabled at onboarding: `nemoclaw unspoken channels add discord` (rebuilds), then `policy add discord --yes`. |
| bot logs in, nothing posts / no replies | Native gateway path (`gateway.discord.gg`) blocked: keep the `discord` preset applied, `openshell term` for blocked requests; Message Content Intent off → enable in the portal. |
| bot replies to nothing in the channel | `DISCORD_REQUIRE_MENTION=1`: @mention it, or re-onboard with `DISCORD_REQUIRE_MENTION=0`. |
| `nemoclaw unspoken exec` slow (> 5 s) | Expected per docs (permission-contract cleanup after one-shot commands) `[DAY-OF CHECK]`; raise `[notifier] timeout`, or switch to `mode = "hooks"`. |
| ✅ ❌ reactions ignored | `reactionNotifications` default `own` covers bot-authored messages only; reactions on human messages need `all`; the agent needs the message id → task id mapping, so posts carry `(task t1)`. |
| hooks → 404 | `hooks.enabled` false or path mismatch; 401 → wrong `OPENCLAW_HOOKS_TOKEN` (not the gateway token). |
| re-onboard fails "Port 18789 … Blocked by: ssh" | stale dashboard forward; current NemoClaw kills it itself; else `sudo lsof -i :18789 -sTCP:LISTEN`. |
| persona ignored | files must be in `/sandbox/.openclaw/workspace/`, not `/sandbox/`; `nemoclaw unspoken exec -- ls /sandbox/.openclaw/workspace`. |

Logs: `~/.unspoken/run/unspoken-mcp.log`, `~/.unspoken/run/unspoken-tls.log`, `nemoclaw unspoken logs`,
`nemoclaw unspoken connect` then `tail -f /tmp/gateway.log`, `openshell term` for blocked requests.

## 8. Security posture (what leaves the box, and who can reach what)

- Only text leaves the box, only to Discord, only through the sandbox (or, in fallback B, the REST notifier);
  every post goes through `on_egress(dest, "text", nbytes)` and the meter is shown on the room screen.
- The single network-facing listener is the TLS proxy: private bridge IP only, TLS with our own CA, bearer
  checked before forwarding, `/mcp` + `/health` only, firewall limited to the sandbox subnet. The MCP server and
  the room screen stay on loopback. `verify.sh` fails if anything listens on `0.0.0.0`/`::` on our ports or if an
  anonymous POST is not a 401.
- Secrets never touch the repo: Discord token lives in OpenShell (and only in the host env for fallback B); the
  MCP bearer is in the host env + OpenShell's provider store; make-ca.sh keys are in `~/.unspoken/tls` (0700).
- Tool descriptions and the persona both state that off-record content never exists; `transcript()` returns
  on-record lines only because the runtime never stored the others.

## 9. Unverified until the GB10 (`[DAY-OF CHECK]` list)

1. `docker network inspect openshell-docker` exists on the NemoClaw Docker-driver topology and its gateway IP is
   routable from the sandbox (else `UNSPOKEN_BRIDGE_IP`, and `openshell term` will say `upstream_unreachable`).
2. `nemoclaw unspoken mcp add` accepts our cert/CA combination (IP SAN leaf, EC P-256 CA baked via
   `NEMOCLAW_CORPORATE_CA_BUNDLE`) and `mcp status` reports `match`.
3. OpenClaw's native MCP client accepts stateless JSON responses (`[mcp] stateless = true, json_response = true`);
   if it insists on SSE sessions, set both to `false` in `config/bridge.toml` and restart the server.
4. `openclaw message send --json` output shape (we accept any `messageId`/`id`), `--message=<text>` form, and
   `message thread create` flags on the installed OpenClaw version.
5. `nemoclaw <sb> exec` latency and that it tolerates ~10 one-shot posts per minute.
6. The `/hooks` routes are reachable on the forwarded `127.0.0.1:18789` (hooks mode only).
7. The GB10 is detected as DGX Spark or N1x and `NEMOCLAW_PROVIDER=install-vllm NEMOCLAW_VLLM_MODEL=qwen3.6-35b-a3b-nvfp4`
   is accepted (on N1x it is the "Deferred preview"); otherwise `onboard.sh --profile <id from nemoclaw profiles list>`.
8. `NEMOCLAW_MINIMAL_BOOTSTRAP=1` + our uploads produce a workspace OpenClaw loads (else upload without the flag).
9. Reaction events reach the agent for messages posted via `openclaw message send` (bot-authored, `own`).
10. ufw is the active firewall on DGX OS 7 (else use the printed nftables rules).
