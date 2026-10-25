# SCREEN workstream report (ws/screen)

## 1. What I built

- `unspoken/meeting/web/__init__.py` — exports the fixed interface `create_app(runtime: MeetingRuntimeAPI) -> FastAPI` plus `serve`, `serve_async`, `require_loopback`, `is_loopback`.
- `unspoken/meeting/web/server.py` — FastAPI app: `GET /`, `GET /api/snapshot`, `GET /api/models` (adds `local: bool` per role), `WS /ws` (bus stream: `screen.state`, `meeting.*`, `egress.meter`, `trace.log`, `scenario.status`, `room.env`, `video.scene`; throttled `ui.snapshot`; `ui.ping` heartbeat), `POST /api/chat`, `GET /api/scenarios`, `POST /api/scenario/{id}`, `POST /api/stop`, `POST /api/reset`, `POST /api/settings`. `StreamHub` subscribes `"*"` on the bus inside `create_app` (before agents start), one bounded queue per client. `require_loopback` on `serve`/`serve_async`; no auth by design.
- `unspoken/meeting/web/static/index.html`, `screen.css`, `screen.js`, `xray.js` — the page: privacy state (text + icon), named captions with "1 min wiped" note, agenda + timer bar + banners, cards (task / decision / answer / catch-up / approval / posted), egress meter, model badges (LOCAL / REMOTE / DOWN), Decider X-ray (key `x`), presenter mode (`?present=1` / key `p`), "how it works" card (key `c`), stage controls for rehearsal (key `s`), light/dark, reduced motion, ARIA live regions. No external requests.
- `unspoken/meeting/web/dev_fake.py` — `FakeMeetingRuntime` (implements `MeetingRuntimeAPI`) playing the 3-minute `demo_story` and 8 `beat_*` scenarios in the contract's scenario DSL with realistic events and X-ray fields; CLI `python -m unspoken.meeting.web.dev_fake --port 18751 [--speed N] [--autoplay demo_story]`, loopback only.
- `tests/meeting_web/test_server.py` (13), `test_static.py` (4), `test_dev_fake.py` (8) — every route, the WS stream driven at the ASGI level, loopback guard, offline/privacy/accessibility static checks, the story's own expectations.
- `requirements/screen.txt` — pinned deps.
- `docs/SCREEN.md` — how to run, what is on the page, API, keys, accessibility, privacy.
- `docs/screens/notes-on-captions.png`, `xray-local-llm.png`, `off-the-record.png`, `task-card-presenter-dark.png` — 960×540 captures of the 1920×1080 page.

## 2. How to run it

```bash
cd unspoken-wt/screen   # or the integration checkout
UV_CACHE_DIR=/local/home/zhangnai/kirocrew-sessions/2026-10-15-strands-decider-research/hosting/uv-cache scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/screen.txt      # adds websockets (uvicorn's WS backend)

.venv/bin/python -m unspoken.meeting.web.dev_fake --port 18751           # fake runtime, loopback only
#   open http://127.0.0.1:18751/            room screen (keys: x X-ray, p presenter, c how it works, s stage, ? help)
#   open http://127.0.0.1:18751/?present=1  projector layout;  ?theme=dark|light forces a theme;  ?xray=1 opens the X-ray
curl -s -X POST http://127.0.0.1:18751/api/scenario/demo_story           # or beat_task, beat_off_record, beat_answer, ...
curl -s -X POST http://127.0.0.1:18751/api/chat -H 'content-type: application/json' \
     -d '{"command":"offrecord","user":"Priya"}'                          # rehearsal: inject a Discord command

# in the real runtime (meeting workstream):
from unspoken.meeting.web import create_app, serve_async
app = create_app(runtime)                       # call BEFORE runtime starts agents (hub subscribes to the bus here)
await serve_async(runtime, "127.0.0.1", 8780, app=app)

.venv/bin/python -m pytest -q                   # whole suite
```

## 3. Verification

- `.venv/bin/python -m pytest -q` → **408 passed, 2 skipped in 15 s** (383 existing + 25 new; the 2 skips are pre-existing live tests). No model servers, no hardware.
- Fast playback of `demo_story` (`fast=True`): 62 steps, **11 expectations passed, 0 failed, 0 bus handler errors**, 213 events; every event JSON-able; every proposal action is in `ACTIONS`; egress total equals the bytes of the posted text, audio/video out stay 0.
- Live dev server at speed 1: `GET /api/snapshot` 1.0–1.3 ms; WS connect → first `ui.snapshot` frame 29 ms; `demo_story` runs ~2 min 15 s at 1× (waits + 118 ms decider / 1.84 s LLM think times).
- Visual check with headless Chromium (`playwright-cli`, session `unspoken-screen`) at 1920×1080 and 1280×720, light and dark, presenter mode, X-ray open: fixed a grid-row bug that collapsed the main area when the banner was hidden, an `[hidden]` override on the wiped note, caption/card clipping (now whole lines/cards are trimmed, measured with rects because the entry animation inflated `scrollHeight`), the X-ray's LLM section being below the fold (request JSON folds when escalated), and the header when the X-ray is open. Final captures are in `docs/screens/`.

## 4. Not verified here; how to verify on the GB10

- Kiosk browser on DGX OS (aarch64 Chromium) font rendering of the LocoCisco name and real readability at 3 m on the TV. Verify: `chromium --kiosk --app='http://127.0.0.1:8780/?present=1'` (or Firefox `--kiosk`) while `dev_fake --port 8780 --autoplay demo_story` runs.
- `uvicorn` + `websockets` on aarch64: pure-Python / manylinux aarch64 wheels exist, but check `uv pip install -r requirements/screen.txt` resolves offline from the USB wheel cache.
- The real runtime's event shapes: built against `unspoken/meeting/contracts.py` and the fake only. Verify by running `python -m unspoken.meeting --profile gb10 --mode sim`, opening the page, and checking the X-ray attaches verdicts to proposals (see risk 1).

## 5. Contract change requests, deps, config

Contract change requests (none required; two optional additions for the meeting workstream):
- `ScreenData.agenda` → add `"minutes_total": int` so the timer bar has a denominator (the page otherwise uses the largest `minutes_left` seen since `meeting.status started`).
- `ScreenData.captions[]` → add `"utterance_id": str` so the page keys the new-line animation on the id instead of `t|text`.

Deps added (`requirements/screen.txt`, all pinned; aarch64 status): `fastapi==0.162.0`, `starlette==1.10.0`, `uvicorn==0.54.0`, `httpx==0.28.1`, `anyio==4.15.1`, `h11==0.16.0` (pure Python), `pydantic==2.14.1` (pure) + `pydantic-core==2.50.1` (manylinux aarch64 wheel), **`websockets==15.0.1`** (new; manylinux aarch64 wheel, pure-Python fallback). None lack aarch64 wheels.

Config blocks other owners must add:
- `config/meeting.toml` (meeting): `[screen]` with `host = "127.0.0.1"` and `port = 8780` (dev: 8781), and the runtime must call `create_app(runtime)` before `start_sensors`/agents and pass that `app` to `serve_async`.
- No secrets, no env vars; the page needs nothing from `config/models.toml` (it reads labels from `snapshot["models"]`).

## 6. Open risks

1. The X-ray links verdicts to proposals via `VerdictData.proposal_id == <proposal Event.id>` (as the fake does). If the policy agent uses another id, verdict badges will not appear; the rest of the page is unaffected.
2. Cards render by `card` kind (`task`, `decision`, `answer`, `catch_up`, `approval`, `posted`, `tasks`); unknown kinds render title + `text` generically.
3. The X-ray history lives in the page: a reload shows only decisions made after it (could be prefilled from `bus.history` later).
4. Model labels in the fake are placeholders; the LOCAL/REMOTE badge comes from `RoleHealth.target` (loopback, RFC 1918, link-local, `.local`, `in-process`, or a model path count as local).
5. Snapshot frames are throttled to 5/s per client; a very chatty runtime still streams every `meeting.*` event, which is fine for one kiosk browser but would need a filter for many clients.
