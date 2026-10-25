# MEETING workstream report (ws/meeting)

Status: sim mode runs the 3-minute demo end to end with the real Strands Decider 2B on this box (17/17 expects,
escalation off), the real room screen page renders it live, every scenario passes with the fake profile, and the
whole suite passes in this worktree (715 passed, 6 skipped). Live mode is wired to `unspoken.sensors`,
`unspoken.devices` and `unspoken.bridge` through the fixed interfaces and needs no code change on the GB10.

## 1. What I built

| Path | What |
|---|---|
| `unspoken/meeting/contracts.py` | ADDED `MT.CLOCK` + `ClockData` (see section 5); nothing renamed or removed |
| `unspoken/meeting/base.py` | `MeetingState` (session-only memory: roster, privacy, transcript, items, screen, devices, egress) and the `Agent` base |
| `unspoken/meeting/fusion.py` | segment + speaker frames -> `meeting.utterance`: seat whose face was talking, check-in name <-> voice <-> seat, media when only the screen zone talks, guests, chat `here` binding |
| `unspoken/meeting/reflex.py` | plain-code rules: off-the-record (EN/ZH) wipe + privacy state, check-in name parsing, slides scene -> lamp; the ONE owner of privacy state and the on-record transcript |
| `unspoken/meeting/questions.py` | the typed decider request (act gate, kind, owner, room, sensitive) and `interpret()` -> (action, args, confidence); versions v1-v4 |
| `unspoken/meeting/dates.py` | due dates in code (Friday, tomorrow, EOD, next week, dates, 周三前, 下周一) |
| `unspoken/meeting/cascade.py` | decider agent: one request per utterance; sure -> `local-decider`; unsure -> Strands room agent (`local-llm`); LLM off/failing -> unsure decider answer; skips check-ins and off-record speech |
| `unspoken/meeting/llm_agent.py` | the Strands `RoomAgent` (registry.strands_model("llm")) with one tool per `ACTIONS`, behind `DeciderToolRouter(execute=False)` |
| `unspoken/meeting/policy.py` | risk table + rules (media denied, guests block docs, off-record -> only pause/resume, paused -> no items, recap asks the owner, confidence bands); user proposals get a verdict for their `proposal_id` |
| `unspoken/meeting/actuator.py` | lamp/fan via `Device.apply`, items + `meeting.item`, Notifier posts (task posts carry `(task <item_id>)`), owner DM + approve/deny, undo, recap, docs, `egress.meter` |
| `unspoken/meeting/screen.py` | `screen.state`: named captions (never off-record; wiped lines vanish), agenda + minutes, banners, cards (task, decision, answer, catch_up, approval, posted, notice) |
| `unspoken/meeting/docs.py`, `recap.py` | local doc search (embed cosine, keyword fallback), recap template + optional local-LLM polish |
| `unspoken/meeting/sim.py` | simulated room: people/seats/voices; scenario steps -> the same `audio.segment` / `video.speakers` / `video.scene` / `chat.command` / `meeting.status` / `meeting.clock` events live mode publishes |
| `unspoken/meeting/scenarios/` | runner, matchers (item, privacy, device, post, no_post, no_item, proposal, verdict, screen), validator, `library/` = demo + 20 regression scenarios |
| `unspoken/meeting/runtime.py` | `MeetingRuntime` (MeetingRuntimeAPI): modes sim / live / replay, registry/devices/notifier/sensors wiring with fallbacks, snapshot incl. sensors X-ray |
| `unspoken/meeting/__main__.py`, `play.py` | `python -m unspoken.meeting [--profile] [--mode] [--port]` (room screen via `web.create_app` before agents start + `serve_async`; MCP server when `UNSPOKEN_MCP_TOKEN` is set); `python -m unspoken.meeting.play <ids>` |
| `unspoken/meeting/fakes.py` | fake lamp/plug, recording console notifier, `MeteredNotifier`, fake embedder, `KeywordDecider` (profile fake + eval baseline), `LocalRegistry` fallback for profiles fake / dev / dev-local |
| `unspoken/meeting/config.py`, `config/meeting.toml` | defaults + the config file (thresholds, wipe, devices, channels, scenes, screen host/port, questions version) |
| `examples/docs/*.md` | 3 made-up company docs (pricing proposal, refund policy, launch plan) |
| `eval/meeting_cases.json`, `eval/run_meeting_eval.py`, `eval/MEETING_RESULTS.md` | 72 labelled utterances, the harness, results and the write-up |
| `tests/meeting/*` | 110 tests: reflex/fusion/dates, questions/cascade, policy/actuator/screen, docs/recap/runtime/scenarios (every library scenario), replay of the sensors fixture WAV, the real MCP server driving the runtime |
| `docs/MEETING.md`, `requirements/meeting.txt` | how it works / how to run; pins |

## 2. How to run it

```bash
cd unspoken-wt/meeting && UV_CACHE_DIR=... scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/meeting.txt -r requirements/screen.txt   # websockets for the page's /ws

.venv/bin/python -m pytest -q                                            # 715 passed, 6 skipped here
.venv/bin/python -m unspoken.meeting.play --profile fake                 # 21/21 scenarios, no servers
.venv/bin/python -m unspoken.meeting.play demo_three_minutes --profile dev --no-escalate --speed 1   # real 2B/12B on :8765/:8766
.venv/bin/python -m unspoken.meeting --profile fake --mode sim --port 8781   # room screen http://127.0.0.1:8781/, click a scenario
.venv/bin/python -m unspoken.meeting --profile dev --mode sim --port 8781 --no-escalate
.venv/bin/python eval/run_meeting_eval.py --model 2b --version v3        # the eval (appends to eval/MEETING_RESULTS.md)
# GB10:
UNSPOKEN_PROFILE=gb10 UNSPOKEN_MCP_TOKEN=<token> .venv/bin/python -m unspoken.meeting --mode live
```

## 3. Verification (what I ran, here)

* `pytest -q` in this worktree after merging hack/main (sensors, devices, screen, deploy, bridge): **715 passed,
  6 skipped** (383 original + the merged workstreams' + 110 meeting tests; the skips are the merged suites'
  model-file tests). Meeting tests alone: 110 passed in ~2 s, no model servers, no hardware.
* `python -m unspoken.meeting.play --profile fake --speed 50`: **21/21 scenarios passed** (demo + 20 regressions).
* `python -m unspoken.meeting.play demo_three_minutes --profile dev --no-escalate --speed 1` with the real
  Strands Decider 2B (CPU, :8765), questions v3, thresholds 0.60/0.40: **PASS 17/17 in 87 s**. Decider rows:
  create_task 0.81 (10.0 s), set_fan 0.58 (12.0 s, acted as low risk), share_docs 0.62 (8.9 s, denied: guest),
  create_task 0.72 (8.1 s). With v2 and 0.80/0.50 the same demo failed 6 expects (proposals at 0.46-0.50 were
  ignored) — the eval drove both changes.
* Room screen: `python -m unspoken.meeting --profile fake --mode sim --port 8781` + `POST /api/scenario/
  demo_three_minutes`; the real page (unspoken/meeting/web) opened with playwright-cli shows live status,
  named captions (Priya / Sam), agenda + timer, the "2 min left. 1 task has no owner." banner, task / notice /
  recap-posted cards, egress "Audio out 0 B · Video out 0 B · Text 529 B", models LOCAL. `/ws` needs the
  `websockets` package (requirements/screen.txt); without it the page polls.
* Eval (eval/MEETING_RESULTS.md): keyword baseline 85%; 2B v2 60%; **2B v3 78% (89% at >= 0.6, 75%
  coverage, 97% act precision; decisions 8/8, Mandarin 8/8, media 3/3)**; 2B v4 worse on the same prefix,
  stopped; 12B v3 90% on the first 31 cases, 100% act precision (full run still going at writing time; its row
  is appended automatically). Latency per request (16-20 scored options): 2B ~9.3 s alone, 12B ~40 s on this CPU.
* Replay: tests/sensors/fixtures/meeting3.wav (3 synthetic voices, 12 lines) through the real Silero VAD +
  scripted ASR + fake diarizer into the full pipeline: 12 named utterances (Priya / Sam / Lee check-ins bind
  the voices), task for Sam with due date, March-launch decision, "off the record" wipe + red lamp
  (`tests/meeting/test_docs_runtime_scenarios.py::test_replay_meeting3_wav_through_the_pipeline`).
* Bridge: the real MCP server (bearer-gated Streamable HTTP) over loopback drives the runtime with the `mcp`
  client: create_task / set_light proposals get verdicts for their proposal ids, checkin / pause / resume /
  undo / recap / search_docs work; the bridge console Notifier posts carry `(task <id>)` and meter egress.

## 4. Not verified here; how to verify on the GB10

* Live sensors (mic + camera) feeding fusion; the registry from `unspoken.models` (`load_registry("gb10")`);
  the real WiZ lamp / Kasa fan; Discord through NemoClaw/OpenClaw. Steps: `UNSPOKEN_PROFILE=gb10 python -m
  unspoken.models check` (all green), `python -m unspoken.meeting --mode live --port 8780`, open the room
  screen, say "Hi, I'm Priya", "Sam, send the deck by Friday", "it's freezing in here", "off the record";
  check captions carry names, #tasks gets the post, the fan and lamp move, the lamp turns red and the
  caption disappears. Then `python -m unspoken.meeting.play demo_three_minutes --profile gb10 --speed 1`
  (sim people, real lamp/fan/Discord): expect 17/17 with the 12B at well under a second per decision.
* LLM escalation (`escalate = true`, llm role on Qwen3.6 via vLLM): tested only with `ScriptedModel`
  (strands) here; no OpenAI-compatible server was running on 127.0.0.1:8010. On the GB10 run
  `play demo_three_minutes --profile gb10` without `--no-escalate` and look for rows with route `local-llm`
  and tokens > 0 only when the decider was below 0.60.
* 12B accuracy on all 72 cases: `python eval/run_meeting_eval.py --model 12b --version v3` (fast on the GPU);
  re-pick `act_threshold` from its table if it differs from the 2B's 0.60.
* Recap polish by the local LLM (`recap_markdown` with the llm role); falls back to the template on any error.

## 5. Contract changes, deps, config for other owners

Contract ADDITIONS in `unspoken/meeting/contracts.py` (meeting-owned additions, nothing renamed):
* `MT.CLOCK = "meeting.clock"` + `ClockData {t, minutes_left, ends_at}`: the sim's `clock` step and live mode
  publish the meeting clock so the screen agent counts down without owning a timer.

Payload fields I ADD beyond the TypedDicts (all `total=False`, nothing removed), please fold into contracts.py:
* `ProposalData`: `speaker`, `source`, `owner`, `owner_confidence`, `kind_probabilities` (X-ray / policy).
* `VerdictData`: `by`, `utterance_id`, `speaker`, `reason`, `confidence` (the actuator writes who proposed on
  items and DMs). `proposal_id` == the proposal `Event.id` as the MCP server and screen require.
* `StatusData` on `started`: `agenda: list[str]`, `minutes: float` (planned length). `ScreenData.agenda.
  minutes_total` (int) and `captions[].utterance_id` as the screen workstream asked. Card kinds used: task,
  decision, answer, catch_up, approval, posted, plus `notice` (title + text; the page's generic renderer shows it).
* `ChatCommandData.args` for `undo`: `item_id` may be `"$last_task"`.

Deps (`requirements/meeting.txt`): `strands-agents==1.62.0`, `httpx==0.28.1` (both pure Python, already in
requirements-dev.txt; no compiled aarch64 wheel needed). `websockets==15.0.1` comes from requirements/screen.txt
and is needed for the room screen's live stream. No new packages lacking aarch64 wheels.

Config other owners need: `config/meeting.toml [devices] lamp="lamp" fan="fan"` must match `config/room.toml`
device ids; `[channels] tasks / notes` must be keys in `config/bridge.toml [notifier.channels]`; the gb10
profile in `config/models.toml` needs `decider` (12B), `llm` (Qwen3.6 on vLLM) and optionally `embed` for
doc search (keyword fallback otherwise). The lamp scenes used: talk, slides, break, privacy, off_record, bright,
dim (all in unspoken/devices/scenes.py).

## 6. Open risks (short)

* Decider latency on CPU is 9 s (2B) / 40 s (12B) per request because one request scores ~16-20 options;
  the GB10 GPU should make it sub-second, but the 45 s scenario windows hide this on the dev box.
* 2B accuracy on factual questions is poor (3/9); the 12B is the demo decider. v4 wording did not fix it.
* Off-record text still exists in the in-memory bus history (and in `--trace` files): the Bus is
  integrator-owned; keep tracing off in real meetings, or let the Bus drop `audio.segment` / `meeting.utterance`
  payloads marked off-record.
* `pause_notes` from the decider (sensitive hint) wipes `wipe_seconds` of transcript on a 0.6+ confidence
  call; false positives cost notes (the owner can `/resume`).
* The approval DM goes to `config/meeting.toml [meeting] owner` by display name; the bridge must map it to a
  Discord user. Approvals expire after 300 s.
* Live mode publishes `meeting.status started` with the config title/owner; there is no spoken "start the
  meeting" yet (use `/agenda` or the config).
