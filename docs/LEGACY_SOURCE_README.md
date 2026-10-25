# LocoCisco: local meeting assistant

No wake word: you never say its name.

LocoCisco is the product name; Python packages, CLI commands, environment variables and deployment
identifiers retain `unspoken` for compatibility.

## Browser demo on the Dell GB10

The current build uses a local meeting video's audio and frames in place of a USB microphone and
camera. Speech, speaker, vision and decision inference run locally using the already-staged models.
The browser provides replay, privacy controls, explanations, questions and recording/export.
Devices and Discord are visibly marked as local rehearsal previews.

Start the prepared stack with `deploy/demo/up.sh` and open **http://127.0.0.1:8780/**.
See the [browser demo guide](BROWSER_DEMO.md) for the credential-file setting, recording,
offline preparation and sample attribution. The launcher does not download models, pull images,
or install packages. The meeting build has no cloud inference fallback.

The [meeting plan](PLAN.md) and [deployment runbook](HACKDAY.md) provide broader hardware
and integration context.

## Legacy smart-home prototype

The material below describes the earlier prototype and its optional Bedrock experiments.
Use the browser demo guide above for the current local meeting build.

A simulated home whose brain is a set of small agents talking over one event bus. The
Strands Decider 2B answers bounded questions ("what does the speaker want?", "what should the
house do about what it sees?") inside ONE Strands house agent: it sits in the agent's path as a
`DeciderToolRouter` intervention and picks the tool; when it is sure, Claude Haiku 4.5 on Amazon
Bedrock is never called (0 tokens, nothing leaves the house); when it is unsure, Claude decides with
the same tools. Plain code keeps the safety rules.

```
mocked sensors ──► perception ──► reflex rules ──────────────────────┐
 (camera, presence,     │                                            ├─► policy ──► actuator ──► simulated devices
  lux, temp, mic)       └──► scene decider ─┐                        │      (code)      │
       mic ──► voice (who? person/TV/outside?) ──► intent decider ───┤                  └──► phone ask / reply
                        └──► keyword baseline (shadow, never acts) ──┘
                 scene + intent decider = the Strands house agent: Decider 2B (router) → Claude on Bedrock if unsure
```

Everything runs on this machine. The decider servers and the web UI bind to `127.0.0.1` only.

## Layout and ownership

| Path | Owner (workstream) | Entry point |
|---|---|---|
| `homedemo/contracts.py`, `bus.py`, `decider_client.py`, `home.json` | integrator | shared contracts, do not edit in workstreams |
| `homedemo/sim/` | **sim** | `World(home, bus, settings)` implements `WorldAPI`; owns SensorHub; applies `action.command` |
| `homedemo/scenarios/` | **scenarios** | `load_scenarios()`, `ScenarioRunner(bus, world, settings)`; files in `scenarios/library/*.json` |
| `homedemo/agents/` | **agents** | `start_agents(bus, home, settings, deciders) -> list[Agent]` |
| `homedemo/decider/` + `eval/` | **questions** | `build_speech_query`, `build_scene_query`, `DeciderQuery.interpret` |
| `homedemo/patterns/` | **bedrock** | `DeciderToolRouter` (reusable Strands intervention, README.md there), `example_assistant.py` |
| `homedemo/agents/house_agent.py`, `agents/cascade.py`, `eval/run_bedrock.py`, `eval/BEDROCK.md` | **bedrock** | the Strands house agent on Bedrock; the 2B -> Claude cascade and its measurements |
| `homedemo/web/` | **web** | `create_app(runtime: RuntimeAPI) -> FastAPI`, static UI in `web/static/` |
| `homedemo/runtime.py`, `homedemo/__main__.py` | integrator | wires everything; `python -m homedemo` |
| `tests/<area>/` | same owner as the area | `pytest tests/<area>` |

Rules for workstreams: edit only your paths; code against `contracts.py`; if a contract must
change, report it instead of editing. Commit on your branch in your own worktree.

## Run

```bash
scripts/dev-venv.sh                     # once: venv with fastapi, uvicorn, httpx, pytest
.venv/bin/python -m pytest -q           # all tests (no model needed: FakeDecider)
# decider servers (2B :8765, 12B :8766, loopback only) are started from the scratch model venv:
#   $KIROCREW_SCRATCH/smarthome/serve.sh
PYTHONPATH=. .venv/bin/python -m homedemo --port 8780 --mode cascade   # UI at http://127.0.0.1:8780
PYTHONPATH=. .venv/bin/python -m homedemo.play --mode cascade --speed 4 hero_three_minute_tour   # headless check
# cascade = Decider 2B first, then --escalate bedrock (default: Claude Haiku 4.5, AWS profile "bedrock",
# us-west-2) or --escalate 12b (the local 12B server). --decider-2b-url / --decider-12b-url pick the servers.
HOMEDEMO_LIVE=1 .venv/bin/python -m pytest -q tests/agents/test_live_bedrock.py   # one real 2B + Bedrock call
```

The page opens on the 3D house (three.js 0.186.1, vendored; Kenney and KayKit CC0 models; no network
requests). The [3D | 2D] switch shows the flat plan, which is also the fallback without WebGL2. Pick
"The three-minute tour" and press Run; at speed 1.15 it fits a 3-minute slot on this CPU.

### The UI, for the video

- **Decider X-ray** ("How the house decided", under the house): for the newest decision, what was heard or
  seen (speaker, voice source, room), the exact typed request sent to the Decider 2B (`request.state` and
  `request.questions` as pretty JSON, question-type badges choice / yes-no / score, the option list with its
  probabilities as bars and the act threshold), the confidence against the threshold, a stopwatch per model
  call that runs while the decision is in flight and stops at `latency_ms`, and where the decision was made:
  LOCAL (2B on the hub, 0 tokens, nothing left the house), BEDROCK (Claude Haiku 4.5 on Bedrock, tokens,
  Claude's Strands tool calls and spoken reply) or 12B. The keyword baseline is judged beside it; earlier
  decisions sit in a short history (click one to show it). Every label comes from the proposal's `route`,
  `model` and `by`; nothing is hard-coded to one model.
- **Scoreboard**: computed in the browser from the proposals since page load (or its reset button): decisions
  judged, handled on the hub by the 2B (count, median latency, 0 tokens), sent to Bedrock (count, median
  2B + Claude latency, total tokens), share of LLM calls avoided, keyword-baseline mistakes avoided.
- **Presenter mode** for recording (`?present=1`, the Presenter button, or key `p`): the intent banner
  ("LocoCisco", constant `PROJECT_NAME` in `static/present.js`), the 3D house on the left (~62%) with the scoreboard under
  it, the X-ray on the right (~38%), debug panes hidden, captions from the scenario narration over the house,
  decision cards that dwell longer, Claude's reply as a speech bubble over the hub. Keys: `x` hides/shows the
  X-ray, `c` shows the "reuse it in your agent" code card (the 5-line `DeciderToolRouter` wiring and the typed
  tool-choice question), `Esc` closes it. Record at 1440x810 CSS px (device scale 4/3 = 1920x1080).
- Develop the UI on fake data: `PYTHONPATH=. .venv/bin/python -m homedemo.web.dev_fake --port 18750`
  (loopback only; the fake proposals carry the same X-ray fields as the real agents).

On this CPU-only box a v1 decider answer takes about 2.4 s on the 2B (dedicated server) and 9 s on the
12B; on a GPU the same models answer in about 0.1-0.2 s (Strands launch post). Claude Haiku 4.5 on
Bedrock answers an escalation in about 1.5 s (two round trips: tool call, then the spoken sentence).
Runtime mode, escalation target, thresholds and speed are live in Settings. Measured numbers for the
2B -> Claude cascade: `eval/BEDROCK.md`.

To view the UI from a laptop: `ssh -L 8780:127.0.0.1:8780 <cloud-desktop>`, then open
`http://127.0.0.1:8780`. Do not bind to `0.0.0.0`.

## Decider facts this demo is built on (measured 2026-10-20, CPU, 16 threads)

| Test | Keyword rule | 2B | 12B |
|---|---|---|---|
| Speech, no wake word (14) | 5 | 12 | 13 |
| Camera scenes (10) | 6 | 7 | 8 |
| Who asked + rules (9) | code | 6 | 7 |
| Median latency | 0 | 0.8 s | 5.0 s |

Both models failed: sunny kitchen -> lights on; parent unlock refused; a TV voice obeyed. So
rules handle reflexes, the voice agent handles TV/outside voices, and policy code handles
permissions.
