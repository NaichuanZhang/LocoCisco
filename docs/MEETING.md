# LocoCisco meeting brain (`unspoken/meeting`)

The room brain: small agents on one bus turn sensor events into named captions, tasks, decisions, room
changes and Discord posts, with no wake word and every model on the box. Owned by the meeting workstream;
the room screen page (`unspoken/meeting/web`) belongs to the screen workstream.

## How it fits together

```
audio.segment + video.speakers ──► fusion ──► meeting.utterance (name, seat, source, context line)
                                               │
                     reflex rules (plain code) ─┤ "off the record" / "这个不要记" -> meeting.privacy + wipe
                                               │ "Hi, I'm Priya" -> name <-> voice <-> seat (session-only)
                                               │ camera says slides -> set_light slides (route "rule")
                                               ▼
                   decider agent: ONE typed request per utterance to the Strands Decider (questions.py)
                       sure (>= act_threshold) ───────────────► meeting.proposal, route local-decider
                       unsure (ask..act) and llm enabled ───► Strands RoomAgent (local LLM, meeting tools)
                                                              behind DeciderToolRouter -> route local-llm
                       below ask_threshold ───────────────────► published anyway, the policy ignores it
                                               ▼
                   policy (plain code): risk table, media denied, guests block docs, off-record -> only
                   pause/resume, high risk -> ask the owner        -> meeting.verdict (proposal_id = Event.id)
                                               ▼
                   actuator: lamp / fan (unspoken.devices), items (meeting.item), Notifier (unspoken.bridge,
                   egress-metered -> egress.meter), screen cards (meeting.command target "screen"), approvals
                                               ▼
                   screen agent -> screen.state (captions, agenda + timer, banner, cards)
```

Agents and files: `fusion.py`, `reflex.py`, `cascade.py` (decider + LLM escalation), `llm_agent.py` (the
Strands agent and its tools), `questions.py` (what the decider is asked), `dates.py` (due dates in code),
`policy.py`, `actuator.py`, `screen.py`, `docs.py` (local doc search), `recap.py`, `sim.py` (simulated room),
`scenarios/` (DSL runner, matchers, validator, `library/*.json`), `runtime.py` (`MeetingRuntime`,
`MeetingRuntimeAPI`), `__main__.py` and `play.py` (CLIs), `fakes.py` (fake devices / notifier / embedder,
the keyword decider and the fallback registry), `config.py` (`config/meeting.toml`).

## Run it

```bash
cd <worktree> && UV_CACHE_DIR=... scripts/dev-venv.sh
uv pip install --python .venv/bin/python -r requirements/meeting.txt -r requirements/screen.txt   # websockets for /ws

# sim mode with the room screen on http://127.0.0.1:8781/ (fake profile: keyword decider, no servers needed)
.venv/bin/python -m unspoken.meeting --profile fake --mode sim --port 8781
# same with the real CPU deciders on this box, LLM escalation off
.venv/bin/python -m unspoken.meeting --profile dev --mode sim --port 8781 --no-escalate
# live mode on the GB10 (mic + camera through unspoken.sensors, real lamp/fan, Discord through the bridge)
UNSPOKEN_PROFILE=gb10 UNSPOKEN_MCP_TOKEN=... .venv/bin/python -m unspoken.meeting --mode live
# replay a recording through the real sensors, or a bus trace (.jsonl written with --trace)
.venv/bin/python -m unspoken.meeting --mode replay --replay tests/sensors/fixtures/meeting3.wav --profile dev

# scenarios: pass/fail per expect, every decider decision with route, confidence, latency, tokens
.venv/bin/python -m unspoken.meeting.play                     # all, profile fake, fake devices + console notifier
.venv/bin/python -m unspoken.meeting.play demo_three_minutes --profile dev --no-escalate --speed 1
```

`--scenario <id>` runs a scenario at start; the room screen's buttons (or `POST /api/scenario/<id>`) do the
same. `--trace file.jsonl` writes every bus event (DO NOT use it in a real meeting: off-record words would be
on disk; see Privacy). `--fake-io` forces fake devices and the console notifier even when the real packages
import. The MCP server for the OpenClaw agent starts in-process when `UNSPOKEN_MCP_TOKEN` is set (host/port
from `config/bridge.toml [mcp]`).

Profiles: `fake` (keyword decider, everything else off), `dev` (Strands Decider 2B on :8765 and 12B on :8766
on this box's CPU, LLM off), `dev-local` (dev + an OpenAI-compatible llama.cpp server on 127.0.0.1:8010 as
the llm role; needs the `openai` package), `gb10` (whatever `config/models.toml` says; `unspoken.models`
owns it). When `unspoken.models` is not importable the fallback `fakes.LocalRegistry` serves these profiles.

## Config (`config/meeting.toml`)

`[meeting]` title, owner (gets approval DMs), docs_dir, `act_threshold` / `ask_threshold` (the cascade
bands), `escalate` (LLM on/off), `wipe_seconds` (how much transcript "off the record" deletes),
`caption_lines`, `owner_threshold` (owner-choice probability needed to pin a task on someone),
`approval_ttl_seconds`, `catch_up_after_seconds`, `absent_after_seconds`, `banner_minutes`.
`[devices]` lamp / fan ids (from `config/room.toml`). `[channels]` tasks / notes keys (from
`config/bridge.toml`). `[scenes]` lamp scene per room mode and privacy state. `[screen]` host/port of the
room screen (loopback only). `[questions] version` = which wording of the decider questions (`v2` .. `v4`,
see eval/MEETING_RESULTS.md).

## What the decider is asked (questions.py)

One request per utterance, all questions typed and answered together:

| key | type | read when | options |
|---|---|---|---|
| `act` (v3+) | noul | always | "did they give a to-do / decide / ask the room / want the room changed?" folded in as a product of experts |
| `kind` | choice | always | Task, Decision, Question for the room, Room request, Sensitive, Nothing to do (v3+; v2 has 8 options incl. small talk / off topic) |
| `owner` | choice | kind = task | the checked-in names + "No one named"; the speaker's option says "takes it on: I'll do it" |
| `room` | choice | kind = room | brighter, dimmer, warmer, cooler, slides, none (v3+: asked only when the words mention the room) |
| `sensitive` | noul | always | pay / HR / health / legal / layoffs hint, noisy-OR'd into kind = sensitive |

`interpret()` maps kinds to `contracts.ACTIONS` and writes the arguments in code: owner from the owner
choice (plus "I'll ..." = the speaker, "can you" with one other person here = that person), due date from
`dates.parse_due` (Friday, tomorrow, EOD, next week, 周三前, dates), lamp scene / fan state from the room
choice, `share_docs` when a question asks to put a document in the channel. Confidence is the probability
mass behind the chosen action; for room actions it is `P(kind=room) * P(room choice)`.

## Policy (plain code, policy.py)

Risk per action: low = answer_question, park_topic, set_light, set_fan, show, pause_notes; medium =
create_task, log_decision, resume_notes, share_docs; high = post_recap. Rules in order: none -> ignore;
words from the room screen -> deny; pause_notes always allowed; a guest in the room -> share_docs denied;
off the record -> only pause/resume; paused -> no items; resume after a sensitive pause -> ask the owner;
post_recap -> ask the owner; reflex rules act; then the confidence bands: >= act -> act (high risk asks the
owner); ask..act -> low risk acts, medium/high asks the owner; below ask -> ignore. User proposals from the
MCP server (`by: user, confidence 1.0`) go through the same table and get a verdict for their proposal id.

## Scenario DSL (sim mode)

`unspoken/meeting/scenarios/library/*.json`: `{"id","title","blurb","tags","steps"}`. World steps (sim.py
turns them into the same `audio.segment` / `video.speakers` events live sensors publish): `start`, `join`
(seat, voice, checkin or null = guest), `say`, `media_say`, `enter`, `leave`, `scene`, `chat` / `react`,
`clock`, `end`. Runner steps: `wait`, `note`, `expect`. Expect kinds: `item` (kind / owner / text_contains /
has_due / status), `privacy` (state, min_wiped), `device` (device_id, state subset), `post` (channel,
text_contains, kind), `no_post`, `no_item`, `proposal` (by / action / route / mode / min_confidence),
`verdict` (verdict / action / rule_contains), `screen` (banner_contains / card / caption_contains /
no_caption_contains / privacy). `validate()` checks every file; `python -m unspoken.meeting.play` runs them.
The library holds the 3-minute demo (`demo_three_minutes`) and 20 regression scenarios.

## Privacy model

Faces, voices and names live in `MeetingState.people` only; `end` (and `reset`, and live mode's
`SensorHandle.stop()`) wipe them. "Off the record" is a reflex: the last `wipe_seconds` of transcript are
deleted from memory and from the screen, the lamp turns red, nothing is sent to a model until "back on the
record", and the policy denies every item. The only bytes that leave the box are the Notifier's text, counted
by the egress meter (`egress.meter`); audio and video bytes out are 0 by construction. Known gap: the bus
history (in memory, 5000 events) and the optional `--trace` file still hold the raw utterance text of
off-record lines; keep tracing off in real meetings (see the report's risks).
