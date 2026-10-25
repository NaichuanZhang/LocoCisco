# Pattern: Decider tool router (`#tool-selection`)

A Strands agent whose **tool is picked by the Strands Decider 2B before the LLM is ever called**. The
2B runs on the device (here: the home hub, CPU); when it is sure, the LLM is skipped: 0 tokens, nothing
leaves the house. When it is unsure, the normal model (Claude Haiku 4.5 on Amazon Bedrock) decides with
the same tools. The LLM becomes the fallback, not the first call.

```
user text ──► DeciderToolRouter.before_invocation ──► Decider 2B (one typed request)
                     │                                       │
                     │  P(tool) ≥ threshold ──► run tool ──► Deny("picked lights_off(room=kitchen) at 91%")
                     │                                       └─► Strands ends the turn, model NOT called
                     └─ unsure / "No tool" doubtful / error ──► Proceed() ──► Claude on Bedrock decides
```

Files: `decider_tool_router.py` (the handler, ~200 lines, no other homedemo imports),
`example_assistant.py` (a 3-tool standalone agent), `fakes.py` (a scripted Strands model and decider
for tests). The smart home wires it in `homedemo/agents/house_agent.py` + `cascade.py`.

## The typed question (default query builder)

For any Strands agent the router builds ONE request from the agent's own tool specs. State = the
user's message. Questions, in the strands-decider wire format:

```json
{
  "tool": {
    "type": "choice",
    "instructions": "A user said this to an assistant that has the tools listed as options. Which ONE tool should the assistant call to handle it? Pick 'No tool' if the message needs none of them or is not a request.",
    "criteria": {
      "set_lights": "Switch or dim the lights in a room.",
      "set_thermostat": "Nudge the thermostat up or down by 1.5 C.",
      "play_music": "Play music or the news on the speaker in a room.",
      "No tool": "The message needs none of these tools, or is not a request."
    }
  },
  "set_lights.room":      {"type": "choice", "instructions": "If the assistant calls set_lights, what should its 'room' argument be?",
                           "criteria": {"living": "room = living", "kitchen": "room = kitchen", "bedroom": "room = bedroom"}},
  "set_lights.state":     {"type": "choice", "instructions": "If the assistant calls set_lights, what should its 'state' argument be?",
                           "criteria": {"on": "state = on", "off": "state = off", "dim": "state = dim"}},
  "set_thermostat.direction": {"type": "choice", "...": "one per enum-typed argument"},
  "play_music.room": {"...": "..."}, "play_music.genre": {"...": "..."}
}
```

The option set of `tool` is exactly the agent's tool names (descriptions = the first line of each tool's
docstring) plus `No tool`. Every `Literal[...]`/enum argument gets its own `choice`. The decider answers
all of them in one call; the router reads:

- **tool** = the chosen option; `No tool` means "needs nothing".
- **args** = the chosen enum values for that tool.
- **confidence** = P(tool) x the product of the chosen argument probabilities (the joint probability of the
  whole call); **probabilities** = the per-option distribution of the `tool` question (for UI bars).

## Wiring (5 lines)

```python
from strands import Agent
from strands.models import BedrockModel
from homedemo.patterns import DeciderToolRouter

router = DeciderToolRouter(decider, threshold=0.8)        # decider: any object with `async ask(state, questions)`
agent = Agent(model=BedrockModel(model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
              tools=[set_lights, set_thermostat, play_music], interventions=[router])
agent("Kitchen lights off.")                               # sure: tool runs, Claude never called
print(router.last)                                         # RouteDecision(route="local", tool="set_lights", args={...}, confidence=0.84, ...)
```

`decider` for the real thing is the strands-decider server (`strands-decider serve <2B> --port 8765`);
`example_assistant.LocalDecider` is the 20-line httpx client. Run the example:

```bash
PYTHONPATH=. .venv/bin/python -m homedemo.patterns.example_assistant --decider-url http://127.0.0.1:8765 --profile bedrock
```

Measured 2026-10-20 with the default query on the example's three tools (CPU 2B, 7 questions per request,
3.7-3.9 s each): "Put on some jazz in the living room." -> `play_music(room=living, genre=jazz)` at 95%,
Claude never called; "My sister said her heater broke last winter." -> no tool at 95%, model skipped;
"Kitchen lights off." -> the 2B's joint pick was `set_lights(kitchen, on)` at 62% -> Claude, who turned them
off (1,884 tokens); "It's freezing in here." -> `set_thermostat(warmer)` at 44% -> Claude; "I can't see a
thing in here." -> no tool at 91%, model skipped (wrong: an indirect request). The generic question set is a
starting point: the smart home gets its numbers below from tuned questions (eval/REPORT.md) and an `accept`
rule that never trusts a 2B "none" for a person's words.

## How to use it in your own agent

1. Give every tool the router may auto-call **enum arguments only** (`Literal[...]`). The 2B can pick, not
   write: a tool with a required free-text argument is still offered as an option, but when it is picked
   the router returns `Proceed()` and the LLM writes the arguments.
2. Pick the threshold from your own measurements (`eval/REPORT.md` section 3 shows how: act precision at
   each threshold). Start at 0.8.
3. Decide what happens on "No tool". The default accepts a confident `No tool` and ends the turn without
   the model. The smart home overrides `accept` so that a `none` for a person's words or a camera scene
   always escalates (the 2B's `none` is weak evidence, REPORT.md s3) while TV/outside voices do not.
4. Optional knobs: `execute=False` (only report the pick, run nothing: the home does this because its
   policy agent decides), `on_decision=callback` (sync or async; the home publishes a proposal from it
   before Claude is even asked), `build_query` / `interpret` / `accept` to plug a tuned question set.
5. Read `router.last` (or `on_decision`) for the X-ray: route, tool, args, confidence, probabilities,
   latency_ms, the exact request `{state, questions}`, raw answers, error.

`on_error` is `"proceed"`: a broken decider, server or tool never blocks the agent; the LLM takes over and
the error is recorded on the decision. Keep `DeciderToolRouter` first in `interventions` so an authorization
handler that follows it still sees the LLM path (a Deny from the router short-circuits the rest).

## Limits

- One request per user turn, on the newest user message only (no conversation history goes to the 2B).
- Picks are single-tool. For "two tools" turns (TV off AND lights off) the 2B picks the main one or is unsure;
  the home lets Claude call up to two tools on the escalation path.
- Argument values must be enums; a required free-text, number or object argument makes the tool LLM-only.
- Tools the router executes itself are called as plain Python (`tool(**args)`), so they must not require
  `tool_context`. With `execute=False` this does not matter.
- The 2B server used here is CPU-only and not concurrency-safe: one request in flight at a time
  (the home's HttpDecider has a semaphore; the demo never runs two processes against it).

## Measured (smart home, 2026-10-20)

Numbers from `eval/BEDROCK.md` (all 93 speech + 42 scene cases of `eval/cases_*.json`, dedicated 2B server
on CPU, Claude Haiku 4.5 in us-west-2, threshold 0.80). See that file for the full tables and the live
scenario runs.

| kind | cases | handled by the 2B alone (model skipped) | right when local | right when escalated to Claude | cascade right | Claude on every case right | tokens: cascade vs Claude-on-everything |
|---|---|---|---|---|---|---|---|
| speech | 93 | 34 (37%) | 34/34 | 54/59 | **88/93 (95%)** | 87/93 (94%) | 212,432 vs 335,345 (-37%) |
| scene | 42 | 3 (7%) | 3/3 | 38/39 | **41/42 (98%)** | 41/42 (98%) | 140,800 vs 151,682 (-7%) |

Latency on this CPU box: 2B call 2.4-2.6 s median (0.1-0.2 s on a GPU per the Strands launch post);
Claude's share of an escalation 1.4 s median, about 3,600 tokens per escalation (two round trips, 12 tool
schemas). The three-minute tour: 4 model decisions, 2 handled by the 2B with Claude never called, 2 escalated.
Thresholds: 0.80 act / 0.50 ask, from eval/REPORT.md s3 (acting on a 2B answer at or above 0.80 was right
100% of the time there and here: 37/37).
