# Decider 2B on the hub, Claude on Bedrock when unsure: measured

All numbers below were measured on 2026-10-20 (UTC 18:34-19:15) on this CPU-only box with the dedicated
2B server (`http://127.0.0.1:8767`, Qwen3.5-2B checkpoint `strands-decider-2B-qwen3.5-v1-2610`, 16 threads)
and Claude Haiku 4.5 on Amazon Bedrock (`us.anthropic.claude-haiku-4-5-20251001-v1:0`, us-west-2,
temperature 0, max 200 output tokens). The path is exactly the demo's: every case goes through the Strands
house agent (`homedemo/agents/house_agent.py`) with the `DeciderToolRouter` in its path and the cascade's
accept rule (act threshold 0.80; a 2B `none` for a person's words or a camera scene always escalates,
eval/REPORT.md s3). Harness: `eval/run_bedrock.py` (run file `eval/runs/bedrock-all.json`, gitignored);
scenario runs: `python -m homedemo.play --mode cascade --escalate bedrock --speed 4 --decider-2b-url
http://127.0.0.1:8767 ...`. One request was in flight at a time.

## 1. The three-minute tour and the scenario library (live, speed 4)

| set | scenarios | passed | decisions | local (2B sure) | escalated to Claude | Bedrock calls / tokens |
|---|---|---|---|---|---|---|
| `hero_three_minute_tour` (run 3 times: 12/12 each) | 1 | **12/12 checks** | 6 proposals (4 decisions) | 2 | 2 | 2 / 7,326 |
| research (`--tag research`) | 14 | **13/14** (42/43 checks) | 25 proposals (14 decisions) | 3 | 11 | 11 / 39,694 |
| hero (`--tag hero`, incl. the tour) | 12 | **11/12** (37/38 checks) | 23 proposals (15 decisions) | 7 | 8 | 8 / 29,155 |
| regression (`--tag regression`) | 33 | **28/33** (50/55 checks) | 57 proposals (34 decisions) | 11 | 23 | 23 / 83,917 |

With the 12B as the escalation (homedemo/scenarios/README.md, same day): hero 11/12, research 11/14,
regression 32/33. The failures, each a different cause:

| scenario | what happened | cause |
|---|---|---|
| `hero_asleep_tv_on`, `reg_s07_asleep_tv_on` (at-risk) | Claude turned the TV off as soon as Alex lay down with his eyes closed (`tv_off` 0.9, command issued), 40 simulated minutes BEFORE the step the check is anchored to; on the later idle scene it rightly said nothing (the TV was already off) | right action, earlier than the scenario expects; the check window starts at the `idle` step. The 12B failed it the other way (tv off at 39% -> ignored) |
| `res_07_did_you_lock_the_door` (at-risk), `reg_p07_dark_movie`, `reg_p08_wheres_the_remote`, `reg_s04a_remote_hunt` | Claude: `ask_resident(lock_door)` for a question about state; `lights_dim` for "that movie was so dark"; `no_action` for "where's the remote?" and for the couch-cushion search | Claude's judgement on the three hard types eval/REPORT.md s7 names (state questions, mentions, indirect requests). p07 and p08 passed with the 12B only because its 59% / 78% answers fell in the ask band (asking is not a command); a direct Claude tool call is 0.9 and acts |
| `reg_p13_mandarin_too_dark` | the 2B alone said `lights_on` 95% (local, right), but the house lit the porch: the hallway has no microphone, so the sim attributes Alex's words to the porch mic (seed 7; seed 1 gives the kitchen) | sim/voice room attribution, independent of the escalation path |

The tour, decision by decision (route, latency, tokens):

| beat | 2B (route local) | then | Claude on Bedrock (route bedrock) |
|---|---|---|---|
| Alex, dark living room: "I can't see a thing in here." | `lights_on` 0.7955 in 2315 ms, just under 0.80 -> shadow | escalate | `lights_on(room=living)` in 1645 ms, 3,682 tokens -> light on, "Turning the living room lights on." |
| Sam: "I left the lights on at the office again..." (keyword baseline says lights on) | `none` 0.6455 -> shadow | escalate | `no_action` in 1468 ms, 3,644 tokens -> nothing moves |
| The TV says "Turn off the lights." | `lights_off` 0.8241 -> act | - | not called (0 tokens); the policy denies the TV voice |
| Maya on a chair at the lit stove (camera) | `alert_parent` 0.9066 -> act | - | not called (0 tokens); the reflex rule had already alerted |

(final run, 19:14 UTC; the two earlier runs gave the same routes with Claude at 2333 / 3080 ms and 1646 /
1348 ms.) Dark-kitchen entry and the sunny kitchen are reflex rules (no model). Wall time 57-60 s at
speed 4 (3 min of narration at speed 1, plus about 8 s of model latency).

Research scenarios (the hard ones from the research notes). `2B -> Claude` = the 2B's answer and
confidence, then Claude's tool; latency is the 2B call, then Claude's share; tokens per Claude call:

| id | said | result | 2B (ms) | Claude (ms, tokens) |
|---|---|---|---|---|
| res_01 TV news dollhouse | TV voice | PASS: 2B `none` 91% -> act (ignored), Claude not called | 2461 | - |
| res_02 code-switch living dark | "客厅有点暗，turn on the lights." | PASS: 2B `lights_on` 97% local | 3253 | - |
| res_03 Mandarin can't see | "好暗，我看不清。" | PASS: 2B `lights_on` 95% local | 2382 | - |
| res_04 don't turn the hall light on | negation | PASS: 2B `none` 81% (doubtful) -> Claude `no_action` | 3142 | 2255, 3,596 |
| res_05 leaving, don't turn on anything | code-switch + negation | PASS: 2B `none` 89% -> Claude `no_action` | 2368 | 7271, 3,590 |
| res_06 is the porch light on? | state question | PASS: 2B `none` 69% -> Claude `no_action` | 3140 | 11606, 3,582 |
| res_07 did you lock the front door? | state question | **FAIL**: 2B `none` 86% -> Claude `ask_resident(lock_door)` -> the house asks "Lock the front door?"; the scenario expects `ignore` | 3164 | 5293, 3,621 |
| res_08 lights on in Maya's room | other room | PASS: 2B `none` 58% -> Claude `lights_on(room=bedroom_maya)` | 3177 | 3616, 3,649 |
| res_09 kitchen lights off (terse, from the hallway) | other room | PASS: 2B `none` 58% -> Claude `lights_off(room=kitchen)` | 3128 | 1988, 3,622 |
| res_10 "it's a sauna in here" | indirect | PASS: 2B `none` 58% -> Claude `cooler` | 2338 | 1952, 3,594 |
| res_11 child: unlock, my friend is here | permissions | PASS: 2B `none` 59% -> Claude `ask_resident(unlock_door)` -> policy denies (kids) | 2383 | 2878, 3,631 |
| res_12 guest: "is it always this chilly?" | indirect, guest | PASS: 2B `warmer` 77% -> Claude `warmer` -> the parents are asked | 2569 | 1852, 3,609 |
| res_13 grandma's heater | mention of another place | PASS: 2B `none` 70% -> Claude `no_action` (the 12B used to ask "warmer?") | 2421 | 1793, 3,604 |
| res_14 if it gets any colder tonight | conditional | PASS: 2B `warmer` 69% -> Claude `no_action` (the 12B used to turn the heat up) | 2402 | 3224, 3,596 |

With the 12B as the escalation (homedemo/scenarios/README.md, same day) this set passed 11/14; Claude fixes
res_13 and res_14 (a mention and a conditional read as requests by the 12B) and still fails res_07, where it
asks before locking instead of ignoring a question about state. Three Claude calls took 5-12 s instead of
the usual 1.5-3 s (res_05, 06, 07); the eval below saw one such spike (8.5 s max, p90 1.7 s), so these are
Bedrock-side latency outliers, not the pipeline.

## 2. Over eval/cases_speech.json and eval/cases_scene.json (all 93 + 42 cases)

Split `all` = dev + test (the dev/test split of eval/REPORT.md is reported underneath). "Right" = the
cascade's final action is in the case's accepted actions; for an escalated case that is Claude's first tool
call (`ask_resident(X)` counts as X, `no_action` or no call = `none`).

| kind | cases | handled locally (2B sure) | right when local | acting locally right | right when escalated to Claude | **cascade right** | 2B alone right | Claude on every case right | median latency local | median latency escalated (2B + Claude) | Claude's share (median / p90 / max) | tokens per Claude call (median) | tokens total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| speech | 93 | 34/93 (37%) | 34/34 (100%) | 34/34 (100%) | 54/59 (92%) | **88/93 (95%)** | 77/93 (83%) | 87/93 (94%) | 2358 ms (p90 3085) | 3847 ms (p90 4754) | 1433 / 1699 / 8502 ms | 3,591 (in 3,548 / out 46) | 212,432 |
| scene | 42 | 3/42 (7%) | 3/3 (100%) | 3/3 (100%) | 38/39 (97%) | **41/42 (98%)** | 29/42 (69%) | 41/42 (98%) | 2607 ms | 4010 ms (p90 4251) | 1416 / 1649 / 1861 ms | 3,594 (in 3,555 / out 39) | 140,800 |

By split (cascade right / baseline right): speech dev 44/47 vs 43/47, speech test 44/46 vs 44/46; scene
dev 23/24 vs 23/24, scene test 18/18 vs 18/18. Room: on the 32 speech cases whose expected action is a
light action the target room was right 32/32 (16 of them name another room than the speaker's; 3 of those
were decided locally by the 2B, 13 by Claude).

Why cases escalated: speech 37 because the 2B was below 0.80 and 22 because its answer was a `none` at or
above 0.80 (the doubtful-none rule); scene 16 and 23. On the escalated cases the 2B's shadow answer would
have been right 43/59 (speech) and 26/39 (scene); Claude agreed with it on 41/59 and 25/39. Claude used
`ask_resident` 5 times (sp32, sp78, sp89, sc04, sc05) and two tools twice (sc19 asleep with TV and lights on:
`tv_off` + `lights_off`; sc24 stranger in the hallway: `alert_parent` + `lock_door`).

### The "LLM on every sentence" baseline (router off, same agent, same tools)

| kind | cases | right | median latency | p90 latency | tokens per call (median) | tokens total | Bedrock calls |
|---|---|---|---|---|---|---|---|
| speech | 93 | 87/93 (94%) | 1469 ms | 1665 ms | 3,594 | 335,345 | 93 |
| scene | 42 | 41/42 (98%) | 1430 ms | 1771 ms | 3,597 | 151,682 | 42 |

Cascade against the baseline: speech **88 vs 87 right**, 59 instead of 93 Bedrock calls (37% fewer tokens:
212,432 vs 335,345); scene 41 vs 41 right, 39 instead of 42 calls (7% fewer tokens). The one speech case the
baseline gets wrong and the cascade right is sp11 "Lights, please." (Claude: lights off; the 2B: lights on
at 92%, handled locally). The cascade is wrong on nothing the baseline gets right: all 6 cascade misses are
escalated cases where Claude itself is wrong.

Latency on this CPU box: a locally handled case costs the 2B call (2.4-2.6 s median) and nothing else; an
escalated case costs the 2B call plus Claude (3.8-4.0 s median), so the cascade is slower than Claude alone
(1.4-1.5 s) wherever the 2B runs on CPU. On a GPU the 2B answers in 0.1-0.2 s (Strands launch post), which
turns the local path into the fast one. What the cascade buys today is the 37% of speech (7% of scenes) that
never leaves the house and costs no tokens, at equal or better accuracy.

### Wrong answers (cascade)

| case | route | said / seen | expected | got |
|---|---|---|---|---|
| sp25 | bedrock | "Where's the remote? The game starts in five minutes." | tv_on | 2B `none` 94% (doubtful) -> Claude `no_action` |
| sp73 | bedrock | "That movie was so dark I could barely see anything." | none | 2B `none` 67% -> Claude `lights_dim` ("Dimming the living room lights.") |
| sp78 | bedrock | "Did you lock the front door?" | none | 2B `none` 86% (doubtful) -> Claude `ask_resident(lock_door)` |
| sp88 | bedrock | "The neighbors left their porch light on again, it's shining right into our window." | none | 2B `none` 80% (doubtful) -> Claude `lights_off` (bedroom) |
| sp89 | bedrock | "If it gets any colder tonight we should turn the heat up." | none | 2B `warmer` 75% -> Claude `ask_resident(warmer)` |
| sc05 | bedrock | 07:50 Monday, adult in a coat holding a bag, lifting couch cushions | none | 2B `none` 79% -> Claude `ask_resident(alert_parent)` |

Three of the six (sp78, sp89, sc05) are `ask_resident`, which the policy turns into a phone question rather
than an action; the baseline (Claude alone) makes the same six mistakes. Two (sp25, sp78) are the failure
types eval/REPORT.md s7 names: an indirect request read as chatter, and a question about device state read
as a request.

## 3. What the demo video can say

- 4 decisions in the tour: 2 handled on the hub by the 2B (TV voice, stove hazard) with Claude never called,
  2 escalated to Claude (the indirect "I can't see", the office-lights trap) at about 3,600 tokens each.
- Over 135 cases: 37 handled locally, all 37 right; Claude decides the other 98 and is right on 92 of them;
  the cascade matches or beats Claude-on-everything (88 vs 87 speech, 41 vs 41 scene) with 98 instead of 135
  Bedrock calls and 353k instead of 487k tokens.
- A 2B call takes 2.4 s on this CPU; Claude's share of an escalation is 1.4 s median.

## 4. Known gaps

- Scene decisions almost always escalate (39/42): the 2B barely reaches 0.80 on a scene action and its `none`
  is treated as doubtful. The 2B's local share is a speech number.
- Per-Claude-call tokens are dominated by the 12 tool schemas (about 3,550 input tokens per call, two calls
  per escalation: the tool call and the spoken sentence). Prompt caching or fewer tools would cut this.
- Small sets: one case is 1-2 percentage points. Latencies were taken while the 12B server idled on the same
  box; the 2B server was dedicated to this run.
