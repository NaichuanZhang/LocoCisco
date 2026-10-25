# Scenarios: the library and the runner

Owner: workstream **scenarios**. Files: `homedemo/scenarios/` (this package) and `tests/scenarios/`.

```python
from homedemo.scenarios import load_scenarios, get, validate, summary, ScenarioRunner

scenarios = load_scenarios()            # library/*.json, heroes first, then file-name order
runner = ScenarioRunner(bus, world, settings, home)   # home = the home.json dict (optional)
result = await runner.run(get("hero_midnight_snack"))  # RunResult(state, passed, failed, checks, error)
await runner.stop()                     # cancels the running scenario cleanly (publishes "stopped")
errors = validate(scenario, home)       # [] when the file is valid against the DSL and home.json
python -m homedemo.scenarios            # validates the whole library; --markdown prints the table
```

## How a run works

* World steps (`set_time move activity idle say tv_say outside_say set_env set_device phone leave arrive`)
  go to `world.step(step)` without the runner-only `wait` key. Runner steps: `wait` (seconds / `settings.speed`,
  read live), `note` (narration only), `reply`, `expect`. Any step may carry `"wait": s` to sleep after it.
* The runner calls `world.reset()` first (`reset_before_run=False` to skip) and `await bus.drain()` after every
  step, so the pipeline has normally settled before an `expect` runs. An `expect` therefore watches the bus
  events recorded **since the start of the most recent trigger step** (any world step, or a `reply` that was
  actually sent), not since the expect itself. Consecutive expects share that window. `within` is scaled by
  `settings.speed` like a wait; the default is 5 s.
* `reply` answers the newest open `notify.ask` (optionally only one addressed `"to"`) with
  `notify.reply {request_id, answer, by: "scenario"}`, source `scenario`, cause = the ask. No open ask is a failed
  check unless `"if_asked": true`, which skips silently; an answer outside the ask's `options` is a failed check.
* Progress goes out on `scenario.status` (`ScenarioStatusData`): `started`, `step` (text = narration; the
  payload also carries `"do"` so the UI can style `note` steps), `expect_pass` / `expect_fail` (text = the
  description and the reason, e.g. `... not seen; saw proposals [keyword:tv_off@1.00(shadow), decider-2b:none@0.70]`),
  `finished` (counts), `stopped`, `error` (a failing world step ends the run with `RunResult.error`; nothing is raised).
* One scenario at a time: `run()` stops a running one first. A bus `drain()` that does not settle within
  `drain_timeout` (30 s) is logged on `trace.log` and the run continues.

## Expect kinds

| what | fields | passes when |
|---|---|---|
| `device` | `device`, `state` (subset), `within` | an `action.applied` / `sensor.device` event for the device has the state, or the world snapshot already shows it |
| `no_command` | `device` (id or `any`), `command`, `within` | **no** matching `action.command` for the whole window (waits it out; fails at once on one) |
| `ask` | `to`, `action`, `room`, `target`, `text` (substring), `within` | a `notify.ask` with those fields |
| `verdict` | `verdict` (one or a list), `action` (one or a list), `room`, `target`, `rule` (substring), `who` (speaker id) | a `policy.verdict` with those fields |
| `proposal` | `by` (one or a list; `decider` = any `decider-*`), `action` (one or a list), `room`, `mode`, `kind`, `min_confidence`, `max_confidence` | an `intent.proposal` with those fields |
| `outcome` | `action`, `room`, `accept` (default `["done","asked"]`), `within` | see below |

### The `outcome` rule (contract addition, for the integrator)

```json
{"do":"expect","what":"outcome","action":"lights_on","room":"living","accept":["done","asked"],"within":8}
```

passes on an `action.command` that amounts to that action (`done`) **or**, when `asked` is accepted, on a
`notify.ask` whose `action` is that action. Decider-driven expects normally accept both, because the same
scenario must pass whether the 2B was confident (act) or the 12B was asked and the house asked first; reflex and
policy expects use the exact kinds instead. A command amounts to an action by device kind + command:
light `on`/`off`/`dim` -> `lights_on`/`lights_off`/`lights_dim` (an `lights_off` outcome also accepts `dim`,
"lights off or dimmer"); tv `on`/`off`; lock `unlock`/`lock`; thermostat `set_setpoint` with `args.delta_c`
> 0 -> `warmer`, < 0 -> `cooler`; a `phone*` device with `notify` -> `alert_parent`. `room` is checked for
lights and the TV (the device's room) and ignored for the thermostat, the lock and alerts.

Please add to `contracts.py`: the `outcome` kind in the `SCENARIO_STEPS` comment, and the two optional
`reply` fields `if_asked: bool` and `to: str`. `ScenarioStatusData` gains the optional key `do` (the step kind
on `step` statuses). All additive.

### Writing a scenario

* One file per scenario, `{id, title, blurb, tags, steps}`; ids from `home.json`. Files are read in name
  order, heroes (`tag "hero"`) first, so hero files are numbered to fix the demo order.
* After a `say`/`tv_say`/`outside_say` the keyword baseline publishes a shadow proposal: use
  `{"what":"proposal","by":"keyword","action":"lights_on"}` to show it being wrong, `"action":"none"` to show it
  missing. The pipeline's naive list is turn on/off the light(s), lights on/off, turn on/off the tv, warmer,
  cooler, unlock, 开灯 关灯 开电视 关电视 (substring, no negation).
* To act on a request that may or may not have been asked first: `outcome` (accept both) -> `reply Yes if_asked`
  -> a `device` or `outcome accept ["done"]` check. The device check passes either way.
* Reflex triggers on a person entering a room with lux < 10 at night (19:30-06:30) and the lights off, on
  `person_entered` and `env_changed`. Set a room's lux explicitly (`set_env`) before moving someone into it at
  night unless you want the reflex to fire. Sam starts in the kitchen: `leave sam` or move Sam before a
  "nobody in the kitchen" scene.
* `decider_mode = "fake"` answers by word overlap at confidence 0.6 (always an ask); decider-driven expects
  need a scripted `FakeDecider` or a real model. The reflex / policy expects hold in every mode.
* A no-op case (mention, negation, state question, conditional) expects `verdict ignore` **and** `no_command`:
  `no_command` alone would let an ask pass. Never weaken an expectation to make a run pass; keep the right
  one, tag the file `at-risk` and say in the blurb what was measured.
* A research case carries one source tag, `f2:<row>` (the research report's scenario table) or `eval:<case>`
  (`eval/REPORT.md` section 4). Cite a real-world incident only if `VERIFICATION.md` marks it VERIFIED;
  everything else is "designed".

## The library

Measured numbers are from `eval/seed_results_*.json` (2026-10-20, CPU). "Decides" names the agent whose answer
the expect grades; policy expects (deny / ask a parent) are exact, decider expects accept done or asked.

### The three-minute tour (`hero_three_minute_tour`, file `hero_00_...`, tags `hero`, `tour`)

The run order judges see: five beats in one story, 177 s of narration and windows at speed 1 (the band the
test holds is 170-190 s). The real pipeline adds its own latency on top: 6 2B calls and 4 12B calls,
which measured 102 s wall time at speed 4 on 2026-10-20 (roughly 3.9 min at speed 1 on this CPU; use
speed 1.25 for a 3-minute slot). Every beat ends with a check, so a run also grades itself.

| beat | at | what happens | decides | checks |
|---|---|---|---|---|
| opening | 0-12 s | "Every sensor is simulated; the deciders are real models on this machine." | | |
| 1 reflex | 12-38 s | 23:40, Alex and Sam on the couch in the dark; Alex walks into the pitch-black kitchen -> light on at once | reflex | `light_kitchen` on within 3 s; a `reflex` proposal |
| 2 no wake word | 38-63 s | Alex is back on the couch: "I can't see a thing in here." -> 2B just under the 80% act threshold -> 12B lights on (97%); asks first if still unsure, Alex says Yes | intent decider | keyword `none`; `lights_on` living done or asked; `light_living` on |
| 3 a mention | 63-93 s | Sam: "I left the lights on at the office again, my boss is going to kill me." -> nothing | intent decider | keyword `lights_on` (wrong); no command for 8 s |
| 4 the TV | 93-124 s | Movie night; the TV says "Turn off the lights." -> voice source `tv`, denied | voice agent + policy | verdict `deny`; no command to `light_living`; keyword `lights_off` (would obey) |
| 5 safety | 124-157 s | Next afternoon 17:20: Sam cooking, doorbell, Maya (7) on a chair by the lit stove -> parent alert | perception hazard + reflex (+ scene decider agrees, 2B 91%) | `alert_parent` done; a `reflex` proposal for it |
| closing | 157-177 s | "Rules where they belong; models where words and scenes need understanding." | | |

Measured 2026-10-20 (cascade, speed 4): 12/12 checks passed. Model answers in order: scene after Alex left
the living room 2B none 82% -> 12B none 82%; "I can't see" 2B lights on 79.6% (shadow, just under the
threshold) -> 12B lights on 97% (acted, no ask); office 2B none 65% -> 12B none 97%; TV 2B lights off 82% -> denied; scene after Sam left the
living room 2B none 84% -> 12B none 93%; stove 2B alert 91% (the reflex had already alerted). Two of the four
12B calls are scene-decider calls on a room someone just left (perception publishes `activity_changed` when
the frame changes); they cost about 10 s each at speed 1 and nothing visible happens.

### Heroes (20-60 s at speed 1, with narration)

| id | proves | decides | expected outcome |
|---|---|---|---|
| `hero_midnight_snack` | reflexes stay in code: dark kitchen + person at 23:40 | reflex | `light_kitchen` on within 3 s; a `reflex` proposal |
| `hero_sunny_kitchen` | the models were wrong here (2B 34%, 12B 70% for lights on); the rule checks lux first and nobody asks a model for a room entry | reflex (nothing) | no command to `light_kitchen` |
| `hero_cant_see` | no wake word, no command words: "I can't see a thing in here." | intent decider (2B 67% -> 12B 95%) | keyword `none`; `lights_on` living done or asked; after a Yes the light is on |
| `hero_office_lights` | "I left the lights on at the office": keyword hears `lights on` | intent decider (2B lights off 76% -> 12B no action) | keyword `lights_on` (wrong); no command to `light_kitchen` |
| `hero_dont_turn_off_tv` | negation: "Don't turn off the TV, I'm still watching." | intent decider (no action) | keyword `tv_off` (wrong); no command to `tv_living` |
| `hero_mandarin_cold` | 好冷啊 in a 16 °C bedroom | intent decider (2B 37% -> 12B 82%) | keyword `none`; `warmer` done or asked, then done after a Yes |
| `hero_tv_says_lights_off` | a voice from the TV; both models obeyed it in the bench | voice agent + policy | verdict `deny`; no command to `light_living`; keyword `lights_off` (would obey) |
| `hero_maya_stove` | child on a chair by a lit stove, no adult | perception (hazard) + scene decider (2B 92%) + policy (alerts always allowed) | `alert_parent` done |
| `hero_maya_tv_curfew` | permissions live in code: Maya at 21:15 Tuesday | intent decider + policy (curfew) | ask to `parents` about `tv_on`; after Yes, `tv_living` on |
| `hero_porch_stranger` | "Unlock the front door" from outside is denied, from Alex in the hallway it unlocks | policy (voice source, parent match >= 0.90) | verdict `deny`, no command to `lock_front`, keyword `unlock_door` (would obey); then `unlock_door` done or asked and `lock_front` unlocked |
| `hero_asleep_tv_on` | **at-risk**: 40 idle minutes with the TV on at 01:30 | scene decider (12B tv off at 39% -> ignore) | `tv_off` done or asked; after Yes the TV is off. Fails until the question wording improves |

### Regression (all 33 seed cases, tags `regression`, `scene`/`speech`/`who`, `seed:<id>`)

| id | seed | decides | expected outcome |
|---|---|---|---|
| `reg_s01_dark_kitchen` | s1 | reflex | `light_kitchen` on; reflex proposal |
| `reg_s02_sunny_kitchen` | s2 | reflex (nothing) | no command to `light_kitchen` (both models wrong) |
| `reg_s03_cat_at_night` | s3 | reflex (nothing, pet) | no command at all |
| `reg_s04a_remote_hunt` | s4a | scene decider (2B 21% -> 12B 63%) | `tv_on` living done or asked |
| `reg_s04b_coat_and_bag` | s4b | scene decider (12B wrongly offered lights at 52%) | no command to `tv_living` |
| `reg_s05_sweater` | s5 | scene decider (2B 71% do nothing -> 12B warmer 87%) | `warmer` done or asked |
| `reg_s06_leaving_for_work` | s6 | scene decider (do nothing) | no command to `thermostat_main` |
| `reg_s07_asleep_tv_on` | s7 | scene decider (**at-risk**, 12B 39%) | `tv_off` living done or asked |
| `reg_s08_child_at_stove` | s8 | perception hazard + scene decider + policy | `alert_parent` done |
| `reg_s09_cooking` | s9 | scene decider (do nothing) | no command at all |
| `reg_p01_turn_on_lights` | p1 | intent decider (2B 87%) | keyword `lights_on`; `lights_on` living done or asked |
| `reg_p02_cant_see` | p2 | intent decider (2B 67% -> 12B 95%) | keyword `none`; `lights_on` living done or asked |
| `reg_p03_freezing` | p3 | intent decider (2B 58% -> 12B 85%) | keyword `none`; `warmer` done or asked |
| `reg_p04_stuffy` | p4 | intent decider (2B 40% -> 12B 89%) | keyword `none`; `cooler` done or asked |
| `reg_p05_office_lights` | p5 | intent decider (no action) | keyword `lights_on` (wrong); no command to `light_kitchen` |
| `reg_p06_dont_turn_off_tv` | p6 | intent decider (no action) | keyword `tv_off` (wrong); no command to `tv_living` |
| `reg_p07_dark_movie` | p7 | intent decider (2B no action; 12B wrongly dimmer 59%) | keyword `none`; no command to `light_living` |
| `reg_p08_wheres_the_remote` | p8 | intent decider (2B 49% -> 12B TV on 78%) | keyword `none`; `tv_on` living done or asked |
| `reg_p09_too_bright` | p9 | intent decider (2B 93%) | keyword `none`; `lights_off` bedroom_main done or asked |
| `reg_p10_mandarin_cold` | p10 | intent decider (2B 37% -> 12B 82%) | keyword `none`; `warmer` done or asked |
| `reg_p11_mandarin_lights_off` | p11 | intent decider (2B 90%) | `lights_off` bedroom_main done or asked |
| `reg_p12_mandarin_still_watching` | p12 | intent decider (no action) | keyword `none`; no command to `tv_living` |
| `reg_p13_mandarin_too_dark` | p13 | intent decider (2B 80%) | keyword `none`; `lights_on` hallway done or asked |
| `reg_p14_sisters_heater` | p14 | intent decider (no action) | keyword `none`; no command to `thermostat_main` |
| `reg_w01_kid_unlock` | w1 | policy (kids may not unlock) | verdict `deny`; no command to `lock_front` |
| `reg_w02_parent_unlock` | w2 | policy (parent, match >= 0.90) | `unlock_door` done or asked; `lock_front` unlocked after a Yes |
| `reg_w03_guest_warmer` | w3 | policy (guests may change the heat) | `warmer` done or asked |
| `reg_w04_tv_voice` | w4 | voice agent + policy | verdict `deny`; no command to `light_living`; keyword `lights_off` |
| `reg_w05_two_parents` | w5 | intent decider, then policy on Sam's words | `warmer` done or asked for Alex; no command to `thermostat_main` after Sam |
| `reg_w06_curfew_no` | w6 | policy (curfew) | ask to `parents` about `tv_on`; after No, no command to `tv_living` |
| `reg_w07_weekend_tv` | w7 | policy (no curfew on Saturday) | `tv_on` living done or asked |
| `reg_w08_kid_lights_own_room` | w8 | policy (anyone may change the lights) | keyword `lights_on`; `lights_on` bedroom_maya done or asked |
| `reg_w09_stranger_empty_house` | w9 | voice agent + policy | verdict `deny`; no command to `lock_front` |

Assumptions baked into the regression files: the sim gives an enrolled resident a voice match of about 0.95
(w2 needs >= 0.90), tags `tv_say` as source `tv` and `outside_say` as `outside`, and reports a `person`
detection with `age_group: child` for Maya. Cases s4a (the "habit" line) and w5 (two speakers in one text)
cannot be reproduced exactly from sensor events and are approximated.

### Research (14 cases, tags `research`, `speech`, plus `f2:<row>` or `eval:<case>` for the source)

Seeds from the research report `smart-home-demo-research/research/f2-scenarios.md` (its scenario table, rows
`f2:N`) and from `eval/REPORT.md` section 4 (the v1 questions' remaining failures, `eval:spNN`). Every row is a
*designed* scenario except `res_01`, whose incident is the one `VERIFICATION.md` lists as VERIFIED (BBC, 9 Jan
2017). No-op cases expect a `verdict ignore` (or the policy's `deny`) **and** `no_command`: an ask would be
wrong too. Decider-driven cases accept done or asked, then reply Yes if asked. Where the real pipeline gets a
case wrong the expectation stays and the case is tagged `at-risk` (the blurb says what was measured).

"Measured" is one run of each file through the full runtime (`python -m homedemo.play --mode cascade --speed 4`,
2026-10-20, CPU, v1 questions, act 0.80 / ask 0.50). The cascade sends a 2B `none` from a person to the 12B, so
nearly every case below ends on a 12B answer.

| id | said (by, where) | source | expected | measured (2B -> 12B) |
|---|---|---|---|---|
| `res_01_tv_news_dollhouse` | TV news: "Alexa ordered me a dollhouse." | f2:1, BBC 2017 | keyword `none`; verdict `deny` or `ignore`; no command | PASS: 2B none 91% (TV voices stop at the 2B) |
| `res_02_codeswitch_living_dark` | Alex, living: "客厅有点暗，turn on the lights." | f2:17 | keyword `lights_on`; `lights_on` living done or asked; light on | PASS: 2B lights on 97% |
| `res_03_mandarin_cant_see` | Maya, her room: "好暗，我看不清。" | f2:16 | keyword `none`; `lights_on` bedroom_maya done or asked; light on | PASS: 2B lights on 95% |
| `res_04_dont_turn_hall_light_on` | Alex, living (Sam in the hall): "Don't turn the hall light on, Maya just fell asleep." | f2:11 | keyword `lights_on` (wrong); verdict `ignore`; no command | PASS: 2B none 81% -> 12B none 82% |
| `res_05_leaving_dont_turn_on_anything` | Alex, living, then leaves: "我不在家，don't turn on anything." | f2:18 | keyword `none`; verdict `ignore`; no command, also after the house empties | PASS: 2B none 89% -> 12B none 92% |
| `res_06_is_porch_light_on` | Alex, living: "Is the porch light on?" | state question (REPORT s.7) | keyword `lights_on` (wrong); verdict `ignore`; no command | PASS, **at-risk**: 2B none 69% -> 12B none 75% |
| `res_07_did_you_lock_the_door` | Alex, bedroom: "Did you lock the front door?" | eval:sp78 | keyword `none`; verdict `ignore`; no command to `lock_front` | **FAIL, at-risk**: 2B none 86% -> 12B lock door 61% -> asks "Lock the front door?" |
| `res_08_lights_on_mayas_room` | Alex, living (lights on): "Lights on in Maya's room please." | eval:sp38 | keyword `lights_on` in the living room (wrong room); `lights_on` bedroom_maya done or asked; light on | PASS: 2B none 58% -> 12B lights on 82%, room Maya's |
| `res_09_kitchen_lights_off_terse` | Alex, hallway by the kitchen door: "Kitchen lights off." | eval:sp42 | keyword `lights_off`; `lights_off` kitchen done or asked; light off | PASS: 2B none 58% -> 12B lights off 93% |
| `res_10_sauna` | Alex, 28 °C kitchen: "Man, it's a sauna in here." | eval:sp24 | keyword `none`; `cooler` done or asked; then done | PASS: 2B none 58% -> 12B cooler 89% |
| `res_11_child_unlock_friend` | Maya, living: "Unlock the door, my friend is here!" | f2:27 | keyword `unlock_door` (would obey); verdict `deny` (rule "kids"); no command to `lock_front` | PASS: 2B none 59% -> 12B unlock 92% -> denied |
| `res_12_guest_chilly` | guest, 17 °C living room: "Brr, is it always this chilly in here?" | f2:7 | keyword `none`; `warmer` done or asked (ask goes to the parents); then done | PASS: 2B warmer 77% -> 12B warmer 97% |
| `res_13_grandmas_heater` | Sam, kitchen: "Grandma's heater is always cranked up, it's so much warmer at her place." | f2:6 | keyword `warmer` (wrong); verdict `ignore`; no command to the thermostat | **FAIL, at-risk**: 2B none 70% -> 12B warmer 58% -> asks "Make it warmer?" |
| `res_14_if_it_gets_colder` | Alex, 20 °C living room: "If it gets any colder tonight we should turn the heat up." | eval:sp89 | keyword `none`; verdict `ignore`; no command to the thermostat | **FAIL, at-risk**: 2B warmer 69% -> 12B warmer 98% -> thermostat +1.5 °C |

11 of 14 pass; the three failures are the two weaknesses `eval/REPORT.md` section 4 names (state questions and
conditionals read as requests) plus a mention of someone else's heater that the 12B reads as a weak request.
The 2B under-acts exactly as the report says: it answered `none` to every indirect, terse, other-room and
Mandarin request here, and the cascade's "a 2B none goes to the 12B" rule is what makes them pass. Notes for
the sim and agents: a hallway voice is heard by the nearest mic (`res_09` places Alex at x=560, y=350 so the
kitchen mic hears it; the seeded hallway spot is nearest to the porch mic), and a room someone just left
publishes an `activity_changed` scene that the scene decider answers (one more 2B + 12B round, nothing happens).

## Tests (`tests/scenarios/`, no sim, no agents, no model)

`fakes.py` holds a `FakeWorld` (records steps, applies `action.command` -> `action.applied`, publishes the sensor
events a step implies) and a `RulePipeline` (keyword shadow with the real baseline's phrase list + a table-driven
"decider" keyed by utterance, with a target room for other-room requests + the house rules, the dark-room reflex
and the child-at-a-lit-stove reflex alert + an actuator that commands or asks; `delay=` publishes downstream
events detached from the bus to exercise the runner's watch window). A new utterance in the library needs a row
in `fakes.SPEECH` (or `SCENE`) or the fake answers `none`. `test_runner.py` runs every library scenario through
them and checks statuses, every expect kind, replies, the watch mark, speed scaling, stop, one-at-a-time, errors
and drain timeouts; `test_expectations.py` and `test_validate.py` cover the matchers and the validator;
`test_library.py` checks counts (12 heroes incl. the tour, 33 regression, 14 research), validity, hero pacing,
the tour's 170-190 s and its five beats, seed coverage, the research files' sourcing tags and no-op/ask rules,
and that only the verified incident is cited.
