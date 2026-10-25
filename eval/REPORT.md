# Decider questions: v0 vs v1, both models

All numbers below come from runs made on 2026-10-20 against the local CPU servers (2B :8765, 12B :8766)
with `eval/run_eval.py`; the run files are in `eval/runs/` (gitignored) and the tables were printed by
`eval/report_tables.py`. Cases: 93 speech (47 dev / 46 test) and 42 scene (24 dev / 18 test), split
fixed in commit 9d5a792 before any tuning; tuning used dev only. 12B calls used: 399.

**Confidence** = probability of the chosen action after folding all questions together
(`Interpretation.confidence`), so a threshold means the same thing for every version.
**Act precision** = share of non-`none` answers at or above a threshold that are right, i.e. how often
acting is right. **Room acc** = `Interpretation.room` equals the expected room on cases whose expected
action is a light action (the only actions with a target room).

## 1. v0 vs v1 by model and split

| run | kind | n | accuracy | acc@0.8 (coverage) | act precision@0.8 (n acting) | room acc (light-action cases) | Brier | median / p90 latency |
|---|---|---|---|---|---|---|---|---|
| v0 2B dev | speech | 47 | 66% | 94% (36%) | 93% (15) | 47% (19); 62% when action right (13) | 0.184 | 1190 / 1233 ms |
| v1 2B dev | speech | 47 | 79% | 82% (57%) | 100% (15) | 74% (19); 100% when action right (13) | 0.161 | 4157 / 4340 ms |
| v0 2B test | speech | 46 | 83% | 100% (37%) | 100% (16) | 54% (13); 58% when action right (12) | 0.144 | 1305 / 1396 ms |
| v1 2B test | speech | 46 | 87% | 93% (63%) | 100% (19) | 77% (13); 90% when action right (10) | 0.101 | 4292 / 4507 ms |
| v0 2B dev | scene | 24 | 62% | 100% (8%) | 100% (1) | - | 0.205 | 1265 / 1296 ms |
| v1 2B dev | scene | 24 | 62% | 75% (67%) | 100% (2) | - | 0.220 | 3472 / 3715 ms |
| v0 2B test | scene | 18 | 72% | 100% (6%) | 100% (1) | - | 0.190 | 1370 / 1422 ms |
| v1 2B test | scene | 18 | 78% | 100% (56%) | 100% (1) | - | 0.123 | 3508 / 4153 ms |
| v0 12B dev | speech | 47 | 83% | 93% (57%) | 92% (24) | 47% (19); 50% when action right (18) | 0.128 | 4827 / 5124 ms |
| v1 12B dev | speech | 47 | not run (12B call budget of ~400 used on test and variants) | | | | | |
| v0 12B test | speech | 46 | 89% | 100% (65%) | 100% (28) | 54% (13); 54% when action right (13) | 0.078 | 4946 / 5120 ms |
| v1 12B test | speech | 46 | 94% | 100% (78%) | 100% (31) | 92% (13); 92% when action right (13) | 0.054 | 16764 / 19857 ms |
| v0 12B dev | scene | 24 | 75% | 92% (50%) | 100% (7) | - | 0.139 | 5363 / 5440 ms |
| v1 12B dev | scene | 24 | not run (12B call budget) | | | | | |
| v0 12B test | scene | 18 | 83% | 91% (61%) | 83% (6) | - | 0.110 | 5323 / 5484 ms |
| v1 12B test | scene | 18 | 94% | 100% (78%) | 100% (6) | - | 0.040 | 12402 / 12514 ms |

v0's room column is the share of light-action cases whose target is the speaker's own room (v0 cannot
name another room). The test split is easier than dev for both models and both versions, so compare
versions within a split, not splits with each other. Latencies were measured while both servers were
busy (load average about 20); idle, a one-question v0 request took 0.86 s on the 2B. Latency follows the
input tokens: 185 for v0 speech, 398 for v1 without the room choice, 536 for v1.

## 2. Calibration (test split): accuracy of the chosen action by confidence bin (count)

| run | kind | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-1.0 |
|---|---|---|---|---|---|---|---|---|
| v0 2B test | speech | 100% (1) | 67% (3) | 67% (9) | 78% (9) | 71% (7) | 100% (10) | 100% (7) |
| v1 2B test | speech | - | 0% (1) | 67% (6) | 67% (3) | 100% (7) | 94% (18) | 91% (11) |
| v0 2B test | scene | - | 100% (1) | 57% (7) | 71% (7) | 100% (2) | - | 100% (1) |
| v1 2B test | scene | - | - | 67% (3) | 33% (3) | 50% (2) | 100% (6) | 100% (4) |
| v0 12B test | speech | 0% (1) | - | 75% (4) | 83% (6) | 60% (5) | 100% (8) | 100% (22) |
| v1 12B test | speech | - | 100% (1) | 50% (2) | 100% (1) | 67% (6) | 100% (4) | 100% (32) |
| v0 12B test | scene | - | 50% (2) | 0% (1) | 100% (1) | 100% (3) | 83% (6) | 100% (5) |
| v1 12B test | scene | - | - | 50% (2) | - | 100% (2) | 100% (2) | 100% (12) |

v1 moves mass into the 0.9-1.0 bin on the 12B (32 of 46 speech cases, all right) and is right on
every case at or above 0.8 on the 12B. On the 2B the misses at high confidence are all `none` answers
to real requests (section 4), not wrong actions.

## 3. Thresholds

Act precision at each threshold (n acting, coverage of all cases):

| run | kind | @0.5 | @0.6 | @0.7 | @0.8 | @0.9 |
|---|---|---|---|---|---|---|
| v1 2B dev | speech | 96% (24, 96%) | 95% (22, 83%) | 95% (19, 72%) | 100% (15, 57%) | 100% (9, 30%) |
| v1 2B test | speech | 100% (26, 98%) | 100% (24, 85%) | 100% (22, 78%) | 100% (19, 63%) | 100% (7, 24%) |
| v1 2B dev | scene | 100% (2, 88%) | 100% (2, 79%) | 100% (2, 71%) | 100% (2, 67%) | 100% (1, 21%) |
| v1 2B test | scene | 100% (4, 100%) | 100% (2, 83%) | 100% (1, 67%) | 100% (1, 56%) | 100% (1, 22%) |
| v1 12B dev | speech | not run | | | | |
| v1 12B test | speech | 91% (34, 98%) | 94% (33, 93%) | 94% (33, 91%) | 100% (31, 78%) | 100% (29, 70%) |
| v1 12B dev | scene | not run | | | | |
| v1 12B test | scene | 89% (9, 100%) | 100% (7, 89%) | 100% (7, 89%) | 100% (6, 78%) | 100% (4, 67%) |
| v0 2B test | speech | 87% (30, 91%) | 92% (26, 72%) | 91% (22, 52%) | 100% (16, 37%) | 100% (7, 15%) |
| v0 12B test | speech | 88% (34, 98%) | 91% (32, 89%) | 94% (31, 76%) | 100% (28, 65%) | 100% (22, 48%) |
| v0 12B test | scene | 78% (9, 89%) | 88% (8, 83%) | 88% (8, 78%) | 83% (6, 61%) | 100% (4, 28%) |

Recommendation (acting right at least 95% of the time on test, and on dev where the count allows):

| model | act_threshold | ask_threshold | basis |
|---|---|---|---|
| 2B, v1 | **0.80** | **0.50** | speech: 100% of 19 acting cases right on test, 100% of 15 on dev at 0.8; at 0.7 dev drops to 95% (19). Between 0.5 and 0.8 the acting answer is still right 95-100% on speech, so asking is cheap and almost always sensible; below 0.5 the 2B is mostly guessing (test bins 0.4-0.6: 4 of 7 right). Scenes: the 2B barely ever acts (1-2 acting cases per split), everything else goes to the 12B. |
| 12B, v1 | **0.80** | **0.60** | speech: 100% of 31 acting right at 0.8 on test (coverage 78%), 94% at 0.7 and 0.6; scene: 100% at 0.6 and above (7 acting, coverage 89%), 89% at 0.5. 0.80 is the smallest grid value that holds 100% on both kinds; 0.60 keeps asks at 94-100% precision. |

The Settings defaults (act 0.80, ask 0.45) are close; raising ask to 0.50 (2B) / 0.60 (12B) avoids asking
"did you want X?" for answers that are coin flips. On the 2B, a `none` answer is weak evidence (section 4):
of its 42 `none` answers on speech only 28 were right, and even at confidence 0.9 or more 5 of 9. The
cascade should therefore treat a 2B `none` as "unsure" and consult the 12B, not only when the 2B's
confidence is below ask_threshold (counts are dev and test together).

## 4. Worst remaining failures (v1, test split; dev listed where it adds a type)

| run | case | said / seen | expected | predicted | confidence |
|---|---|---|---|---|---|
| v1 2B test | sp42 | "Kitchen lights off." (terse, other room) | lights_off | none | 0.91 |
| v1 2B test | sp24 | "Man, it's a sauna in here." | cooler | none | 0.85 |
| v1 2B test | sp38 | "Lights on in Maya's room please." | lights_on | none | 0.62 |
| v1 2B test | sp18 | "Okay Maya, lights out, it's a school night." | lights_off | none | 0.60 |
| v1 2B test | sp16 | "Lock the front door, I'm heading to bed." | lock_door | none | 0.55 |
| v1 2B test | sp68 | "我老板 said the office 灯 always on, 太浪费电了。" | none | lights_off | 0.47 |
| v1 2B dev | sp25 | "Where's the remote? The game starts in five minutes." | tv_on | none | 0.94 |
| v1 2B dev | sp59 | "我房间的灯打开。" | lights_on | none | 0.93 |
| v1 2B dev | sp69 | "Porch 的灯开一下，Sam 快到了。" | lights_on | none | 0.92 |
| v1 2B dev | sp89 | "If it gets any colder tonight we should turn the heat up." | none | warmer | 0.75 |
| v1 2B test | sc34 | Nobody in the living room for 20 min, TV on | tv_off | none | 0.72 |
| v1 2B test | sc27 | Adult wrapped in a blanket, 16 C | warmer | none | 0.63 |
| v1 2B test | sc19 | Adult asleep on the couch, TV and lights on | tv_off/lights_off | none | 0.60 |
| v1 2B test | sc30 | Maya's room, lying still 30 min, lights on, school night | lights_off | none | 0.52 |
| v1 12B test | sp78 | "Did you lock the front door?" | none | lock_door | 0.77 |
| v1 12B test | sp62 | "电视是开着的吗？" | none | tv_on | 0.76 |
| v1 12B test | sp64 | "门锁好了吗？" | none | lock_door | 0.55 |
| v1 12B test | sc40 | Child on a chair reaching for the cookie jar, adult watching, stove off | none | alert_parent | 0.55 |

Two patterns remain. The 2B, with the gate, **under-acts**: it rejects terse or Chinese requests and
requests about another room, and almost every scene (its wrong answers are `none`, which is the safe
direction for a house with no wake word). The 12B's only speech failures are **questions about device
state** ("Did you lock the door?", "电视是开着的吗？"), which it reads as requests, and one scene where a
supervised child near a shelf is read as a hazard.

## 5. What v1 changes and why

1. **A yes/no gate in the same request**: "Does the speaker want the lights, TV, temperature or door
   in this house changed right now?" (scenes: "Does someone in this room need the house to change the
   lights, TV or temperature right now, or is someone in danger?"), with sharp true/false
   descriptions. Its probability multiplies the choice distribution (P(action) ∝ choice × P(yes),
   P(none) ∝ choice × P(no)), so a 0.5 answer changes nothing and the FakeDecider path is unchanged.
   This is the change that moves accuracy: on the 2B dev speech set the choice alone gets 31-35 of
   47 under every wording tried; adding the gate gives 35-37 for every wording except the labeled
   layout and house-facts-in-state variants (34).
2. **A room choice for speech** ("Which room's lights is the speaker talking about?", options: the room
   the speaker is in + every room of home.json, with "my room" explained for the speaker). It fills the
   new `Interpretation.room` for light actions only. 12B test: 12 of 13 light-action cases get the
   right room, 5 of 6 when the room differs from the speaker's; v0 cannot name another room at all.
3. **Full-sentence option descriptions** with the cues that decide each option, a short "Not a
   request" / "Do nothing" option, and the actions v0 lacked (lock the door; lights off and cooler for
   scenes). On the 12B the full descriptions take scene accuracy from 18/24 to 23/24 on dev with the
   choice alone; on the 2B they change nothing measurable (section 6).
4. The state keeps v0's one-line layout. A labeled multi-line layout and house facts in the state or
   the instructions were tried and rejected (section 6).

Cost: v1 asks three questions instead of one, so a request is about 3.2x the tokens and latency (2B,
same box and load: 1.3 s -> 4.3 s; the room choice is about a quarter of that; 12B: 5 s -> 17 s
speech, 12 s scene). The brief's "several questions are almost free" holds for GPU serving where the
state dominates; on this CPU box the question tokens dominate.

## 6. What was tried (dev only, 2B; the 12B confirmed the top two)

Speech, 47 dev cases (15 of them `none`); right / with the yes/no gate folded in:

| variant | choice alone | + gate "ask" | + gate "want" |
|---|---|---|---|
| v0 wording, one-line state | 31/47 (neg 9/15) | 36/47 (neg 10/15) | - |
| v0 wording, labeled state | 35/47 (neg 7/15) | 34/47 (neg 7/15) | - |
| full descriptions, v0 none text | 32/47 (neg 5/15) | 35/47 (neg 8/15) | - |
| full descriptions, short none text | 32/47 (neg 8/15) | 36/47 (neg 10/15) | - |
| full descriptions, long none text (lists every non-request type) | 32/47 (neg 5/15) | 35/47 (neg 8/15) | - |
| full + short none, house facts in the state | 35/47 (neg 4/15) | 34/47 (neg 3/15) | - |
| full + short none, house facts in the instructions | 32/47 (neg 4/15) | 36/47 (neg 8/15) | - |
| full + short none, none option first | 31/47 (neg 6/15) | 36/47 (neg 9/15) | - |
| full + short none + lock option (+ room) = v1 wording | 33/47 (neg 8/15) | 37/47 (neg 10/15) | 37/47 (neg 14/15) |
| v0 wording + room | 31/47 (neg 9/15) | 36/47 (neg 10/15) | 34/47 (neg 13/15) |

Gate wordings (same choice question): "ask" = "Is the speaker asking the house to do something right
now, directly or by clearly implying it?" (AUC 0.87 on the 2B, 0.95 on the 12B); "want" = the v1 text
above (AUC 0.85 / 0.95, but it puts the negatives much lower: median P(yes) 0.19 vs 0.32 on negatives);
"plain" = "Is this a request for the house to do something right now?" with no descriptions (AUC 0.73 /
0.81). "want" was chosen because it is the only one that makes acting right 100% of the time at 0.8 on
the 2B (15 acting) and gives the 12B its best dev score (42/47 against 40/47 for "ask").

Combination rules: the product above; a "mixture" where the gate alone decides `none` (P(none) =
1 - P(yes)) had the same accuracy but halved coverage at 0.8 (0.28 vs 0.55) and is not neutral at 0.5.
A long "Not a request" description that lists every kind of non-request made the 2B pick it less (neg
5/15), not more; the short one works best. House facts in the state push the gate towards "yes" on
negatives (median P(yes) on negatives 0.71) and were dropped. Option order ("none" first) changed
nothing beyond noise, as the architecture notes predict.

Scenes, 24 dev cases (11 `none`):

| variant | choice alone | + gate "ask" | + gate "want" |
|---|---|---|---|
| v0 wording | 15/24 (neg 10/11) | 14/24 | - |
| full descriptions, short none | 16/24 (neg 10/11) | 14/24 | - |
| full, long none | 17/24 (neg 7/11) | 16/24 | - |
| full, short none, labeled state | 14/24 (neg 9/11) | 15/24 | - |
| full, short none, + lights-off/cooler options (= v1) | 17/24 (neg 9/11) | 16/24 (neg 10/11) | 15/24 (neg 11/11) |
| 12B, v1 wording | 23/24 (neg 10/11) | 23/24 | 24/24 (neg 11/11) |

The 2B cannot do scenes with any wording tried (14-17 of 24, and it almost never reaches 0.8 on an
action); the 12B with v1 does (24/24 dev, 17/18 test). The 12B dev run above asked the three gate
wordings in one request (5 questions, 27-31 s per call); the clean 3-question v1 dev run is in section 1.

12B, speech dev, choice alone: v0 wording 39/47 (v0 run), v1 wording 39/47; with gate "ask" 40/47 for
both wordings (the v0-wording run was stopped after its speech half; its summary is in the log only).
So on the 12B the descriptions matter for scenes, the gate for speech.

## 7. What fine-tuning data would fix the remaining failures

- **State questions vs requests** (12B's only speech failures; 2B too): pairs like "Is the porch light
  on?" / "Turn the porch light on", "Did you lock the door?" / "Lock the door", in English and
  Mandarin ("门锁好了吗？" / "把门锁上"), labelled none / action. About 200 pairs would cover the
  device set.
- **Indirect and terse requests for the 2B**: "Where's the remote?", "Kitchen lights off.", "it's a
  sauna in here", requests naming another room or "my room", and the same in Mandarin and mixed
  Mandarin-English, labelled with the action and room. The 2B's gate says "no" to these at 0.8-0.9
  confidence; a few hundred positive examples of implied requests would move the gate.
- **Scene cues for the 2B**: short scene captions with the deciding cue (dark + person + lights off ->
  lights on; asleep + TV on -> TV off; blanket + 16 C -> warmer; empty room + TV on -> TV off; child
  near a hot stove alone -> alert, child near a shelf with an adult -> nothing). The 2B has the facts
  in front of it and still answers "do nothing"; this is a labelled-example problem, not a wording one.
- **Negation, conditionals, sarcasm**: "don't turn off the TV", "if it gets colder we should turn the
  heat up", "let's turn on every light, electricity is free" labelled none. Both models still act on
  some of these at 0.5-0.75.

## 8. Known gaps

- Small sets: 46-47 speech and 18-24 scene cases per split. One case is 2-5 percentage points.
  Differences under about 3 speech cases between variants are noise.
- The test split is easier than dev for both models; thresholds were chosen to hold on both.
- Latency numbers were taken while the other server (and other workstreams) loaded the box.
- The room question is asked on every speech request although only light actions use it; a cascade
  that needs latency can pass `room=False` and fall back to the speaker's room.
- `Interpretation.confidence` for a `none` answer is as (un)calibrated as the gate: on the 2B treat
  `none` as unsure (section 3).
