# Meeting decider eval (eval/meeting_cases.json, 72 labelled utterances)

Cases: 13 tasks (named owner, volunteer, "you", with/without dates, Mandarin and mixed), 8 decisions, 9 questions
for the room (incl. one "share the doc in the channel" = share_docs), 10 room-comfort lines, 8 sensitive
(pay / HR / health / legal / layoffs), 8 small talk + jokes ("let's just fire Bob, ha ha" = none), 4 off-topic,
9 ordinary discussion (3 "hard" negatives: past tense / other place), 3 media lines (the room screen talking,
accepted answers none or the literal action because the policy denies media anyway). Checked-in names: Priya
(owner), Sam, Lin. Due dates are judged on presence (the parser is deterministic code, tested separately).
Run: `.venv/bin/python eval/run_meeting_eval.py --model 2b|12b|keyword --version v2|v3|v4`; one row per run
below; full per-case output in eval/runs/ (gitignored).

## Question versions (unspoken/meeting/questions.py)

* **v2** one 8-way `kind` choice (task, decision, question, room request, sensitive, small talk, off topic,
  nothing) + `owner` choice + `room` choice (always) + `sensitive` noul (noisy-OR). The 2B pulled almost every
  task and question into "Nothing" (raw choice "Nothing" at 0.65-0.85 for "I'll draft the launch email
  tomorrow", "When is the press briefing again?"): 60% accuracy, 44% coverage at 0.8.
* **v3** (chosen) adds a yes/no `act` gate ("did they give a to-do / decide / ask the room / want the room
  changed?") folded in as a product of experts, merges chatter / off-topic / nothing into one sharp "Nothing to
  do" option (6 options), and asks the `room` choice only when the words mention the room (fewer scored
  options, same request). 2B: 78% accuracy; 89% accuracy at >= 0.6 with 75% coverage and 97% act precision;
  decisions 8/8, Mandarin 8/8, media 3/3, hard negatives 3/3. Weak spot: factual questions ("When is the press
  briefing again?") still read as "nothing" (3/9).
* **v4** (rejected) rewrote the gate and the "question" option so factual questions count as "for the room".
  On the first 36 cases it was WORSE than v3 (64% vs 72%; owner 67% vs 75%): the broader gate let more
  chatter through and did not rescue the questions. The run was stopped there (see the partial numbers below).

## Operating point

act_threshold **0.60**, ask_threshold **0.40** (config/meeting.toml): on the 2B v3 run, 0.6 gives 89%
accuracy at 75% coverage and 96.7% act precision, which dominates 0.7 (87% / 65%) and 0.8 (89% / 50%). Below
0.4 the room ignores the decider; between 0.4 and 0.6 low-risk actions (light, fan, answer) still act, medium
risk (task, decision) asks the owner or, with the llm role enabled, goes to the local LLM.

## 12B (gemma-4-12B decider, CPU here)

v3, first 31 of 72 cases (the full run was still going when this was written; the row below is appended when
it finishes): 90% accuracy, 92% at >= 0.6 with 84% coverage, **100% act precision at every threshold**,
owner 92% (n=12), args 86% (n=14); tasks 11/12, decisions 8/8, questions 6/8, room 3/3. The two misses were
"Priya, can you confirm the August 1 date with legal?" (read as sensitive, 0.41) and two factual questions read
as "nothing". Median 40 s per request on this CPU (the 12B is the GB10's GPU decider; expect well under a second).

## Plain-code keyword baseline (fakes.KeywordDecider)

85% on these cases, which it was partly written against (the sim's fake profile uses it): it is a sanity
check of the harness, not a candidate. It cannot read "Is it just me or is it boiling in this room?" or
"Let's lock the launch name" (0/2), where the 12B does.

## Latency

Per request (all questions in one call, CPU): 2B ~9.3 s alone, 12-15 s with other runs sharing the CPU;
12B ~40 s. The brief's 2.4 s / 9 s figures are for one question; this request scores ~16-20 options.
On the GB10 (Blackwell, 12B) the same request is expected at a few hundred ms; the sim demo passed 17/17 with
the real 2B at ~9 s per decision because the scenario windows allow 45 s.

## 2B v4 partial (first 36 cases, stopped)

n=36 acc=0.639; conf>=0.6: coverage 0.72, acc 0.654, act precision 0.944; conf>=0.8: coverage 0.25, acc 0.889.
Same prefix with v3: acc 0.722; conf>=0.6: coverage 0.67, acc 0.833, act precision 1.0.

## Runs

| stamp | version | model | n | acc | acc@0.8 (cov) | acc@0.9 (cov) | act prec / recall @0.8 | owner | args | Brier | median / p90 ms | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261024-232330 | v2 | keyword | 72 | 85% | 96% (71%) | 100% (10%) | 97% / 79% | 77% | 73% | 0.11 | 0 / 0 | plain-code baseline |
| 20261024-232330 | v2 | 2b | 72 | 60% | 78% (44%) | 100% (14%) | 100% / 12% | 23% | 36% | 0.239 | 9348 / 9928 | first run |
| 20261024-233617 | v3 | 2b | 72 | 78% | 89% (50%) | 93% (21%) | 95% / 40% | 77% | 73% | 0.141 | 12831 / 17578 | gate+6 kinds+room auto |
