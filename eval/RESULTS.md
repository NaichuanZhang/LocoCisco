# Eval runs

One row per run and kind, appended by eval/run_eval.py. acc@t (cov) = accuracy among cases with confidence >= t and the share of cases that reach t; act prec@0.8 = share of non-none predictions at confidence >= 0.8 that are right; room = Interpretation.room matches the expected room (speech).

| stamp | version | model | kind | split | n | acc | acc@0.8 (cov) | acc@0.9 (cov) | act prec@0.8 | room | Brier | median / p90 ms | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261020-031331 | v0 | 2b | speech | dev | 47 | 66% | 94% (36%) | 100% (13%) | 93% | 47% | 0.184 | 1190 / 1233 | baseline |
| 20261020-031331 | v0 | 2b | scene | dev | 24 | 62% | 100% (8%) | 100% (4%) | 100% | - | 0.205 | 1265 / 1296 | baseline |
| 20261020-031508 | v1 | 2b | speech | dev | 47 | 70% | 90% (66%) | 91% (49%) | 90% | 90% | 0.168 | 4320 / 4551 | v1 candidate |
| 20261020-031508 | v1 | 2b | scene | dev | 24 | 67% | 100% (8%) | 100% (4%) | 100% | - | 0.242 | 3406 / 3451 | v1 candidate |
| 20261020-032144 | v0 | 12b | speech | all | 93 | 86% | 96% (61%) | 98% (47%) | 96% | 50% | 0.103 | 4837 / 5120 | baseline |
| 20261020-032144 | v0 | 12b | scene | all | 42 | 79% | 91% (55%) | 100% (26%) | 92% | - | 0.127 | 5339 / 5457 | baseline |
| 20261020-041958 | v0 | 2b | speech | test | 46 | 83% | 100% (37%) | 100% (15%) | 100% | 54% | 0.144 | 1305 / 1396 | baseline |
| 20261020-041958 | v0 | 2b | scene | test | 18 | 72% | 100% (6%) | 100% (6%) | 100% | - | 0.19 | 1370 / 1422 | baseline |
| 20261020-043756 | v1 | 2b | speech | dev | 47 | 79% | 82% (57%) | 79% (30%) | 100% | 74% | 0.161 | 4157 / 4340 | final v1 |
| 20261020-043756 | v1 | 2b | scene | dev | 24 | 62% | 75% (67%) | 100% (21%) | 100% | - | 0.22 | 3472 / 3715 | final v1 |
| 20261020-044236 | v1 | 2b | speech | test | 46 | 87% | 93% (63%) | 91% (24%) | 100% | 77% | 0.101 | 4292 / 4507 | final v1 |
| 20261020-044236 | v1 | 2b | scene | test | 18 | 78% | 100% (56%) | 100% (22%) | 100% | - | 0.123 | 3508 / 4153 | final v1 |
| 20261020-045315 | v1 | 12b | speech | test | 46 | 94% | 100% (78%) | 100% (70%) | 100% | 92% | 0.054 | 16764 / 19857 | final v1 |
| 20261020-045315 | v1 | 12b | scene | test | 18 | 94% | 100% (78%) | 100% (67%) | 100% | - | 0.04 | 12402 / 12514 | final v1 |
