# RetinaSathi SimEvents verification

Generated: 2026-09-22 19:48:14 *

> This verifies software behavior and configured planning scenarios. It does not prove clinical safety.

| Check | Result | Evidence |
|---|---|---|
| SimEvents installed | PASS | 26.1 |
| SimEvents licence | PASS | license('test','SimEvents') must return 1 |
| Required blocks | PASS | 13/13 required blocks found |
| Baseline produces visits | PASS | 51 arrivals; 51 completed |
| Outcome accounting | PASS | routine + reviewed equals completed |
| Utilization bounds | PASS | all utilization values are 0–1 |
| Bandwidth calculation | PASS | 1.500 minutes/image |
| Bottleneck generated | PASS | camera at 71.7% |
| Total screening time | PASS | 16.17 minutes |
| Seven scenarios | PASS | 7 scenarios available |
| Network constraint changes delay | PASS | 8.10 vs 1.50 min |
| Extra camera improves throughput | PASS | 105 vs 68 completed |
| Extra camera reduces queue | PASS | 4 vs 38 waiting |
| District exceeds 100k planning target | PASS | 127000 annual-equivalent screenings |
