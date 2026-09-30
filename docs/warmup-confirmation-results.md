# Long warmup confirmation results

The primary comparison did not establish an improvement under the preregistered gates. Qualitative review is pending; no model has been promoted automatically.

Both new models trained from random initialization for 1,000,013,824 tokens with the same fresh seed, 60/40 FineWeb/WikiText mixture, learning rate 0.0006 and architecture. Only warmup differs: short=200 steps, long=1000. Both finished before any confirmation evaluation. Historical models have different budgets/recipes and are descriptive references.

| Model | Training tokens | Correct | Confirmation accuracy | FineWeb NLL ↓ | WikiText NLL ↓ |
|---|---:|---:|---:|---:|---:|
| short | 1,000,013,824 | 78/285 | 27.37% | 3.3645 | 3.0921 |
| long | 1,000,013,824 | 81/285 | 28.42% | 3.3703 | 3.0986 |
| published | 983,040,000 | 77/285 | 27.02% | 3.4047 | 3.1108 |
| previous-B | 3,000,008,704 | 89/285 | 31.23% | 3.2615 | 3.2250 |

Long minus short accuracy: +1.05 percentage points; paired 95% question-bootstrap interval [-2.81, +4.91]. General-domain regression guard passed: True. This does not measure training-seed uncertainty or establish broad intelligence. Scores use the frozen 285-question ARC validation confirmation split and mean continuation-token likelihood; they are not official test scores.

New charged tokens: 2,000,027,648; cumulative including screens: 2,768,076,800. New training subprocess elapsed time: 5.082 hours. Preparation and evaluation are additional.

Inspect every item in `results/warmup-confirmation/blind-review.json` using `docs/controlled-quality-rubric.md` before reading its mapping. Some historical text may be recognizable. Author review is not independent human validation. No architecture experiment or public release is triggered by this numerical report.
