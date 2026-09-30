# Warmup confirmation progress

The matched long comparison started on September 30, 2026 at 13:10 Singapore time on the local RTX 4070. Automatic evaluation finished at 18:19 Singapore time; qualitative review is now complete. Implementation and frozen protocol commit: `79012ff`. The primary improvement gate failed; retain the published model. See [final assessment](warmup-confirmation-review.md) for the evaluated results, limitations and next step.

| Condition | Short warmup | Long warmup |
|---|---:|---:|
| Warmup steps | 200 | 1000 |
| Training tokens | 1,000,013,824 | 1,000,013,824 |
| Initialization seed | 20261001 | 20261001 |
| Peak learning rate | 0.0006 | 0.0006 |
| FineWeb / WikiText / stories | 60% / 40% / 0% | 60% / 40% / 0% |

Both started from random initialization using the same 49,295,872-parameter architecture and fixed tokenizer. Both completed their full budget, using 5.082 training subprocess hours in total. Actual source token counts match exactly and no replay was charged.

Both training runs finished before either was evaluated on the 285 reserved confirmation questions. The published model and previous B were evaluated on the same questions. The automatic numerical report is `docs/warmup-confirmation-results.md`; the subsequent completed review is `docs/warmup-confirmation-review.md`. Long warmup scored 81/285 versus short warmup's 78/285, with a paired interval spanning zero. All 64 new continuations had a clear causal/factual failure under the author rubric. No model was promoted and no further training was launched.

Preflight: 71 tests passed, one skipped; verified existing full-size GPU restart evidence against unchanged trainer hashes. Fresh code review's recovery finding was fixed: before resuming either model, the runner checks whether both can still finish within the charged budget. Prior screening data, code, results and decisions are preserved.

Automatic runner status: `results/warmup-confirmation/status.json` (preserved at its final pre-review stage). Final review decision: `results/warmup-confirmation/review-decision.json`. Logs and per-attempt timing: `results/warmup-confirmation/`. Training checkpoints and telemetry: `runs/controlled/warmup-confirm-200/` and `runs/controlled/warmup-confirm-1000/`.

New charged budget is 2,000,027,648 tokens. Cumulative budget including the six screens is 2,768,076,800. This cap includes partial/replayed steps and has no replay allowance. If an interruption leaves charged work beyond a saved checkpoint, the runner stops before spending more on an unaffordable pair. It does not silently extend the budget.

The experiment is complete; no recovery run is needed. The frozen experiment files and automatic outputs remain unchanged, with final interpretation recorded separately.
