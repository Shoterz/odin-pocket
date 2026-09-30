# Warmup confirmation progress

The matched long comparison started on September 30, 2026 at 13:10 Singapore time on the local RTX 4070. Implementation and frozen protocol commit: `79012ff`. This is an active experiment; no improvement has been established.

| Condition | Short warmup | Long warmup |
|---|---:|---:|
| Warmup steps | 200 | 1000 |
| Training tokens | 1,000,013,824 | 1,000,013,824 |
| Initialization seed | 20261001 | 20261001 |
| Peak learning rate | 0.0006 | 0.0006 |
| FineWeb / WikiText / stories | 60% / 40% / 0% | 60% / 40% / 0% |

Both start from random initialization using the same 49,295,872-parameter architecture and fixed tokenizer. The short-warmup run trains first, followed automatically by the long-warmup run. Expected duration is approximately five hours plus evaluation, based on the completed screens' throughput; this is an estimate.

Both training runs must finish before either is evaluated on the 285 reserved confirmation questions. The runner then evaluates the published model and previous B on those same questions, generates the fixed continuation suite, creates anonymized review items, and writes `docs/warmup-confirmation-results.md`. That numerical report leaves qualitative review pending; it does not publish a model or launch architecture experiments.

Preflight: 71 tests passed, one skipped; verified existing full-size GPU restart evidence against unchanged trainer hashes. Fresh code review's recovery finding was fixed: before resuming either model, the runner checks whether both can still finish within the charged budget. Prior screening data, code, results and decisions are preserved.

Live status: `results/warmup-confirmation/status.json`. Logs and per-attempt timing: `results/warmup-confirmation/`. Training checkpoints and telemetry: `runs/controlled/warmup-confirm-200/` and `runs/controlled/warmup-confirm-1000/`.

New charged budget is 2,000,027,648 tokens. Cumulative budget including the six screens is 2,768,076,800. This cap includes partial/replayed steps and has no replay allowance. If an interruption leaves charged work beyond a saved checkpoint, the runner stops before spending more on an unaffordable pair. It does not silently extend the budget.

Recovery command from this worktree: `/home/ylz/Desktop/odin_llm/project/.venv/bin/python scripts/warmup_confirmation.py` with GPU access. The shared writer lock prevents concurrent pipelines. Existing checkpoints and reports are reused only after identity verification. Keep the frozen experiment files unchanged.
