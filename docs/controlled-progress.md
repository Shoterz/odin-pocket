# Controlled refinement — screening completed

Started 2026-09-30 09:33 Singapore time. Implementation commit: `26f0fe4`.

All six screening runs completed on the RTX 4070 at 11:32 Singapore time on September 30, 2026. Neither story mixture beat the no-story control, so the runner skipped the conditional confirmation jobs. Evaluation and continuation review found no demonstrated overall improvement. Read the [completed results and assessment](controlled-results.md).

- Verification: 65 tests passed, 1 skipped; full-size GPU deterministic restart had zero weight difference and identical source-token totals.
- Four optimizer screens: learning rates 0.0006 / 0.0012 × warmup 200 / 1000 steps. Each uses 128,008,192 tokens and 60% FineWeb / 40% WikiText.
- Two mixture screens: hold WikiText at 40%; replace FineWeb with 10% or 20% stories, using the selected optimizer.
- Confirmation condition was not met: neither planned 1,000,013,824-token run was launched. The 285 confirmation questions remain unused.
- Maximum charged training: 2,768,076,800 tokens, including replayed/partial steps. Smoke tests are separate. No paid compute.
- Quality review completed on the screening checkpoints and previous models. Retain the existing release. Architecture work remains conditional on a confirmed improvement.

Final stage: `completed-no-mixture-winner` in `results/controlled/status.json`. Per-run telemetry and append-only work ledger: `runs/controlled/<run>/metrics.jsonl` and `work.jsonl`. Stage logs: `results/controlled/*.log`. Charged training: 768,049,152 tokens; training subprocess time: 1.96 hours. No model was published automatically.

## Historical references on the new screening split

These are diagnostic references with different training budgets, not matched experiment controls. The 285 confirmation questions remain unused until the confirmation stage.

| Checkpoint | ARC validation screening accuracy | FineWeb NLL | WikiText NLL | Story NLL |
|---|---:|---:|---:|---:|
| Published baseline | 37.89% | 3.40472 | 3.11084 | 2.86566 |
| Previous B (3B tokens) | 39.30% | 3.26149 | 3.22504 | 1.21189 |

ARC scores use mean continuation-token likelihood and are not the official harness's test results. The small observed accuracy difference is not evidence of significance.

## Recovery

This run finished successfully and needs no recovery. Rerunning `/home/ylz/Desktop/odin_llm/project/.venv/bin/python scripts/controlled_refinement.py` verifies and reuses the frozen results, then reaches the same no-mixture-winner outcome. A follow-up experiment must have a separate protocol; do not alter frozen source or decisions to force confirmation.
