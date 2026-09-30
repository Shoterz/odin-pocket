# Controlled refinement — execution in progress

Started 2026-09-30 09:33 Singapore time. Implementation commit: `26f0fe4`.

The experiment is running locally on the RTX 4070. This is not a completed quality improvement or a release claim.

- Verification: 65 tests passed, 1 skipped; full-size GPU deterministic restart had zero weight difference and identical source-token totals.
- Four optimizer screens: learning rates 0.0006 / 0.0012 × warmup 200 / 1000 steps. Each uses 128,008,192 tokens and 60% FineWeb / 40% WikiText.
- Two mixture screens: hold WikiText at 40%; replace FineWeb with 10% or 20% stories, using the selected optimizer.
- If a mixture passes the frozen gates, train it and its matched control from scratch for 1,000,013,824 tokens each with a fresh seed.
- Maximum charged training: 2,768,076,800 tokens, including replayed/partial steps. Smoke tests are separate. No paid compute.
- Final quality review is required before publication. A failed gate is a valid result; architecture work is conditional on a confirmed improvement.

Live stage: `results/controlled/status.json`. Per-run telemetry and append-only work ledger: `runs/controlled/<run>/metrics.jsonl` and `work.jsonl`. Stage logs: `results/controlled/*.log`. The runner performs automatic selection and confirmation, then stops for blinded generation review; it never publishes a model automatically.

## Historical references on the new screening split

These are diagnostic references with different training budgets, not matched experiment controls. The 285 confirmation questions remain unused until the confirmation stage.

| Checkpoint | ARC validation screening accuracy | FineWeb NLL | WikiText NLL | Story NLL |
|---|---:|---:|---:|---:|
| Published baseline | 37.89% | 3.40472 | 3.11084 | 2.86566 |
| Previous B (3B tokens) | 39.30% | 3.26149 | 3.22504 | 1.21189 |

ARC scores use mean continuation-token likelihood and are not the official harness's test results. The small observed accuracy difference is not evidence of significance.

## Recovery

From this research worktree, rerun `/home/ylz/Desktop/odin_llm/project/.venv/bin/python scripts/controlled_refinement.py` with GPU access. Do not start another runner while the current one holds the lock. Existing data, checkpoints, reports and decisions are verified before reuse. If a failure occurs, inspect its log; never remove the experiment lock file or alter frozen source to force a resume. The charged compute cap may prevent completing all remaining stages after repeated interruptions.
