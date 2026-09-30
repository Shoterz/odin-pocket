# Beta2 experiment progress

September 30, 2026. The user approved execution of the [deadline research plan](deadline-research-2026-09-30.md). Implementation is committed as `7e32db6` on `research/refinement-v2`; historical trainers, results, and the published model are preserved.

The single candidate changes AdamW beta2 from 0.95 to 0.999. It retains the 49,295,872-parameter model, original tokenizer, 60/40 FineWeb-Edu/WikiText data, seed 20261001, LR 0.0006, warmup 1000, batch 32, context 512 and 61,036-step budget. The existing 1B long-warmup checkpoint is the matched control.

## Verification

- Full host suite: 91 passed, 1 skipped. The skipped test checks CUDA-unavailable handling; CUDA is available on the host. The sandbox's HTTP socket restriction was resolved by running tests on the host.
- Exact CPU training equivalence with the original trainer at beta2 0.95, exact resume at 0.999, incompatible-beta resume refusal and charged-work restart refusal passed.
- Installed lm-evaluation-harness 0.4.13 agreed on formatting for all 570 ARC development documents and raw/character-normalized metrics for 285 saved control results. Sixteen live CPU likelihood pairs matched through the harness adapter.
- Independent reviewer found two process-lifetime issues. Both were reproduced by failing tests and fixed: reporting failures now reap children; a Linux parent-death signal stops training if its supervisor dies. The test waits for the executed child to signal readiness before killing the supervisor. Reviewer follow-up found no remaining launch blockers.
- GPU verification passed at 23:56 Singapore on September 30: three 32-step checks consumed 1,572,864 tokens. Legacy and new-beta2-0.95 compiled GPU weights and optimizer moments matched exactly. The .999 candidate remained finite with identical source counts.

## Execution and follow-up

The initial detached shell launch (PID 313580) exited before creating a candidate directory or consuming training work. A transient user service, `odin-beta2-20261001.service`, was then started to survive terminal cleanup. It adds a five-hour whole-job limit. Actual training started at 00:08:13 Singapore on October 1, supervised by PID 318315 with training child PID 318483. At 00:11:31 Singapore, the first checkpoint was loaded and verified: 1,000 steps / 16,384,000 tokens, beta2 .999 in recipe and Adam state, correct source-token sum, 49,295,872 parameters and CUDA provenance. Recorded training time was 156.9 seconds with 6.41GiB peak allocated memory. Early loss is not a capability-improvement claim. Estimated training completion is around 02:45–03:10 Singapore, followed by development evaluation; the four-hour hard timeout remains in force.

Live state is [status.json](../results/beta-refinement/status.json). Training metrics will be under `runs/beta-refinement/candidate/metrics.jsonl`; the supervisor log is `results/beta-refinement/runner.log`.

The runner enforces a four-hour training timeout, an October 1 noon Singapore cutoff, an 8GiB disk floor and the shared writer lock. It does not retry interrupted candidate training automatically. Charged partial or replayed work must not exceed 1,000,013,824 tokens.

After successful training it exports the candidate locally, evaluates candidate and control on all 570 already-observed ARC validation questions plus fixed domain windows, generates 32 fresh prompt/seed continuations per model, and writes `comparison.json`, `blind-review.json` and the separate mapping. This is development evaluation, not an official test result or untouched confirmation.

Next review must read all complete anonymous outputs before opening their mapping, record the numeric and quality gates, and then decide whether the authorized conditional continuation/data experiment fits the remaining time. A successful numeric gate alone cannot promote the model. The public release is unchanged.

Commands from this worktree, with the project Python environment:

```bash
/home/ylz/Desktop/odin_llm/project/.venv/bin/python scripts/beta_refinement.py smoke
/home/ylz/Desktop/odin_llm/project/.venv/bin/python scripts/beta_refinement.py run
```

The first command refuses stale or partial verification artifacts. The second refuses an existing candidate directory. If training has completed but evaluation was interrupted, inspect checkpoint and logs, then `scripts/beta_refinement.py evaluate` validates the completed candidate and cached evidence before finishing evaluation. None of these commands publishes a model or starts a continuation automatically.
