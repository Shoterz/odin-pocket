# Research results and final checkpoint selection

The submitted model remains **v0.1.0**, SHA-256 `4634f90b2120a7a128a0a4bbd59ae70056dbeff7baee2a35bf551fe0da90ab7e`. Its weights, demo and complete benchmark report agree. Subsequent experiments did not establish an overall improvement sufficient to replace it. This is a documented selection decision, not a claim that the baseline beats every candidate on every metric.

## What we tried

| Experiment | Measured outcome | Decision |
| --- | --- | --- |
| Three 250M-token pilots, followed by extending candidate B to 3B tokens | B fit story text better and improved some benchmark point estimates, but WikiText perplexity worsened from 19.463 to 21.037 and WinoGrande fell from 50.04% to 48.15%. Training budget and data proportions differed from the release, so this was not a clean causal comparison. | Retain B as a research result. |
| Six matched 128M-token optimizer/data screens | Tested learning rates 0.0006/0.0012, warmup 200/1000, and 0/10/20% public TinyStories with explicit source weights. Neither story candidate met its selection gate. | No mixture winner. |
| Two matched 1B-token warmup runs | The preregistered token-mean development metric did not establish a win. A later raw-score diagnostic was promising but post hoc; it motivated scoring verification rather than a capability claim. | Keep the release. |
| Matched 1B-token AdamW beta2 0.95 versus 0.999 | Fixed-window perplexity improved 5.7–6.7% across three domains. Raw ARC development accuracy changed from 243/570 to 244/570; its paired interval included zero. Neither model produced a fully coherent continuation in the 32-output author review. | Useful optimization result; quality gate failed. |

Research code, frozen protocols and complete original evidence are preserved at [research commit 043b890](https://github.com/Shoterz/odin-pocket/tree/043b890), separate from the stable submitted runtime. Selected reports are also included in [the submission evidence](../submission/evidence/research/).

The beta2 review was blind to which model produced each passage; numerical results were already known. It is a project-author assessment of 16 locally authored prompts with two sampling seeds, not independent human validation or a competition score. Its ARC questions are observed validation data, not the official test split. These numbers must not replace the submitted model's README benchmark table.

## Findings that changed our process

Expanding FineWeb while sampling uniformly from a concatenated stream reduced the expected WikiText share from 41.39% in the baseline to 9.94% in candidate B. That confounded the original story-mixture comparison. Later experiments specified and recorded each source's sampling weight explicitly.

Our early selector rewarded story loss improvements while allowing general-text regression. Fluent openings and fewer repeated n-grams did not establish entity or causal consistency. Later reviews therefore recorded relevance, entity consistency and factual/causal consistency separately.

An optimizer improvement can lower held-out prediction loss without producing a detectable reasoning improvement at the tested budget. We preserve the positive loss result and the negative capability result together. None of these single-seed experiments proves the best possible 50M-parameter design.

## Compute disclosure

The submitted checkpoint's recorded training loop took **3.923 hours** on one RTX 4070 12 GB and processed **983,040,000 tokens**. Its approximate compute is **2.908 × 10^17 FLOPs**, using `6 × parameters × processed tokens`.

The broader project used additional experiments. The available completed-run summaries total **22.355 recorded training hours**, including the submitted run, pilot, research runs and verification runs. Approximately **18.432 hours** of that total are additional to the submitted run. Their checkpoint token counters sum to **8,254,717,952**, with a corresponding approximate dense-training compute sum of **2.441 × 10^18 FLOPs**. These are processed-token counters, not distinct text.

This is an inventory of available recorded training loops, not a precise bill for all GPU occupancy or project time. Preparation, evaluations, inference, compilation outside recorded loops, and partial work absent from summaries are excluded. Resumed-run summaries retain their cumulative checkpoint counters. Where append-only work ledgers exist, their charged-token counts are retained separately. No paid cloud compute was purchased.

The [compute inventory](../submission/evidence/research/compute-inventory.json) lists every included summary, its hash, run ID, parameter count, time and token count. Copies of the [training summaries](../submission/evidence/research/training-summaries/) are included for inspection.

## Evidence shortcuts

- [Candidate B complete benchmark report](../submission/evidence/research/refinement-final-official.json)
- [Source-mixture audit](../submission/evidence/research/refinement-research-audit-measurements.json)
- [Controlled mixture decision](../submission/evidence/research/controlled-mixture-decision.json)
- [Warmup primary comparison](../submission/evidence/research/warmup-confirmation-primary-comparison.json)
- [Beta2 numerical comparison](../submission/evidence/research/beta2-comparison.json)
- [All beta2 passages and author ratings](../submission/evidence/research/beta2-quality-review.json)
- [Beta2 decision and uncertainty intervals](../submission/evidence/research/beta2-review-decision.json)
