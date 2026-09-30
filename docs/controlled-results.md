# Controlled refinement results

The six screening runs completed successfully on September 30, 2026. They do **not establish a meaningful model improvement over the published baseline or the previous 3B-token candidate**. The experiment does provide useful direction: neither tested story mixture improved the selection target at this budget, and a longer warmup produced a tentative comprehension signal that remains unconfirmed.

The runner ended with `completed-no-mixture-winner` at 11:32 Singapore time. It did not run either 1B-token confirmation job. The reserved 285 confirmation questions remain unused. No model was promoted or published during this review.

## Matched screening results

Every row below trained the same 49,295,872-parameter architecture from scratch for 128,008,192 tokens, with the same tokenizer, initialization seed, batch size, and cosine schedule length. The four optimizer runs used the same source draws. All mixtures kept WikiText at 40%; stories replaced FineWeb. Question accuracy is mean-continuation-token-likelihood accuracy on the 285-question ARC-Easy validation screening split, not an official test score.

| Peak learning rate | Warmup steps | FineWeb / WikiText / stories | Correct | Accuracy | FineWeb NLL ↓ | WikiText NLL ↓ | Story NLL ↓ |
|---|---:|---|---:|---:|---:|---:|---:|
| 0.0006 | 200 | 60 / 40 / 0 | 98/285 | 34.39% | 3.7665 | 3.5234 | 3.2989 |
| 0.0006 | 1000 | 60 / 40 / 0 | 107/285 | 37.54% | 3.7679 | 3.5227 | 3.2882 |
| 0.0012 | 200 | 60 / 40 / 0 | 105/285 | 36.84% | 3.7312 | 3.4807 | 3.2429 |
| 0.0012 | 1000 | 60 / 40 / 0 | 101/285 | 35.44% | 3.7450 | 3.4997 | 3.2575 |
| 0.0006 | 1000 | 50 / 40 / 10 | 103/285 | 36.14% | 3.8033 | 3.5323 | 1.7007 |
| 0.0006 | 1000 | 40 / 40 / 20 | 94/285 | 32.98% | 3.8572 | 3.5441 | 1.5500 |

The longer warmup at learning rate 0.0006 answered nine additional questions correctly: +3.16 percentage points. Its paired question-bootstrap interval is −1.40 to +7.72 points. General-domain losses were nearly unchanged. This is a candidate hypothesis, not evidence that warmup was the original quality bottleneck or that 1000 steps is optimal.

Doubling the learning rate with the shorter warmup yielded the lowest general-domain losses, but two fewer correct answers than the selected optimizer. The metrics disagree about the best recipe. Selecting among four noisy scores can exaggerate the apparent winner; a longer run with a fresh seed and untouched questions is needed.

At the selected optimizer, 10% stories lost four correct answers (−1.40 points; interval −6.32 to +3.51). It passed the regression limits but did not beat the control. At 20%, thirteen fewer answers were correct (−4.56 points; interval −9.12 to −0.35), and FineWeb NLL worsened by 0.0893, exceeding the frozen +0.05 limit. Lower story loss therefore did not qualify either mixture for confirmation.

These intervals resample paired questions, not training seeds. They are unadjusted for multiple comparisons and selection, and should be interpreted as exploratory uncertainty estimates. None is an independent confirmation result.

## Comparison with previous models

All rows were measured with the same new evaluation protocol. Training budgets and recipes differ, so this table describes current checkpoint performance and does not isolate the effect of a recipe change.

| Checkpoint | Processed training tokens | Correct | Screening accuracy | FineWeb NLL ↓ | WikiText NLL ↓ |
|---|---:|---:|---:|---:|---:|
| Published baseline | 983,040,000 | 108/285 | 37.89% | 3.4047 | 3.1108 |
| Previous candidate B | 3,000,008,704 | 112/285 | 39.30% | 3.2615 | 3.2250 |
| Selected new screen | 128,008,192 | 107/285 | 37.54% | 3.7679 | 3.5227 |

The selected screen used 13.0% as many tokens as the published model and 4.27% as many as B. Its close screening accuracy is encouraging, but does not establish equivalent capability or better sample efficiency: we lack older-recipe controls at the same budget, selected the new winner using these questions, and measured only one question family.

Relative to the published model, the new screen's question-score difference is −0.35 points with an exploratory paired interval of −4.91 to +4.21. Relative to B it is −1.75 points, interval −7.02 to +3.17. There is no demonstrated accuracy improvement.

The new screen's perplexity is approximately 43.8% higher on FineWeb and 51.0% higher on WikiText than the published model, computed as `exp(new_NLL - old_NLL) - 1` using the shared tokenizer and windows. Its general text prediction remains materially weaker at this short training budget.

## Continuation quality

Generated and inspected all 128 continuations: 16 protected prompts × two seeds × four checkpoints (published, previous B, short warmup, long warmup). Conditions were fixed: temperature 0.8, top-k 40, maximum 128 new tokens, CUDA evaluation. Model labels were concealed until ratings were recorded. Some historical outputs were recognizable from earlier reviews, limiting blinding. The reviewer was Codex, not an independent human.

The selected new screen still changes entities, repeats fragments, and gives false causal explanations. Examples from the saved anonymous items:

- On the friend-on-the-road prompt, it changes to unrelated bus/car/family fragments and calls a friend a family (`item-01-1`).
- For melting ice, it attributes melting to the presence of a water supply and drifts into kitchen plumbing (`item-17-4`).
- For moonlight, it says the Moon is not visible, is well watered, and cycles through incompatible colors (`item-25-1`).

Under the recorded rubric, all 16 scientific/practical continuations from the selected screen had clear causal/factual failures. Both short-trained variants and the published baseline also had pervasive narrative problems in this demanding 128-token continuation set. B produced some more relevant, fluent stories, including one coherent apple-cutting cautionary story, but retained identity reversals and incorrect science. Its smoother surface text does not establish reliable reasoning.

The small, locally authored and previously observed prompt suite is diagnostic. Author-assigned scores are descriptive judgments, not calibrated capability scores. Results do not establish a model's maximum capability under every decoding setting.

## Integrity and compute

- Verified all six exported checkpoint identities, recipes, source fingerprints, source counts, parameter counts, evaluation protocols and complete question coverage. Recomputed both selection decisions and verified the experiment lock.
- Every screen completed 7,813 steps. All six loss curves decreased without a nonfinite failure; peak allocated training memory was approximately 6.41 GiB.
- Retained and charged training both total 768,049,152 tokens: no replay overhead. Recorded training time was 6,987.38 seconds (1.94 hours); training subprocess elapsed time including startup was 7,059.08 seconds (1.96 hours). Preparation, evaluation and smoke tests are additional.
- The unused confirmation budget is 2,000,027,648 tokens. The architecture-experiment prerequisite was not met.

## Direction and next decision

The evidence supports continuing with controlled evaluation and explicit source weighting. It does not support announcing a smarter model, replacing the release, adding more stories by default, or declaring an architecture breakthrough.

The next focused experiment would compare the 200-step and 1000-step warmups at the same longer budget, fresh seed, learning rate 0.0006 and 60/40 no-story mixture. Evaluate the reserved confirmation questions only after both runs finish, and require coherent generation as well as preserved domain losses. This would test the warmup hypothesis without changing another factor. The higher-learning-rate loss result remains another hypothesis, not a settled rejection.

No additional training was launched as part of this evaluation request. The frozen screening plan is preserved unchanged; any follow-up confirmation needs its own recorded protocol rather than retroactively rewriting this experiment.

## Evidence

- [Verified metrics and identities](../results/controlled/review-verification.json)
- [Optimizer decision](../results/controlled/optimizer-decision.json) and [mixture decision](../results/controlled/mixture-decision.json)
- [Compute ledger summary](../results/controlled/compute.json)
- [All rated anonymous continuations](../results/controlled/screen-quality-review/blind-review.json), [model mapping](../results/controlled/screen-quality-review/blind-review-mapping.json), and [descriptive review summary](../results/controlled/screen-quality-review/summary.json)
- [Frozen experiment plan](superpowers/plans/2026-09-30-controlled-refinement.md)
