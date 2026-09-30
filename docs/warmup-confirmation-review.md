# Warmup confirmation: final assessment

Completed September 30, 2026. Neither new checkpoint warrants replacing the published model. Longer warmup did not establish a gain on the preregistered primary measure, and the generated text still fails basic coherence and factual checks. There is modest improvement in text prediction relative to the published model and an exploratory signal under an alternative answer-scoring method. These are narrower findings than a useful capability improvement.

Both new models have 49,295,872 parameters and trained from scratch for 1,000,013,824 tokens. They share seed 20261001, architecture, tokenizer, learning rate, data sequence and actual source counts: 599,670,272 FineWeb tokens and 400,343,552 WikiText tokens. Only warmup differs: 200 versus 1,000 steps. Checkpoint states, frozen source/data identities, token ledgers and all four evaluation report identities were verified. Both trainings completed before confirmation evaluation.

## Matched evaluation

All four models below were scored on the same 285 reserved ARC-Easy validation questions. The primary measure averages continuation-token log likelihood for each answer. These are custom validation results, not official benchmark test scores. Earlier screening percentages used different questions and must not be compared directly with this table.

| Model | Training tokens | Correct | Primary accuracy | FineWeb NLL ↓ | WikiText NLL ↓ | Story NLL ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Published | 983,040,000 | 77/285 | 27.02% | 3.4047 | 3.1108 | 2.8657 |
| New short warmup | 1,000,013,824 | 78/285 | 27.37% | 3.3645 | 3.0921 | 2.8188 |
| New long warmup | 1,000,013,824 | 81/285 | 28.42% | 3.3703 | 3.0986 | 2.8359 |
| Previous B | 3,000,008,704 | 89/285 | 31.23% | 3.2615 | 3.2250 | 1.2119 |

Long minus short: **+1.05 percentage points**, paired 95% question-bootstrap interval **−2.81 to +4.91 points**. The domain-loss regression guard passed, but the primary improvement gate failed. Long warmup had slightly higher loss in all three domains. This does not prove equivalence or harm; the experiment did not resolve a small benefit. Only one fresh training seed was tested, and the interval covers question sampling, not training-seed variation.

Long warmup versus published: +1.40 points, interval −2.46 to +5.26. Versus previous B: −2.81 points, interval −6.67 to +1.05. Historical comparisons are descriptive because training recipes, seeds and budgets differ. Previous B remains the numerical leader on these questions, but its accuracy advantage is not established by these intervals either.

Compared with published, short warmup reduces FineWeb perplexity by 3.94% and WikiText perplexity by 1.86%; long warmup reduces them by 3.39% and 1.22%. These are modest predictive improvements on fixed evaluation windows, without uncertainty estimates; they do not establish better reasoning or attribute the improvement to warmup.

## Generated-text assessment

The fixed suite contains 16 prompts with two seeds per model, temperature 0.8, top-k 40, up to 128 generated tokens. All 64 new anonymous outputs were read in full and rated before opening their model mapping. The 64 historical outputs exactly matched prior prompt/text pairs, so their existing ratings were reused. All 128 rated outputs were checked against the evaluation reports and mapping hashes.

Reviewer: **OpenAI Codex, author assessment, not independent human validation**. Historical text was recognizable; blinding is limited. Prompts were used previously in development. Scores are descriptive rubric judgments, not calibrated measurements of intelligence. Entity consistency is scored only on narrative prompts, as in the prior review. Each full continuation is assessed: an initially plausible clause can be followed by a clear failure.

| Model | Relevance mean /2 | Narrative entity mean /2 | Causal/factual clear failures | Science/practical clear failures |
|---|---:|---:|---:|---:|
| Published | 0.53 | 0.25 | 32/32 | 16/16 |
| New short warmup | 0.66 | 0.38 | 32/32 | 16/16 |
| New long warmup | 0.59 | 0.31 | 32/32 | 16/16 |
| Previous B | 1.13 | 0.56 | 22/32 | 15/16 |

Both new models tie at zero fully coherent/correct continuations on the causal/factual dimension. Their small differences in relevance/entity averages are not a convincing practical gain. Examples preserved verbatim in the rated artifact include boats entering themselves, repeated sun rises, melting attributed to a cooler climate, the Moon's brightness attributed to Earth's magnetic field, and freezing water turning into liquid. The recurring problem is unreliable continuation of the premise, even when individual sentences sound plausible. Previous B has better surface relevance but still substantial errors.

## Alternative scoring: exploratory only

Summing answer-token log likelihood instead of averaging it gives different results:

| Model | Raw-sum correct | Accuracy |
|---|---:|---:|
| Published | 100/285 | 35.09% |
| New short warmup | 100/285 | 35.09% |
| New long warmup | 117/285 | 41.05% |
| Previous B | 120/285 | 42.11% |

Long minus short is +5.96 points, with a nominal paired 95% question-bootstrap interval of +0.70 to +11.23. This is a real diagnostic computed from the saved raw scores, but it was examined after the primary results, has no multiplicity correction and uses the same questions. It cannot replace the preregistered endpoint or be treated as independent confirmation. Summing and averaging introduce different answer-length effects. This sensitivity warrants auditing answer scoring and a frozen standard harness protocol before spending more training compute. It does not establish that the primary implementation is defective.

## Decision and next step

Retain the published release. Preserve both new checkpoints and all results; do not claim a warmup win, launch another long training run, or change the public model based on this experiment. The longer-warmup hypothesis remains unconfirmed on the chosen primary measure. The current generated-text evidence does not support calling either candidate a competition-winning product.

The next useful work is a bounded scoring audit using saved outputs: inspect where raw-sum and token-mean rankings disagree, examine answer-length effects, and verify the intended benchmark protocol against the pinned harness. Freeze that protocol before any next training selection. Follow with a small, controlled data-quality experiment targeted at coherent explanations or a specific useful domain, with fresh held-out material and an explicit practical success criterion. This is a hypothesis to test, not a demonstrated remedy. The current evidence does not justify blaming or redesigning the architecture yet. This confirmation partition is now observed and must not be presented as untouched confirmation for subsequent choices.

New training consumed **2,000,027,648 tokens** and **5.082 subprocess hours** on the RTX 4070. Including the six earlier screens, this controlled campaign consumed 2,768,076,800 tokens; this subtotal excludes the original published and previous-B training. No charged replay or partial-step overhead was found. Preparation, evaluation and review time are additional. No new training was launched during this assessment.

Evidence: [numerical report](warmup-confirmation-results.md), [primary comparison](../results/warmup-confirmation/primary-comparison.json), [rated continuations](../results/warmup-confirmation/quality-review.json), [quality aggregates](../results/warmup-confirmation/quality-summary.json), [scoring diagnostic](../results/warmup-confirmation/posthoc-scoring-diagnostic.json), [checkpoint verification](../results/warmup-confirmation/review-checkpoint-verification.json), [compute accounting](../results/warmup-confirmation/compute.json). The separate final review decision supersedes the automatic report's pre-review pending state; automatic numerical artifacts remain preserved.
