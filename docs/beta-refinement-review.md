# Beta2 refinement evaluation — October 1, 2026

Changing AdamW beta2 from 0.95 to 0.999 improved held-out text prediction at the same training budget. It did not demonstrate a meaningful reasoning or generation-quality improvement. The numeric gate passed, but the quality gate failed. Do not start the conditional continuation or promote this checkpoint on these results.

## Matched comparison

Both runs used 49,295,872 parameters, the same tokenizer and architecture, seed 20261001, 1,000 warmup steps, learning rate 0.0006 and 1,000,013,824 training tokens. Realized source counts matched exactly: 599,670,272 FineWeb-Edu tokens and 400,343,552 WikiText tokens. The control is `runs/controlled/warmup-confirm-1000/submission.pt`; the candidate is `runs/beta-refinement/candidate/submission.pt`. This comparison is against that matched control, not the differently trained public release.

| Measure | Control: beta2 0.95 | Candidate: beta2 0.999 | Interpretation |
| --- | ---: | ---: | --- |
| FineWeb perplexity | 29.086 | 27.314 | 6.09% lower |
| WikiText perplexity | 22.166 | 20.896 | 5.73% lower |
| Stories perplexity | 17.046 | 15.910 | 6.67% lower |
| ARC raw accuracy | 243/570, 42.63% | 244/570, 42.81% | One additional correct answer |
| ARC character-normalized accuracy | 191/570, 33.51% | 195/570, 34.21% | Four additional correct answers |
| Fully coherent continuations, author review | 0/32 | 0/32 | Quality gate failed |

Lower perplexity means better prediction of held-out text under this tokenizer. Each domain uses 131,072 fixed evaluation tokens. The consistent reduction across all three domains is useful evidence for this optimizer setting, but these domain results have no estimated uncertainty interval and only one training seed.

ARC uses 570 already-observed validation questions, not the official test set or an untouched confirmation set. The paired 95% bootstrap interval for raw accuracy improvement is **−3.16 to +3.51 percentage points**, containing zero. The character-normalized interval is **−2.46 to +3.86 points**. These intervals cover question sampling only, not variability across training seeds. The candidate fixes 48 questions the control missed and loses 47 the control got right. The practical numeric gate required any positive raw improvement; passing that permissive condition does not establish a reasoning gain.

## Full continuation review

All 64 complete anonymous continuations were read. Ratings and explanations were saved before opening the passage-to-model mapping. Numerical results were already known. This is a project-author assessment, not independent human validation. There are 16 locally authored prompts with two sampling seeds per model, temperature 0.8, top-k 40 and up to 128 new tokens. Paired seeds and related prompts do not provide 32 independent trials.

Each applicable dimension is scored 0 for clear failure, 1 for mixed/unclear and 2 for coherent and correct. Entity consistency applies to the 16 narrative continuations per model. Full success requires all applicable dimensions to score 2. Causal/factual score 2 alone also yields zero successes for both models, so the failed gate does not depend on adding the other dimensions. Token-budget truncation alone is not a failure; neither is an invented name in fiction or established fantasy.

| Dimension | Control mean / clear failures | Candidate mean / clear failures |
| --- | ---: | ---: |
| Relevance, 32 outputs | 0.5625 / 14 | 0.65625 / 11 |
| Entity consistency, 16 narratives | 0.1875 / 13 | 0.1875 / 13 |
| Causal/factual consistency, 32 outputs | 0.15625 / 27 | 0.09375 / 29 |

Both models produce plausible opening clauses followed by identity drift, repetition, disconnected events and physical errors. Examples include losing the two-apple count, confusing hot/cold outcomes for a spoon, or substituting unrelated advice for a simple procedure. These are whole-passage failures; a locally fluent sentence does not establish a coherent continuation. The small subjective sample does not establish that the candidate is generally worse, but it clearly provides no evidence for the required four additional fully coherent outputs. No clear new systematic failure category was identified; existing failures remain widespread.

## Compute and verification

Candidate training completed in 9,042.69 seconds, approximately 2 hours 31 minutes, on the RTX 4070. Peak PyTorch allocated GPU memory was 6.41 GiB; this is not total device memory use. The control took 9,115.52 seconds; this small timing difference is not evidence of a reliable speed improvement.

The candidate consumed exactly its 1,000,013,824-token charged budget, without replay. The separately recorded verification runs consumed 1,572,864 tokens, bringing new candidate-plus-verification work to 1,001,586,688 tokens. This excludes the historical control and evaluation inference.

Checkpoint identities, candidate latest/export equality, optimizer state, frozen source/data identities, token counts and recomputed numerical comparison were checked. The review additionally rechecked source/plan hashes, checkpoint/report bindings, summaries, charged ledgers and all 64 mapping-to-generation matches. Model and trainer code were not changed during this assessment.

## Decision and next work

Preserve beta2 0.999 as a promising optimization result for future controlled experiments. Do not describe it as a reasoning breakthrough, and do not automatically replace the public checkpoint. The pre-agreed quality condition for further training was not met.

At this review, approximately 10:10 Singapore on October 1, another run of the observed duration would exceed our internal noon experimentation cutoff. Prioritize submission packaging, a reproducible benchmark report, the parameter/provenance evidence and a candid demonstration of the model's capabilities and limits. The organizer's deadline is October 1 at 23:45 UTC+8 according to the [official rules](https://gibc-v2.devpost.com/rules), previously verified during deadline planning. Noon is our internal cutoff, not the organizer's deadline.

This work has produced a real scratch-trained small model, a functioning local demo and a controlled experiment with inspectable evidence. Its present results support an engineering/research submission; they do not justify claiming reliable general reasoning or reliable long-form generation, and they cannot predict placement against unknown competitors.

## Evidence

- [Numerical comparison](../results/beta-refinement/comparison.json)
- [All anonymous passages and saved ratings](../results/beta-refinement/quality-review.json)
- [Quality aggregates](../results/beta-refinement/quality-summary.json)
- [Final decision and paired counts](../results/beta-refinement/review-decision.json)
- [Checkpoint and compute verification](../results/beta-refinement/review-checkpoint-verification.json)
- [Frozen experiment plan](superpowers/plans/2026-09-30-beta2-refinement.md)
