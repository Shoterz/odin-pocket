# Research audit — 2026-09-30

The strongest evidence concerns experimental design, source balancing, and model selection. This audit has not demonstrated a fatal Transformer implementation bug. The completed B run is useful evidence, but does not establish that the model became a better general language model. Retain the published baseline pending a stronger candidate.

This audit corrects the earlier emphasis on reducing the story percentage: 5–10% stories remains an untested hypothesis. A larger, previously unreported change was the dilution of WikiText within the general-text training stream.

## 1. Source weights changed implicitly

Expected token sampling shares, reconstructed from corpus token counts:

| Source | Published baseline | Candidate B |
|---|---:|---:|
| FineWeb | 58.61% | 70.06% |
| WikiText training split | 41.39% | 9.94% |
| TinyStories | 0% | 20.00% |

FineWeb and WikiText were concatenated and sampled uniformly by token position. Expanding FineWeb changed their internal proportions; then assigning 20% to stories reduced WikiText's overall share further. The nominal “80% web” label concealed this change. At the actual training lengths, expected WikiText exposure fell from approximately 407M to 298M tokens despite the longer total run.

This is a confound, not proof of the cause of WikiText perplexity worsening from 19.463 to 21.037. Restoring the original share is not automatically optimal either. Future mixtures should specify and log each source's token weight, with domain-specific held-out measurements.

Evidence: [measurements](../results/refinement/research-audit-measurements.json), [sampling implementation](../odin/sampling.py), [corpus preparation](../odin/corpus.py). Token counts were recovered using concatenation order and EOS boundaries, checking document counts against manifests. These are expected shares, not realized draw counts; windows crossing a source boundary introduce negligible approximation error.

## 2. The selection objective rewarded story fit more than demonstrated usefulness

| Pilot | General-text NLL ↓ | Story NLL ↓ | Local comparisons correct |
|---|---:|---:|---:|
| A | 3.6758 | 3.1641 | 6/8 |
| B | 3.7501 | 1.5113 | 5/8 |
| C | 3.7735 | 1.5171 | 4/8 |

The selector minimized `0.8 * web_NLL + 0.2 * story_NLL`. Relative to A, B's general-text contribution worsened by 0.0594, while its story contribution improved by 0.3306. It passed the permitted 0.10-nat general-text regression and won. Comparison accuracy was measured but did not affect selection. Repeated four-grams cannot detect identity changes, causal contradictions, or false science.

The arithmetic implements its stated policy; the policy is insufficiently aligned with the intended product. Eight local comparisons are themselves too small to serve as a reliable replacement objective. Build a larger development set and manually assess blinded continuations for entity consistency, causal consistency, and relevance; keep these criteria separate from domain losses.

OpenAI recommends task-specific evaluations and human calibration, and identifies relying solely on perplexity as an evaluation anti-pattern. This is applicable evaluation guidance, not an OpenAI recipe for training a 50M model. [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)

Evidence: [selector](../odin/development.py), [A](../results/refinement/A-development.json), [B](../results/refinement/B-development.json), [C](../results/refinement/C-development.json), [quality review](refinement-quality-review.md).

## 3. We lack a full-length matched control

Each pilot saw 250M tokens, only 8.33% of the final 3B-token budget. Only B continued. Comparing B with the published model changes training length, corpus size, source proportions, and training schedule together. Comparing pilot C with pilot B holds the data fixed but uses only one seed and an early checkpoint. Neither comparison establishes an architecture ranking at the final budget.

SmolLM2 used controlled data ablations and reran smaller-model data experiments at the target training length. Its experiments also found different strengths for FineWeb-Edu and DCLM. These support matched controls and testing data diversity; their exact ratios and token budgets are not prescriptions for ODIN. [SmolLM2, sections 3 and 6](https://arxiv.org/html/2502.02737v1)

Before declaring a mixture winner, compare it with a control at a common token budget and comparable schedule. Record wall-clock cost as well. Short screens can reject clearly poor candidates but cannot prove the ultimate winner.

## 4. Optimizer choices were not tuned

Peak learning rate was 0.0006 with 16,384 tokens per optimizer step. Warmup uses `max(1, min(200, int(steps * .02)))`: the final run warmed up for 200 steps, 3,276,800 tokens, or 0.109% of training. It was not a 2% warmup. No experiment establishes whether that cap, the learning rate, or the batch size was appropriate.

The loss declined smoothly; there is no demonstrated divergence that proves these settings caused the weak output. A small, controlled learning-rate/warmup screen is more defensible than declaring a new setting correct. DeepSeek LLM explicitly investigated learning rate and batch size jointly; its fitted scaling laws should not be transferred blindly across data and model regimes. [DeepSeek LLM, section 3.1](https://arxiv.org/html/2401.02954v1)

Evidence: [training schedule](../odin/train.py), [measurement calculations](../results/refinement/research-audit-measurements.json).

## 5. Architecture ideas require evidence at our scale

- **Multi-token prediction:** The original paper reports worse small-scale code results in its 300M–13B sweep, but improved induction and synthetic arithmetic at some much smaller sizes. Thus it is a plausible targeted ablation, not a guaranteed upgrade. Count auxiliary trainable heads within the 50M budget and keep next-token evaluation loss separate. [Original MTP paper, sections 3–4](https://arxiv.org/html/2404.19737v1)
- **DeepSeek-V3:** Its “small” MTP ablation has 15.7B total parameters, far beyond ODIN. Evidence at that scale does not establish a benefit here. [DeepSeek-V3, section 4.5.1](https://arxiv.org/html/2412.19437v1)
- **DeepSeek-R1:** R1-Zero starts from DeepSeek-V3-Base, not random weights. Reinforcement learning does not replace establishing a useful pretrained base. [DeepSeek-R1](https://arxiv.org/html/2501.12948v1)
- **TinyStories:** It demonstrates coherent language in a restricted domain at very small sizes. Story fluency does not establish broad factual or commonsense capability. [TinyStories](https://arxiv.org/abs/2305.07759)

The competition counts total trainable parameters and requires training from scratch; pretrained distilled models are not an eligible shortcut. Public TinyStories use is explicitly allowed. [Official rules](https://gibc-v2.devpost.com/rules)

Our short failing continuations fit within the 512-token context. Their errors do not establish that extending context would fix them. Likewise, the screenshot is a valid continuation prompt: absence of instruction tuning does not excuse incoherent continuation.

## 6. What Reddit contributed

A firsthand 8GB training post initially overstated its model size and reported a loss-derived perplexity that included an auxiliary objective. The author corrected both in discussion and clarified that it was a brief training demonstration. Its MTP success claim is an anecdote, not controlled evidence for ODIN. It usefully illustrates why parameter accounting, training budget, and comparable loss definitions matter. [Consumer-GPU training discussion](https://www.reddit.com/r/LocalLLaMA/comments/1treb2z/me_train_llm_on_8gb_from_scratch_me_happy/)

Another author reported failed runs involving repeated streaming data after resume and excessive learning rate. These are useful failure modes to check, not diagnoses of our run. ODIN's recorded exact-resume verification and smoothly declining loss do not show those failures. [Training failure discussion](https://www.reddit.com/r/LocalLLaMA/comments/1m52h10/i_posted_3_weeks_ago_about_training_my_own_model/)

## Recommended order of work

1. Freeze a larger development evaluation aligned with useful continuations and comprehension. Existing official test scores are already observed; avoid repeatedly selecting against them.
2. Make FineWeb, WikiText, and story weights explicit. Audit source samples and near-duplicates before merely downloading more text. Existing local filtering covers length, exact document deduplication, and held-out n-gram exclusions; it does not establish comprehensive near-duplicate or prose-quality control.
3. Design one-variable mixture comparisons and a bounded optimizer screen. Keep tokenizer, architecture, evaluation, token budgets, and schedule treatment comparable. Account for seeds when interpreting close results.
4. Confirm any apparent winner against a longer matched control. Require broad-language preservation and improved blinded generation quality, not just lower story loss.
5. Only then spend a separate, capped experiment on MTP or architectural changes. A positive result must survive total-parameter accounting and the same evaluations.

These are proposed experiments, not promises of improvement. The audit did not launch training or modify frozen training code. The baseline remains the published release.
