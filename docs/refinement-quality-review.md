# Refinement quality review — 2026-09-30

Decision: **retain v0.1.0 as the published baseline; keep B as a research candidate.** The completed experiment does not establish an overall improvement sufficient to replace the release. Candidate B is available separately at http://127.0.0.1:8767 while its local server is running.

Verified the final checkpoint SHA256 `dab13b8947f5a7ac4bc5dcd0a9eb0056d69bcad0dcd4edc99f757711fd8811bb`, actual 49,295,872 parameters, 3,000,008,704 trained tokens, corpus/run identity, full official evaluation counts, and complete frozen development report identity. Total recorded training time including losing pilots: 8.857 hours. This excludes preparation, probes, smoke tests, and evaluation.

## Findings

- B (original architecture, 80% educational / 20% public stories) won the short development comparison. C's deeper architecture did not win under this budget; one seed and 250M-token pilots cannot establish a general architectural conclusion.
- Official raw accuracy changes: HellaSwag 27.21→27.58%; ARC-Easy 41.20→43.22%; PIQA 57.94→58.65%; WinoGrande 50.04→48.15%. These are observed score differences, not established statistically significant gains or losses.
- WikiText token perplexity worsened 19.463→21.037 with the same tokenizer and protocol. Web development NLL improved slightly 3.2893→3.2429; story NLL improved substantially 2.8373→1.1782. Stronger story-domain fit did not translate into broad reasoning improvement.
- Frozen eight-item development comparisons fell 6/8→3/8. Original six-item product comparisons remained 3/6. Both sets are too small and locally authored to establish general ability, but they provide no support for a reasoning claim.
- All 32 frozen final continuations and all 6 original product continuations were inspected. Stories often have more conventional sentence structure and story endings, but lose entities and causal consistency. Examples: Nora becomes Anna; a leaking boat is treated as stuck; Sam speaks to another Sam; an umbrella grows to cover a house. Scientific explanations remain incorrect, including heat transfer, photosynthesis, lunar light, braking, and thunder.
- Repeated 4-gram fraction slightly increased 0.0444→0.0510. This metric misses many semantic failures. The selector's domain-loss/repetition gates were insufficient to guarantee coherence or factuality.
- The exact screenshot prompt was rerun on CPU at temperatures 0.6, 0.8, 1.1, seed 42, 96 tokens. All three still show perspective or entity drift. At 0.8, a sunset becomes a sunrise; at 1.1, an unnamed friend turns into a bear/rabbit exchange. Saved verbatim in `results/refinement/screenshot-prompt-review.json`.
- CPU cached generation median 96.86 tokens/s; all 12 cached/uncached paired outputs matched. This is a host-specific efficiency result, not an intelligence measure.

## Next experiment recommendation

**Research update (2026-09-30):** The [subsequent audit](refinement-research-audit.md) found that expected WikiText sampling fell from 41.39% to 9.94%. This confounds the story-percentage recommendation below. Explicit per-source weights, stronger selection, and matched controls take priority; 5–10% stories is only an untested hypothesis.

Do not spend another long run repeating the same recipe. First strengthen development selection to include a larger, independently sourced held-out comprehension set and explicit entity/causal consistency review, with the final official test results reserved for reporting. Existing official test outcomes are already observed and must not become a repeatedly optimized selection target.

Then test a smaller story proportion (e.g. 5% and 10%) against the same expanded educational control under equal budgets, retaining the current architecture initially. This is a hypothesis, not a proven optimal mixture. Data curation should also target fluent explanatory prose and basic causal explanations; any added public corpus requires source/license verification and exclusion of held-out material. The previous pilot suggests data composition is the more useful immediate axis than adding depth, but does not prove depth cannot help.

Use short screens followed by longer confirmation before committing another full budget. Require preservation of held-out general-language/comprehension performance alongside story improvements. Preserve both previous checkpoints for matched comparison. No new long training run was started during this review.
