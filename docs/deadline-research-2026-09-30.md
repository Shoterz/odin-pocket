# ODIN research for the final training decision

Prepared September 30, 2026 for Yong Li Zhong, Axiom AI. Recommendation: first align development scoring with the installed benchmark harness, then test AdamW beta2 0.999 against the existing 0.95 control with one otherwise identical 1B-token run. Allow 2.6–3 GPU hours. If the result warrants further work, spend the remaining experimental budget on a bounded continuation or a matched data experiment. Preserve time to submit the existing product regardless of the outcome.

This is a research recommendation, not a launched run, a completed preregistration, or a prediction of winning. It incorporates Moonshot/Kimi, Cursor/Composer, Z.ai/GLM, Alibaba/Qwen, Meta/Llama, Mistral, Google, Anthropic, DeepSeek, Hugging Face and independent research. Relevant methods, ablations and limitations were read; this is not a claim to have exhaustively reviewed every paper or every lab's newest model. Recommendations below are our inferences unless explicitly attributed.

## Deadline and usable compute

The official deadline is **October 1, 2026 at 23:45 UTC+8**, the same clock time in Singapore. At 18:40 Singapore on September 30, approximately 29 hours remained. The rules cap **total trainable parameters, including embeddings and output**, at 50M and prohibit external pretrained initialization and larger-model distillation. Public training datasets are permitted. Required evaluation covers HellaSwag, ARC-Easy, PIQA, WinoGrande and held-out WikiText-103 perplexity. [Official rules](https://gibc-v2.devpost.com/rules)

Our two recent 1B-token runs each took about 2.54 training hours. Their recorded peak allocated GPU memory was 6.41 GiB; that is not a measurement of total device occupancy or reserved memory. Memory is not presently preventing another run on the RTX 4070 12GB.

| Additional tokens with the current implementation | Measured-rate projection | Planning allowance |
|---|---:|---:|
| 1B | 2.54 hours | 2.6–3 hours |
| 2B | 5.07 hours | 5.5–6 hours |
| 3B | 7.61 hours | 8–9 hours |
| 4B | 10.15 hours | 11–12 hours |

These projections exclude preparation, implementation, evaluation, review and packaging. They do not apply automatically to Muon, a deeper model, longer context, or extra training objectives. Proposed internal cutoff: end GPU experimentation by **October 1 at noon Singapore**, and target submission by **18:00**, leaving 5h45 before the official deadline. Registration and the hosted video upload remain unverified; a public repository alone does not establish a completed Devpost entry.

## What the local evidence actually says

ODIN has 49,295,872 parameters, 12 layers, width 512, 8 attention heads, feed-forward width 1536, a 16,384-token vocabulary and context 512. It already uses RoPE, RMSNorm, SwiGLU, tied embeddings, causal SDPA, compiled training and BF16 autocast with FP32 parameters/optimizer states. Tied embeddings consume 8,388,608 parameters, about 17% of the total. Many commonly recommended architectural improvements are already present.

The latest effective batch is **32 × 512 × 1 = 16,384 tokens per update**. AdamW has beta1 0.9, beta2 0.95, epsilon 1e-8, matrix weight decay 0.1 and peak learning rate 0.0006. Earlier screens varied learning rate and warmup but never beta2. Gradient accumulation is already one.

We already use **FineWeb-Edu**. Our stored training-token inventory is 778.6M FineWeb-Edu, 110.4M WikiText and 576.9M TinyStories. The latest 60/40 FineWeb/WikiText run consumed 0.77 and 3.62 corpus-size exposure equivalents respectively. Sampling is with replacement, so these are not literal complete epochs. A 3B run with the same mixture would expose WikiText roughly 10.9 times. This is a reason to examine the mixture and loss curves, not evidence that repetition caused our failures.

### Scoring correction

The custom screening endpoint divides answer log likelihood by continuation **token count**. Installed lm-evaluation-harness 0.4.13 reports raw summed likelihood as `acc` and normalizes by answer **character count** for `acc_norm`. Inspection of `lm_eval/api/task.py` and `tasks/arc/arc_easy.yaml` confirmed this distinction. The rules name the harness but do not specify which ARC metric determines the organizer's aggregate; report both.

Reconstruction from saved likelihoods, on the same 285 previously observed validation questions:

| Model | Custom token mean | Raw sum corresponding to acc | Character normalization corresponding to acc_norm |
|---|---:|---:|---:|
| Published | 77 | 100 | 79 |
| Previous B | 89 | 120 | 103 |
| Short warmup | 78 | 100 | 84 |
| Long warmup | 81 | 117 | 87 |

These are correct-answer counts, not fresh official test results. Long warmup's raw score is 41.05%, versus 35.09% published and 42.11% previous B. This is more encouraging for benchmark performance than the custom endpoint suggested. It does not overturn the failed preregistered primary endpoint, make these questions untouched, or fix the generated passages. The raw long-minus-short interval was exploratory and uncorrected for multiple comparisons. Our new character reconstruction removes exactly the separator space added by the custom data builder before counting characters. An actual harness run remains necessary before reporting official numbers.

I should have aligned screening with the benchmark harness earlier. This was a design mismatch, not evidence that the saved likelihood computations were wrong. Future selection must freeze the metrics before viewing new results and keep final test reporting separate from iterative tuning. See [previous assessment](warmup-confirmation-review.md) for the unchanged quality findings.

## The most actionable optimizer evidence

[Small Batch Size Training for Language Models](https://arxiv.org/html/2507.07101v2), NeurIPS 2025, is unusually close to our setting: its model called “30M” has approximately **49M total trainable parameters**, trains on 600M FineWeb-Edu tokens, and uses context 512. Sections 4.3 and Appendix A show that fixed beta2 values around 0.95–0.98 can underperform at small batches; a token-based second-moment half-life improves transfer across batch sizes. Architecture and precision differ from ODIN, so this is relevant evidence, not a reproduction.

Local calculation: `half_life_tokens = 16384 * log(0.5) / log(beta2)`.

| beta2 | Half-life in observed tokens |
|---|---:|
| 0.95, current | 221,404 |
| 0.99 | 1,129,965 |
| 0.999, proposed | 11,350,844 |

Exactly 10M tokens gives beta2 0.9988649923. Testing 0.999 is a practical approximation. This parameter controls averaging of squared gradients in the optimizer; it does not extend the model's context or factual memory. We have not established that it caused our poor output.

[Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/pdf/2509.02046) compares eleven optimizers with substantial tuning. Its 130M-model example improves AdamW by changing the learning rate, and matrix methods' advantages depend on scale and training duration. That supports checking our baseline settings before replacing the optimizer. Its 0.008 learning rate is not a transferable prescription for our much smaller batch; a blind jump from 0.0006 would be unjustified.

## What transfers from the requested labs

| Source and inspected material | Evidence and scale | Decision for ODIN |
|---|---|---|
| Moonshot, [Muon is Scalable](https://arxiv.org/html/2502.16982v1), §§2–3, Appendix B | RMS-scaled Muon; scaling experiments start at 399M non-embedding parameters, with much larger batches. Reported compute savings concern their fitted regime. | Strong alternative optimizer candidate, but benchmark local wall time and tune the Adam baseline first. Their Appendix C also reports no significant MTP benefit. |
| Moonshot, [Kimi K2](https://arxiv.org/html/2507.20534v1), §2 | MuonClip controls attention-logit instability; verified rephrasing improves knowledge-token use. Main run: approximately 1T total parameters and 15.5T tokens. | Monitor attention/gradient stability if trying Muon. Do not add QK clipping without evidence of the relevant instability. Private teacher-generated training is not our deadline plan under the distillation restriction. |
| Z.ai, [GLM-4.5](https://arxiv.org/html/2508.06471v1), §§2.1–2.4 | Deeper/narrower design, Muon, quality buckets, semantic deduplication and staged data. Batch size grows from 16M to 64M tokens. Cosine outperformed WSD in their early experiments. | Supports optimizer and data work, not copying frontier hyperparameters. GLM's table excludes embeddings/output from its stated counts; our competition count must include them. |
| Alibaba, [Qwen3](https://arxiv.org/html/2505.09388v1), §3 | Instance-level data labels and proxy-model mixture tests; a later higher-quality reasoning stage with faster LR decay. Models begin at 0.6B and train on 36T tokens. | Borrow source labeling, small ablations and a bounded quality-focused later stage. Their synthetic teacher pipeline and multilingual scale are not a ready 50M recipe. |
| Meta, [Llama 3](https://arxiv.org/html/2407.21783v1), §§3.1–3.2 | Authors attribute gains primarily to data quality/diversity and scale. Dataset tests anneal an 8B checkpoint on a 30% candidate/70% existing mixture for 40B tokens. | Useful experimental pattern: compare a new-source branch with an equal-budget old-source branch. The percentages are an experiment design, not proven ODIN optima. |
| Meta, [MobileLLM](https://arxiv.org/html/2402.14905v1), architecture ablations | Deep/narrow 125M/350M models, tied embeddings, GQA and layer sharing; training budgets reach 1T tokens. | A deeper 48–49M design is plausible later. We already tie embeddings. Sharing parameters does not eliminate repeated-layer training compute. |
| Cursor, [Composer 2 report](https://cursor.com/resources/Composer2.pdf), §§3, 5–6 | Starts from Kimi K2.5, uses coding continued pretraining then RL. A Qwen-based ablation relates base-model loss to downstream RL performance. | Borrow target-domain evaluation and the importance of a capable base. The pretrained initialization, infrastructure and agentic RL are outside this attempt. |
| Mistral, [Ministral 3](https://arxiv.org/html/2601.08584v1), §3.1 | Cascade distillation repeatedly prunes and distills from Mistral Small 3.1 to produce 14B/8B/3B models. | Their central efficiency shortcut conflicts with our scratch-training/distillation constraints. It is not evidence that a scratch 50M model can acquire the same capabilities overnight. |
| Google, [Gemma 3](https://arxiv.org/html/2503.19786v1), §2 | All reported sizes use teacher distillation; even 1B trains on 2T tokens. Local/global attention targets long-context KV memory. | Data reweighting and evaluation decontamination transfer. Teacher logits and long-context cache engineering do not solve our immediate problem. |
| Anthropic, [Learning from Repeated Data](https://arxiv.org/html/2205.10487v1), §§1–3 | Controlled 100B-token experiments show repetition can harm generalization non-monotonically and damage copying behavior. | Track per-source exposure and copying/entity consistency. Our 3.62× WikiText exposure does not by itself reproduce their harmful regime. |
| DeepSeek, [V3](https://arxiv.org/html/2412.19437v1) and [R1](https://arxiv.org/html/2501.12948v1), training sections | MTP evidence includes a 15.7B-total ablation; reasoning RL starts from an already capable base. | No evidence here for rescuing ODIN with RL or extra prediction heads before the deadline. All additional trainable heads would count toward 50M. |

The common theme is not a secret architecture. These reports combine data engineering, optimizer tuning, long training and evaluation aligned to the intended capability. Only some of those components fit our remaining time and rules.

## Smaller models and contrary evidence

| Primary source | Finding relevant to our decision |
|---|---|
| [TinyStories](https://arxiv.org/html/2305.07759v2), small-model and depth experiments | Restricted vocabulary and coherent stories make coherent generation possible at tiny scales. That does not establish broad factual reasoning; previous B already shows the limitation of improved story-like surface text. |
| [SmolLM2](https://arxiv.org/html/2502.02737v1), §6 | The 135M/360M models benefit from consistently high-quality mixtures and train on 2T/4T tokens. Their success is not evidence that 1B tokens is enough for comparable breadth. |
| [SmolLM author report](https://huggingface.co/blog/smollm), training section | Deep/narrow 135M/360M models trained on 600B tokens. Increasing curated Cosmopedia and introducing instruction data late produced no significant gain in their experiment. This directly tempers a proposed textbook-data rescue. |
| [FineWeb](https://arxiv.org/html/2406.17557v2), filtering and deduplication ablations | Educational filtering helps in their larger-model regime, but some additional filtering/deduplication choices hurt. We already have FineWeb-Edu; “clean the data more” is not a sufficient prescription. |
| [DCLM](https://arxiv.org/html/2406.11794v1), filtering experiments | Strong data selection includes educational/explanatory examples, with the smallest benchmark scale still 412M parameters. Another public source offers diversity, but preparation and measured transfer matter. |
| [Scaling Data-Constrained Language Models](https://arxiv.org/html/2305.16264v5), repeated-data experiments | Moderate repetition can remain useful; returns diminish with repeated exposure. This qualifies Anthropic's result: repetition is not universally bad, nor is four epochs a universal safety threshold. |
| [Simple and Scalable Continual Pretraining](https://arxiv.org/html/2403.08763v4) and [Reuse, Don't Retrain](https://arxiv.org/html/2407.07263v1) | The former supports rewarming/redecaying and replay; the latter finds no rewarm better in its setup. Continuing our own model is plausible, but no universal LR restart is established. |
| [Practical Efficiency of Muon](https://arxiv.org/html/2505.02222v1), efficiency experiments | Includes 100M models, but uses large batches and TPU infrastructure. Newton–Schulz overhead must be measured on our 16K-token updates before claiming GPU-hour savings. |
| [Multi-token Prediction](https://arxiv.org/html/2404.19737v1), scale and natural-language ablations | Gains depend on task and size; small code models can worsen, and additional natural-language prediction heads do not consistently improve multiple-choice performance. Too uncertain for the first deadline experiment. |
| [Recursive Transformers under Limited Data](https://arxiv.org/html/2608.26973v1), 10M-word experiments | A 27.6M recursive model improves an aggregate over a 28.7M standard comparison, but uses about 14.6× its training compute. Parameter savings alone are not a solution to our time budget. |
| [Improved Regularization under Data Constraints](https://arxiv.org/html/2606.06888v1), scale ablations | Additional masked-input regularization is promising mainly at larger scales; the 72M result is modest and an extra training pass costs time. Lower priority than a single optimizer setting. |

## Recommended next sequence

1. **Freeze corrected evaluation before training.** Validate custom scorer/harness agreement on identical documents, with exact prompt and tokenizer boundary handling. Record both raw and character-normalized ARC accuracy, fixed FineWeb/Wiki losses, and generated-text quality. Preserve old reports as historical evidence. The observed confirmation partition is now development data; newly authored prompts are useful but are not independent human validation. Do not choose a recipe by repeatedly inspecting official test results.

2. **Run one 1B-token beta2 ablation from scratch.** Candidate: AdamW (0.9, 0.999), LR 0.0006, warmup 1000, batch 32, accumulation 1, context 512, same 60/40 mixture, seed 20261001, architecture and token sequence as the existing long-warmup control. Keep epsilon, weight decay, schedule and precision fixed. The old 0.95 checkpoint can serve as the control only after verifying the updated trainer reproduces the old configuration's numerical path. Make beta2 explicit in recipe/provenance and enforce it on resume. Do not silently change beta2 while loading old Adam moments and its step counter. Allow 2.6–3 hours plus setup/evaluation.

3. **Decide using more than one metric.** A lower training loss alone is insufficient. Suggested practical targets to freeze before launch: at least a 3% reduction in held-out FineWeb perplexity with no more than 2% WikiText perplexity regression; a directional gain in raw ARC accuracy without a greater than 1-point character-normalized regression; and improved full-passage coherence on a fixed fresh prompt set. These are our proposed decision thresholds, not paper results or significance guarantees. A one-seed result remains preliminary. Report paired question uncertainty and all failures. Optimizer efficiency and a useful product capability are separate claims.

4. **Choose only one subsequent route if time remains.** If the optimizer clearly helps, continue the successful scratch lineage for a bounded additional 1–2B tokens using an explicitly recorded new stage. Current resume code requires an identical recipe and code identity; a changed schedule is not a transparent resume and needs a separate, tested continuation path. If gains remain mainly linguistic and coherence is still poor, prioritize a matched data-stage experiment: equal budgets from the same parent, old mixture versus a modest fraction of simpler explanatory material. A 2 × 0.5B comparison is roughly 2.6–3 GPU hours, excluding data preparation.

5. **Keep data preparation bounded.** A possible public source is [SmolLM Corpus Cosmopedia v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), with audience/format metadata and public synthetic educational text. Sample a few shards, inspect actual content and license, pin revisions, and exclude benchmark overlap before use. The corpus is about 122GB: do not download it all. Its applicability under the rules is our interpretation of permitted public datasets, not an organizer-specific approval of Cosmopedia. TinyStories is explicitly named by the rules. Do not conflate public-corpus use with privately distilling a teacher. New data must beat an equal-budget control; the SmolLM counterexample means this route has no guaranteed boost.

6. **Stop experimentation and ship the best defensible checkpoint.** By noon October 1, run the pinned full benchmark suite on the selected candidate and baseline, inspect complete generations, verify lineage and parameter count, and finalize model weights, README, demonstration, screenshots and hosted video. Complete the Devpost entry with the real team identity. Keep the existing release available if the new experiment fails.

Muon is the next optimizer research candidate after a tuned Adam baseline, but adding it competes with the data experiment for the remaining hours. A hypothetical 22-layer, width-384, FF-1152 model has 48,481,152 parameters under the current implementation. That fits the cap, but is not the existing `pocket-deep.json` config, has no measured throughput/result, and requires a fresh training comparison. Neither it nor recurrent layers, MTP, MoE, RL, longer context, or a new tokenizer should be stacked into the proposed beta2 ablation.

We have time for another useful attempt. The evidence supports a specific, inexpensive optimizer test and a bounded data/continuation option. It does not yet establish a meaningful generation improvement or a likely competition win.

## Reproducible local evidence

Run `python3 scripts/audit_deadline_research.py` from this worktree. It reconstructs scoring and calculates throughput, memory, corpus exposures, beta2 half-lives and parameter counts without training or inference. Input artifact hashes are recorded in [deadline audit JSON](../results/research/deadline-audit-2026-09-30.json). This supplements, and does not overwrite, the frozen [warmup review](warmup-confirmation-review.md) and its evidence.
