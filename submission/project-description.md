# ODIN Pocket

**Team:** Axiom AI · **Solo developer:** Yong Li Zhong

## What it is

ODIN Pocket is a small, locally runnable language model with a workbench for passage completion, candidate-ending comparison and model inspection. The full model has 49,295,872 trainable parameters, including its tied embedding/output weights. Every generated token and every comparison score comes from our own checkpoint.

## Why we built it

Small language models make an unusually concrete engineering challenge: a fixed parameter budget, limited consumer hardware, a finite amount of training time, and nowhere to hide weak capability. We wanted a model whose training and behavior a judge could inspect directly, and whose entire inference flow could run on an ordinary CPU without a service account.

## How it works

We implemented a causal Transformer in PyTorch: 12 blocks, width 512, rotary-position attention, RMS normalization, SwiGLU, and tied output embeddings. We trained a 16,384-entry byte-level BPE tokenizer on training-only public text. All model weights start randomly.

The corpus combines educational web text from FineWeb-Edu with WikiText-103 training articles. Preparation deduplicates complete documents, uses stable document-level development splits, and rejects matches to protected benchmark 13-grams. It produces 280,797,698 stored training tokens and 2,918,352 development tokens. The filter's short-overlap and paraphrase limitations are documented.

Training runs on one NVIDIA RTX 4070 with 12 GB VRAM. Checkpoints preserve optimizer/RNG state and bind the run to data, tokenizer, configuration and source hashes. The local UI loads those weights directly, measures inference speed, exposes token probabilities, and shows benchmark evidence only when the report's checkpoint hash matches.

## Results

| Benchmark | Accuracy | Length-normalized accuracy | Examples |
|---|---:|---:|---:|
| HellaSwag | 27.21% | 27.80% | 10,042 |
| ARC-Easy | 41.20% | 36.24% | 2,376 |
| PIQA | 57.94% | 56.96% | 1,838 |
| WinoGrande | 50.04% | — | 1,267 |

WikiText-103 raw test: **19.463 token perplexity**, 39.107 word perplexity, 0.9908 bits/byte over all 297,911 targets. The protocol and complete text identity are in [the full report](../results/official.json).

The model processed 983,040,000 tokens in 3.923 hours on one RTX 4070. Approximate training compute: 2.908e+17 FLOPs. On a 13th Gen Intel(R) Core(TM) i5-13400F with four CPU threads, paired generation measured **90.4 tokens/s** with KV caching versus **30.1** without it (3.01×). All 12 paired outputs were identical. This is one host and a six-prompt workload, not a universal speed guarantee. [Raw conditions and outputs](../results/efficiency.json).

These are baseline results, not evidence of state-of-the-art performance or a guaranteed win. WinoGrande is approximately chance. Frozen qualitative examples scored 3/6 comparisons and show repetition and false factual claims; inspect [all final outputs](../results/product.json). The model is suitable for studying local language modeling, not factual advice.

## What we learned

Correctness has to span the entire experiment. A good-looking loss curve is insufficient if benchmark scoring drops targets, a resume silently changes data, or a UI shows measurements from another checkpoint. We added tests for those boundaries and used independent code review to find stale-log and stale-evidence errors before submission.

## Limits and disclosure

This is an English-focused base language model, not an instruction-tuned assistant. It can hallucinate and repeat; candidate preference is not factual verification. We do not claim a novel Transformer architecture or production reasoning reliability. The contribution is a complete, measurable small-model system built under the competition's constraints.

OpenAI Codex assisted the implementation, testing, interface, documentation and review. It did not generate private training targets. Datasets, frameworks, evaluation tooling and hardware are credited in Built With and the data/model cards.
