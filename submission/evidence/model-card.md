# ODIN Pocket model card

Checkpoint SHA-256: `4634f90b2120a7a128a0a4bbd59ae70056dbeff7baee2a35bf551fe0da90ab7e`. All values below are measured, bound to this checkpoint.

Parameters: **49,295,872**. Training tokens processed: **983,040,000**.
Training hardware: **NVIDIA GeForce RTX 4070**. Recorded training wall time: **3.923 hours**.
Approximate dense-transformer training compute (6 × parameters × tokens; excludes evaluation and some attention overhead): **2.908e+17 FLOPs**.

## Official zero-shot evaluation

| Task | Accuracy | Examples |
|---|---:|---:|
| hellaswag | 27.21% | 10,042 |
| arc_easy | 41.20% | 2,376 |
| piqa | 57.94% | 1,838 |
| winogrande | 50.04% | 1,267 |
| WikiText-103 raw test | 19.463 token perplexity | 297,911 tokens |

WikiText word perplexity: 39.107; bits per byte: 0.9908. Context 512, stride 256; every target counted once. Token perplexity is not comparable across different tokenizers. Harness version 0.4.13; task definitions and versions are in the JSON report. Results do not establish reasoning reliability or instruction-following ability.

## Data and intended use

FineWeb-Edu and WikiText-103 training text, with 280,797,698 unique stored training tokens and 2,918,352 development tokens. Sampling repeats training windows; processed tokens are not unique tokens. Byte-level BPE fitted only to training text. Random initialization, no pretrained weights or distillation.

Local next-token completion, candidate-continuation scoring, and education about compact model behavior. English-focused base model. Not a factual authority or medical/legal/financial advisor. May repeat, hallucinate, and reproduce biases or fragments of training text. No safety alignment training was performed. The benchmark overlap filter misses short and paraphrased overlaps; it is not proof of zero contamination.

AI assistance: OpenAI Codex wrote and reviewed implementation, tests, interface and documentation. The model was trained locally from scratch; Codex was not used to generate training targets.

## Measured local inference and qualitative behavior

On a 13th Gen Intel(R) Core(TM) i5-13400F with four CPU threads, paired generation measured **90.4 tokens/s** with KV caching versus **30.1** without it (3.01×). All 12 paired outputs were identical. This is one host and a six-prompt workload, not a universal speed guarantee. [Raw conditions and outputs](../../results/efficiency.json).

These are baseline results, not evidence of state-of-the-art performance or a guaranteed win. WinoGrande is approximately chance. Frozen qualitative examples scored 3/6 comparisons and show repetition and false factual claims; inspect [all final outputs](../../results/product.json). The model is suitable for studying local language modeling, not factual advice.
