# ODIN Pocket design

## Intent and constraints
Build a competitive GIBC V2 Track 01 submission: a genuinely trained, locally runnable language model under 50,000,000 parameters, supported by reproducible evidence. Winning is an ambition, not a claim or a completion check that software can guarantee. User authorized autonomous execution until completion.

Rules verified 2026-09-29: https://gibc-v2.devpost.com/rules . Random initialization; no pretrained weights or distillation; embeddings and output head included. Required tasks are HellaSwag, ARC-Easy, PIQA, WinoGrande using lm-evaluation-harness and held-out WikiText-103 perplexity. Report hardware, time and compute; disclose AI assistance. Plan for the earlier published deadline: October 1, 23:45 UTC+8.

## Chosen approach
A 12-layer causal Transformer with width 512, 8 attention heads, SwiGLU intermediate width 1536, rotary positions, RMS normalization, tied embeddings and a byte-level 16,384-entry BPE tokenizer trained on training documents. Context 512 initially, extensible to 1024 after measuring throughput. Parameter budget enforced before training. PyTorch scaled dot-product attention; bfloat16 CUDA training with float32 optimizer, clipping and warmup/cosine learning rate. A tiny configuration exercises correctness on CPU.

Alternatives considered: a narrow policy encoder would fit a product use case but mismatches the causal benchmarks; a recurrent/shared-layer model has innovation potential but adds optimization risk with two days remaining. A measured causal baseline takes precedence over speculative novelty.

ODIN Pocket is an offline reading and writing workbench: complete a passage, compare candidate continuations by actual model likelihood, inspect tokenization and resource use, and examine the training/evaluation record. No chat-capability claim. Every generated token and candidate score must come from the submitted checkpoint, without an API or rule-based answer substitute.

## Data and training
Use licensed public corpora with source/revision/split attribution, exact-document deduplication and deterministic development split. Never train on benchmark validation/test records. Fit tokenizer only on training data. Persist token files and hashes. Public TinyStories is expressly allowed by rules; document synthetic origin if used, with no private teacher generation. Start weights afresh in this directory. Existing environments and public cached data may be read; sibling projects are not modified and their checkpoints are not reused.

Training writes atomic resumable checkpoints containing model, optimizer, RNG, config, tokenizer hash and token counters. JSONL logs record measured loss, tokens, wall time, hardware, throughput and peak VRAM. Evaluate development data on a fixed sample; official test evaluations are reporting, never training selection. Nonfinite loss or unavailable requested CUDA fails clearly. Choose training duration from measured throughput, within local compute and deadline; no paid resources without a spending limit.

## Evidence and interface
The local HTTP service serves a responsive accessible static interface plus bounded generation/likelihood endpoints. Bind loopback by default, validate request sizes and numeric limits, render text safely, serialize inference, return explicit errors if checkpoint is missing. Evidence reads recorded artifacts; missing metrics display as unmeasured. Visual language: cool paper (#f2f5f8), deep navy (#15283f), cobalt (#345ee8), muted slate (#63748a), pale blue (#e1eafa), white. A document editor with a token probability strip is the signature; typography uses system sans for interaction, Georgia for generated prose, monospace for measured evidence.

## Completion criteria
Passing causal masking, shifting, tokenizer roundtrip, checkpoint/resume, data separation, evaluation likelihood and HTTP tests; a trained checkpoint with provenance; full official benchmark reports and WikiText perplexity; runnable local demo; English README, model/data cards, source/license inventory, parameter count, demo video and three screenshots. Public repository, video upload and Devpost team submission require actual available account access; never label them done without evidence. Low quality must remain visible and trigger improvement, not inflated claims.
