# ODIN refinement implementation plan

> **For agentic workers:** Use superpowers:executing-plans inline. User has authorized execution; do not repeat approval gates.

**Goal:** Run and honestly evaluate a controlled improvement experiment within the 50M cap.
**Architecture:** Add source-weighted sampling and a separate reproducible corpus builder. Existing Transformer supports the deeper configuration without changes. A resumable experiment runner sequences pilots, development-only selection, and final training/evaluation.
**Tech Stack:** Existing PyTorch 2.7, NumPy, tokenizers, lm-eval environment.
**Spec:** ../specs/2026-09-29-refinement.md

## Global constraints
- Preserve published baseline and keep work in research/refinement-v2.
- Random initialization; <=50,000,000 parameters including tied embeddings counted once.
- Public pinned data, no new teacher output, holdouts excluded from training.
- All pilot and final cost disclosed; never call a launched run complete.

## Review focus
- Mixture resume must restore RNG and verify every sampled source fingerprint.
- Invalid mixture weights/files and out-of-vocabulary IDs must fail before training.
- Story separator must not become training text or glue stories together.
- Selection must use development data, not previously observed official scores.
- Runner restart must not overwrite checkpoints or silently reselect a winner.

### Task 1: Reproducible mixture data and training
**Files:** odin/corpus.py, odin/train.py, tests/test_corpus.py, tests/test_training.py, configs/pocket-deep.json.
**Interfaces:** corpus builder emits train.bin/dev.bin/tokenizer.json/manifest.json and story train/dev bins; manifest optional train_sampling entries {file,weight}. train() reads source weights and fingerprints every file.
- [ ] Test mixture distribution, exact resume, invalid weights and source mutation rejection; test story parsing and heldout exclusion.
- [ ] Implement corpus source filtering, frozen tokenizer, indexed source provenance, and sampling with existing torch RNG.
- [ ] Run CPU tests and GPU smoke; verify both parameter counts.
- [ ] Commit and run preparation to an atomic completion manifest.

### Task 2: Development evaluation and orchestration
**Files:** scripts/refinement.py, odin/development.py, tests/test_development.py, experiments/prompts.json.
**Interfaces:** evaluate_development(checkpoint, data, prompts) produces checkpoint-bound per-domain NLL, samples, repetition and local comparison results. select_candidate(reports) implements spec gates. Runner saves immutable pilot exports, decision JSON, and resumes only matching run recipes.
- [ ] Test deterministic evaluation and selection gates, missing/corrupt reports fail closed.
- [ ] Implement fixed 256 windows/domain, frozen prompt suite, restart-safe runner, source hashes and atomic status.
- [ ] Run tests, baseline development evaluation, and GPU smoke before long run.
- [ ] Run all three pilots at equal processed tokens; inspect complete sample sets and decision.

### Task 3: Train, compare, and deliver
**Files:** results/refinement/*, docs/refinement-results.md, evidence/model card as warranted.
- [ ] Train selected candidate to 3B tokens, retain recovery checkpoint and immutable final export.
- [ ] Run official full evaluation and original/new qualitative suites; measure CPU throughput.
- [ ] Write results with negative findings, limitations, pilot overhead, provenance and reproduction commands.
- [ ] Fresh whole-branch review; fix important findings with regression tests.
- [ ] Preserve v0.1.0; update product/release only if supported by observed results.
