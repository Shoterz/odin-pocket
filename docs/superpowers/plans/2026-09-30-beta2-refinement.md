# AdamW beta2 refinement implementation plan

> Execute inline with executing-plans; obtain one fresh code review before GPU work.

**Goal:** Execute the approved deadline research plan's first, single-factor 1B-token experiment and prepare matched evaluation for a continuation decision.
**Spec:** `docs/deadline-research-2026-09-30.md`, approved by the user's “go on with this plan”.
**Architecture:** Preserve the prior frozen trainer/evaluator and historical evidence. Fork the small controlled trainer into `odin/beta_train.py` with explicit beta2; add an evaluation wrapper with standard harness metrics and a bounded sequential runner.
**Tech stack:** Existing PyTorch 2.7 environment, RTX 4070, pinned lm-evaluation-harness 0.4.13.

## Global constraints

- Existing research/refinement-v2 worktree; original public release unchanged until evidence warrants promotion.
- Candidate: 49,295,872 parameters, seed 20261001, 61,036 steps, batch 32, accumulation 1, context 512, LR .0006, warmup 1000, existing cosine floor .1, existing 60/40 FineWeb-Edu/Wiki mixture. Only optimizer beta2 changes from .95 to .999.
- Train from random initialization. Validate numerical equivalence of the new trainer at beta2 .95 to the original engine, then reuse the existing long-warmup 1B control.
- New charged candidate cap: 1,000,013,824 tokens; no automatic retries or replay allowance. Verification training is separately accounted and bounded to small smoke runs.
- Shared original runner lock; refuse mismatched code/data/plan/checkpoint identities. Stop the training child after four hours, at October 1 noon Singapore, or below 8GiB free storage, whichever comes first. No simultaneous writer.
- Develop on all 570 previously observed ARC validation questions, reporting raw acc and character-normalized acc_norm. Do not call them untouched confirmation or use official test data for recipe selection.
- Freeze fresh 16 prompts × two seeds, temperature .8, top-k 40, 128 new-token maximum. Assess complete outputs. Author review is not independent human validation.
- Practical numeric gate: FineWeb perplexity reduction >=3%, WikiText perplexity increase <=2%, raw ARC improvement >0, character-normalized regression <=1 percentage point. These are heuristic thresholds, not statistical guarantees.
- Quality gate for further training: at least four additional fully coherent/factually consistent outputs out of 32, with no newly systematic failure category; separately review relevance/entity consistency. No automatic release/continuation based only on the numeric gate.
- Automatic runner stops after matched development evaluation, numeric comparison and anonymous generation package. Inspect those outputs before selecting the next conditional step. Final official test evaluation remains separate.

## Task 1: Trainer and scoring

**Interfaces:** beta_train.train adds beta2 to the frozen train signature and recipe; beta_eval.evaluate returns standard-scored development questions, fixed domain losses, fresh generations and identity-bound protocol; beta_eval.compare reports the frozen numeric gates.

- [ ] Write failing behavioral tests for beta2 use/validation/exact resume, old-engine equivalence, raw versus character-normalized scoring, and mismatched reports.
- [ ] Implement minimal trainer fork and scoring wrapper; preserve original files.
- [ ] Run targeted tests, then the full suite. Expect green; sandbox HTTP socket requires host execution.

## Task 2: Bounded runner and verification

**Interfaces:** scripts/beta_refinement.py provides prepare, smoke, run; writes results/beta-refinement and runs/beta-refinement/candidate, never historical result directories except the shared advisory lock.

- [ ] Test budget/replay and deadline refusal, single-writer behavior, and refusal of stale checkpoint/verification evidence.
- [ ] Freeze source/data/checkpoint identities, all thresholds and prompts before candidate training.
- [ ] Verify exact CPU and compiled GPU beta2 .95 equivalence, candidate smoke finiteness and matching sampler counts; validate scoring against real installed harness processing and requests.
- [ ] Obtain fresh review and fix important findings before the full GPU run.

## Task 3: Train and assess

- [ ] Launch the bounded runner, verify actual progress and publish a durable status path.
- [ ] After completion read matched numeric results and all anonymous outputs; record decision and compute. Only then choose the authorized conditional continuation/data route if time permits.
- [ ] End GPU experimentation by October 1 noon Singapore; protect final benchmarking, demo and submission work.

## Review focus

- A changed beta2 cannot silently reuse old Adam moments or evade recipe checks.
- Standard acc_norm uses characters of the original answer, excluding only the inserted separator.
- Cached evaluations bind to the exact checkpoint, prompts, scoring implementation and stored data.
- Replayed or partial charged work cannot exceed the new experiment cap.
- GPU equivalence and smoke records bind to code versions; stale verification must block launch.
