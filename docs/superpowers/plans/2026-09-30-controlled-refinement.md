# Controlled refinement execution plan

> Execute natively using superpowers:executing-plans; obtain a fresh review before long GPU training.

**Goal:** Execute the approved research-audit sequence and establish whether an explicitly balanced recipe improves ODIN under matched training budgets.

**Spec:** `docs/refinement-research-audit.md`, especially the recommended order of work. User approved execution on 2026-09-30.

**Architecture:** Add separate controlled-experiment modules; preserve the completed refinement's frozen source and checkpoints. Reuse the fixed tokenizer and 49,295,872-parameter model. Prepare individually weighted FineWeb, WikiText, and story streams from verified existing data. Record checkpoints, code/data identities, source draw counts, and evaluation identities.

**Global constraints:** RTX 4070 12GB, no paid services, no pretrained weights or distillation, total trainable parameters below 50M. Do not publish a replacement without demonstrated improvement. Never use official test scores to select a recipe.

## Experiment protocol

- Independent public comprehension: all 570 ARC-Easy validation questions, already excluded from training, deterministically split into 285 screening and 285 confirmation questions. Freeze the split and input hashes before any new result. No use of confirmation questions for screen selection.
- Domain losses: 256 fixed 512-token windows separately for FineWeb (extra-dev), WikiText validation, and stories; aggregate general loss is the mean of FineWeb and WikiText losses. Story loss is diagnostic only.
- Generation: existing protected 16 prompts × two seeds, scored later with a fixed relevance/entity/causality rubric. These prompts have been observed before; do not describe them as fresh independent validation. Produce randomized anonymous comparisons for final review.
- Source mixture control: 60% FineWeb / 40% WikiText / 0% stories. Story comparisons replace only FineWeb: 50/40/10 and 40/40/20.
- Four optimizer screens: peak LR 0.0006 or 0.0012 crossed with explicit warmup 200 or 1000 steps; batch 32 × context 512, accumulation 1, same seed, 7,813 steps (128,008,192 tokens). Full cosine decay to 10% in each screen. These are deliberately bounded hypotheses, not a claim of optimality.
- Select optimizer using screening ARC length-normalized accuracy, protected by per-domain NLL ≤ control +0.05 and accuracy ≥ control −0.03. Tie-break by mean general NLL; retain reference when no alternative qualifies. Story loss cannot select a winner.
- Two story-mixture screens use the selected optimizer and the same seed, steps, batch and schedule. Apply the same gates against the selected no-story control. A candidate must also improve ARC accuracy or mean general NLL; otherwise stop with no mixture winner.
- Longer confirmation: if a mixture qualifies, train that mixture and its no-story control from random initialization for 61,036 steps (1,000,013,824 tokens) each, same fresh seed and optimizer/warmup. Evaluate both on the confirmation questions only after both finish. Report paired bootstrap intervals and domain losses, then blinded generation review. A screening win is never a release claim.
- Six screens total 768,049,152 tokens; confirmation adds 2,000,027,648. Maximum charged training is 2,768,076,800 tokens plus separately reported smoke tests. Charge each step before computation in an append-only ledger, including partial/replayed steps after interruption; refuse a later stage that would exceed the cap. Preserve per-attempt elapsed time independently of checkpoint rollback, marking missing end records as unknown. Record actual duration and stop on nonfinite values or low disk.
- Architecture experiment is conditional on a confirmed data/optimizer gain. If it is not demonstrated, document that the prerequisite failed; do not spend an MTP run to disguise an inconclusive result. If confirmed, design a separate capped parameter-matched MTP probe using the confirmed recipe before implementation.

## Task 1 — data and evaluation

- [x] Test token splitting rejects incorrect EOS/document counts; source weights remain explicit; malformed weights fail.
- [x] Implement `odin/controlled_data.py`: verified source separation, domain validation bins, ARC split and immutable mixture manifests. Audit deterministic source samples; document limits of near-duplicate checks.
- [x] Test evaluation gates reject story-only wins, domain regressions, mismatched identities and incomplete questions. Implement `odin/controlled_eval.py` with separate screening/confirmation partitions, per-item scores, deterministic paired confidence intervals, and blind-review output.
- [ ] Freeze protocol and artifacts before screening; evaluate old checkpoints as diagnostic references only.

## Task 2 — training and orchestration

- [x] Test explicit warmup boundaries, realized source counts, exact resume and changed-recipe rejection on a tiny CPU model.
- [x] Implement `odin/controlled_train.py` as a separate trainer based on the frozen trainer, with explicit warmup and source-count telemetry included in checkpoints. Preserve old modules unchanged.
- [x] Implement `scripts/controlled_refinement.py`: bounded stages, code/data lock, single-writer lock, verified cached results, safe subprocess cleanup, atomic progress, resumable recovery, decision provenance and matched budgets.
- [ ] Run the complete suite, full-size GPU deterministic resume smoke, and fresh code review. Fix important findings before launch.

## Task 3 — execute and assess

- [ ] Execute all four optimizer screens and both mixture screens; persist decisions and exclusions.
- [ ] If eligible, execute both longer confirmation runs at the same budget and fresh seed.
- [ ] Inspect blinded continuations and paired confirmation results. Record success, inconclusive evidence, or regression honestly. Update product/release only if supported.

## Review focus

1. Corpus source boundaries and EOS counts: wrong boundaries must fail, not silently mix sources.
2. Changed source, protocol, recipe, or code during resume: reject stale checkpoints/reports.
3. Partial writes or failed child process: atomic records, checkpoint-authoritative recovery, no orphan GPU writer.
4. Incomplete or mismatched evaluations: refuse selection; story improvements cannot compensate for failed general-domain gates.
5. Compute and provenance: full parameter count, realized per-source token totals, no silent budget extension or test-set selection.

## Execution ledger

- Baseline suite: sandbox blocked only the HTTP socket test (53 other tests passed); rerun with socket permission requested.
- Authorization: user explicitly requested execution of the audit sequence; no additional design-approval round is needed.
- Baseline rerun: 53 passed, 1 CUDA-unavailable test skipped because CUDA was available.
- Task 1/2 verification after review fixes: full suite 65 passed, 1 skipped; failing recovery and work-ledger tests observed before fixes.
- Fresh review by review_controlled: three important findings (mixture link identity, partial preparation recovery, rollback accounting) fixed. Existing source links are validated; interrupted preparation reuses only correct links; each training step is charged before execution and attempts preserve timing. Corpus content hashes are checked at every stage.
- Minor review note addressed: rerunning blind-review generation preserves prior ratings when input texts match.
- Source audit: 100 deterministic samples per source, no within-sample pair above 0.8 five-word-shingle Jaccard. Manual inspection of initial samples found varied educational prose, encyclopedic articles, and simple stories; this does not establish corpus-wide cleanliness.
