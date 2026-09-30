# Warmup confirmation plan

> Execute natively using superpowers:executing-plans and obtain code review before GPU launch. The user approved the proposed longer matched comparison.

**Goal:** Test whether the longer warmup signal survives a longer budget, fresh training seed, and the previously unused confirmation questions.

**Spec:** Follow-up proposed in `docs/controlled-results.md`; this is a separate experiment, not a modification of the completed screening protocol.

**Architecture:** Reuse the frozen controlled trainer, data, evaluator and runner helpers unchanged. Add `scripts/warmup_confirmation.py` to orchestrate two new runs and write results under `results/warmup-confirmation`. Use the original runner lock to prevent simultaneous training pipelines.

## Frozen protocol

- Architecture: 49,295,872 parameters, 12 layers, width 512, context 512; original 16,384-token BPE.
- Both models start from random weights with seed 20261001. No continuation of the short-screen checkpoints.
- Data: exactly the verified 60% FineWeb / 40% WikiText / 0% stories mixture.
- Learning rate: 0.0006 with existing cosine decay to 10%; batch 32, accumulation 1, 16,384 tokens per step. Compiled deterministic CUDA training.
- Only changed factor: warmup of 200 versus 1000 steps.
- Each model: 61,036 steps, 1,000,013,824 processed tokens. New budget: 2,000,027,648 charged tokens. Including prior screens, cumulative cap remains 2,768,076,800. Count partial/replayed steps against the cap.
- The cap has no replay allowance. Before launching either training subprocess, validate both checkpoints and ensure charged work plus both remaining training budgets fits. If interruption creates uncheckpointed charged work, stop immediately rather than spending on a pair that can no longer finish within the cap.
- Run names: `warmup-confirm-200` and `warmup-confirm-1000` in the existing `runs/controlled` directory. Do not edit the prior run's results or decisions.
- Both training runs must finish before either is evaluated on the 285 reserved ARC validation confirmation questions. Evaluate the frozen published model and previous B on the same questions afterward as historical references. References have different recipes/budgets and are not causal controls.
- Primary comparison: long minus short warmup in question accuracy, with the existing deterministic 10,000-resample paired bootstrap interval. A positive lower bound plus no FineWeb or WikiText NLL regression above 0.05 is evidence on this question set, not evidence across training seeds.
- Equal accuracy or an interval spanning zero is inconclusive for a reliable accuracy improvement. Report domain losses separately; do not compensate a failed general-domain guard with story loss.
- Generate the same 16 protected prompts × two seeds for all four models. Create a randomized anonymous review and preserve existing review ratings on rerun. Require qualitative review before any release decision. No automatic publication or architecture experiment.
- Produce a numerical report automatically after evaluation; explicitly mark qualitative review pending and any inconclusive primary result. Preserve all checkpoints, identities, costs and failed attempts.

## Implementation tasks

1. Test matched schedules/budgets, refusal of mismatched confirmation reports, interval/guard interpretation, and train-before-evaluate ordering using dependency-boundary fakes. Implement the small wrapper and result summarizer.
2. Verify prior source/data locks and unchanged GPU smoke identity. Run the entire test suite and a fresh review. Fix important findings before launch.
3. Freeze this plan and wrapper identities. Launch on the RTX 4070, verify actual training/checkpoint progress, and record status/recovery instructions. When complete, inspect automatic results and all anonymous continuations before deciding on a release.

## Review focus

- No confirmation evaluation before both matched checkpoints finish; no short-run continuation.
- No rewriting old decisions or results through reused module globals.
- Same data, tokenizer, architecture, seed, budget and LR schedule treatment; only warmup differs.
- Resume/cached results reject changed sources, protocol or checkpoint identities.
- Shared writer lock and charged replay accounting remain effective; historical compute and this experiment's compute are clearly distinguished.

## Review record

Fresh review found one important recovery issue: a current-run-only budget check could finish the first model before discovering that replay made the pair unaffordable. Added a checkpoint-authoritative pair budget check and regression test (observed failing before implementation). Existing trainer, evaluator and prior experiment code remain unchanged.
