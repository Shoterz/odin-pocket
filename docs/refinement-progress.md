# Refinement execution ledger

Plan: docs/superpowers/plans/2026-09-29-refinement.md
Spec: docs/superpowers/specs/2026-09-29-refinement.md

- Baseline f251662 isolated into research/refinement-v2 under experiments/refinement.
- Baseline CPU suite: 30 passed, HTTP socket test sandbox-blocked; rerun outside sandbox requested.
- GPU RTX4070 idle, 10,982MiB free; filesystem ~76GB free.
- Ruling: retain existing tokenizer, holdout, model implementation and inference engine — isolates corpus/depth effects and protects baseline compatibility.
- Ruling: full-run schedule declared before pilots, pilot stop_after instead of short-run LR anneal — permits identical-recipe exact resumption without silently changing provenance.
- Interfaces: builder train_sampling entries consumed by train fingerprint/sampler; development evaluator consumes additional story-dev.bin, original dev.bin. Both must be hashed in manifests. Final release must disclose added synthetic public corpus.

- Task 1 complete: weighted sampler passes exact CPU resume and source mutation tests. Full GPU compiled smoke verified both actual parameter counts (49,295,872 and 49,069,440), with zero maximum weight difference after interruption/resume under deterministic kernels.
- Investigation: default CUDA kernels produced small differences before restart (max full/resumed weight difference 0.00234 after four steps). Enabling deterministic kernels and CUBLAS workspace :4096:8 removed the discrepancy for both architectures. Settings now in checkpoint recipe; source-bound smoke saved.
- Ruling: use compiled batch32/accumulation1, preserving 16,384 tokens/step — measured ~109,548 tokens/s in short throughput probe versus ~70,442 uncompiled batch16; probe uses random repeated batches and is not a quality measurement or final runtime guarantee. All candidates share settings. First full model smoke peak remains within12GB.
- Prepared corpus: 889,026,560 web/wiki training positions; 576,905,922 story training positions. Shared original web dev2,918,352; story dev5,880,301. FineWeb726,000 documents scanned, TinyStories2,717,495. No claim that all positions are unique natural-language content.
- Independent review caught incomplete holdout/source verification and report identity gaps. Added full baseline protected-file inventory equality, input hashes and pinned story SHA, plus complete report identity/protocol/sample validation.
- Ruling: preparation was already in progress when input audit improved. The runner independently validates exact original input hashes and every finished output before attaching the stronger audit and freezing corpus views; no training may start earlier.
- Independent runner review caught JSON tuple restart mismatch, checkpoint/summary publication race, missing SIGTERM cleanup, and unfrozen architecture contents. Added canonical JSON lock, authoritative checkpoint summary recovery, child cleanup handler, and config hashes. Regression tests cover restart and actual SIGTERM child cleanup.
- Task 2 implementation complete; final suite and pilot execution pending. Runtime/output status will be written to results/refinement/status.json. No automatic publication or baseline replacement.

- Final verification: 53 tests passed, 1 skipped (the no-CUDA failure test skips when CUDA is available). CPU-only runs exercised that path successfully. Final independent review: no remaining blockers. Source commit a1797d6.
- Training launched 2026-09-29 14:54 UTC (22:54 Singapore), runner PID3402960. Current stage A, run_id065e2c54-c151-480b-baf3-6e1f4f0b6141. First recovery checkpoint at step1000 /16,384,000 tokens: dev NLL4.7576 from initialization9.7859, ~108,243 tokens/s, peak allocated6,882,606,080bytes. GPU observed100% utilized,65C.
- Baseline development reference saved: webNLL3.28934, storyNLL2.83726, repeated4gram fraction0.04443, local comparison6/8. All32 raw continuations retained. These local comparisons are not independent benchmarks.
- Current stage/status: results/refinement/status.json. Training metrics: runs/refinement/{A,B,C}/metrics.jsonl. On success the runner writes docs/refinement-results.md and all final measurements; it keeps v0.1.0 unchanged and records completed-needs-quality-review. On failure it records the exception and leaves recovery checkpoints.
- Run remains IN PROGRESS. No new model quality or benchmark improvement claimed.

- 2026-09-30: runner finished2026-09-29 23:54UTC /Sep30 07:54Singapore. B selected and trained to3,000,008,704tokens. All pilots+winner total3,500,015,616tokens /8.857recorded hours. Full evaluations completed.
- Quality review complete: verified final checkpoint/run/corpus/report identities; inspected all32 frozen final continuations and6 original product continuations, and reran the screenshot prompt at three temperatures on CPU. Mixed benchmark results and continuing semantic failures do not justify release replacement. See docs/refinement-quality-review.md.
- Local B preview started separately on127.0.0.1:8767 with matching benchmark evidence. Published v0.1.0 unchanged. No new training launched in this review.
