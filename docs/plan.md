# ODIN Pocket implementation plan

Goal: deliver the design in [design.md](design.md), with measured training and evaluation.
Architecture: independent model/data/training/evaluation modules feeding a local inference workbench.
Stack: Python 3.10+, PyTorch, tokenizers, datasets, lm-evaluation-harness, standard-library HTTP and browser JavaScript.

Global constraints: <=50,000,000 parameters; no pretrained weights/distillation; benchmark test separation; real metrics only; loopback defaults; no paid compute authorized.
Execution: native in this workspace under the user's autonomous instruction. The provided .git is a read-only empty placeholder, so retain file-based ledger rather than pretend commits exist.

Review focus: future-token leakage; lost/duplicated targets in perplexity; mislabeled partial benchmarks; stale evidence from another checkpoint; malformed or concurrent inference requests.

- [x] 1. Model and tokenizer. Files `odin/model.py`, `odin/tokenizer.py`, `configs/pocket.json`, `tests/test_model.py`. Test causality, cap, shifted loss, Unicode roundtrip and save/load; implement; run pytest. Interface `ModelConfig`, `LanguageModel.forward(ids, targets=None)`, `Tokenizer.encode/decode`.
- [x] 2. Data and trainer. Files `odin/data.py`, `odin/train.py`, `tests/test_training.py`. Test dedup/split, checkpoint state, deterministic resume and real loss decrease. Implement public-corpus preparation, token cache, telemetry and bounded resumable trainer; run small CUDA pilot then production training. Interface token arrays + manifest; `load_checkpoint(path)` -> model/tokenizer/metadata.
- [x] 3. Evaluation. Files `odin/evaluate.py`, `tests/test_evaluation.py`. Verify hand-computable causal log-likelihood and rolling token coverage. Integrate harness; report task versions, counts, checkpoint SHA and partial/full status. Run required benchmark tasks and efficiency measurements.
- [x] 4. Product. Files `odin/server.py`, `web/`, `tests/test_server.py`. Test missing checkpoint, bounded input, real generated continuation, candidate likelihood and HTTP access. Build browser interface, inspect desktop/mobile and capture screenshots.
- [x] 5. Local submission artifacts. Files `README.md`, `docs/model-card.md`, `docs/data-card.md`, `submission/`. Document actual run/results/limits, parameter report, dependencies, attribution and AI assistance. Produce demo recording and screenshots. Run full tests and independent final review; address findings. Publish only with available authorization/account access.

Training and full evaluation completed. Local capture, visual inspection, final source and evidence are complete. Public publication and Devpost submission remain unverified.
