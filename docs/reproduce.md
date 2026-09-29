# Reproduction and evidence boundaries

## What a judge can check quickly

1. Install the documented Python environment and download the provided final checkpoint.
2. Run the parameter-count command: 49,295,872 parameters, below 50,000,000.
3. Run `python -m pytest -q` to verify model, data, resume, scoring and API behavior.
4. Start `bash run.sh --checkpoint runs/pocket/submission.pt --evidence submission/evidence` and open localhost:8766. The source archive includes the exported training evidence; the original training directory is not required for inference.
5. Compare checkpoint SHA-256 with `results/official.json`. The UI refuses to show benchmark reports from different weights.
6. Re-run `odin.evaluate` for all four official tasks plus the exact WikiText-103 raw test Arrow file. The README documents the cache path and command.

## What takes longer

Recreating the public corpus requires the pinned FineWeb shard and pinned benchmark/WikiText files. The prepared manifest records source hashes, filters, tokenizer and token-file hashes. The tokenizer is trained from the training partition only. Full from-scratch pretraining requires the documented GPU time; a short `--stop-after` run validates the path without being represented as a final model.

`scripts/train_and_evaluate.py` runs the verified corpus through training, immutable export, full evaluation and local evidence packaging. Use `--dataset-cache` when the datasets were cached outside the default Hugging Face location. Its offline evaluation uses already-cached official datasets and does not send private passages anywhere.

Optional browser verification and recording need `pip install playwright` followed by `python -m playwright install chromium`; video encoding also needs FFmpeg with the subtitles filter. Run `python scripts/browser_check.py --require-model` against the local server. `--chromium /path/to/chrome` selects an existing installation. After full evaluation, `python scripts/record_demo.py` records real interactions and burns English captions into the MP4. The recorder rejects a server checkpoint that differs from the full report and checks all five rendered scores. `scripts/finalize_local.py` runs final package validation, tests, frozen product and CPU measurements, browser captures and recording in sequence. Its `--wait` option queues these steps behind completed training/evaluation; it does not publish anything.

## Meaning of measurements

- Training tokens count processed sampled windows, including repeated windows. Stored unique corpus token positions are separately reported.
- Development loss uses the same seeded sample of eight batches for periodic comparison; it is not a full-corpus development estimate.
- Training wall time covers the loop's development evaluations and prior checkpoint saves. Final checkpoint writing, preprocessing, the pilot and official evaluation are additional work.
- Approximate FLOPs use `6 × parameters × processed tokens`; this estimate omits some attention overhead and is not a hardware profiler measurement.
- CPU inference speed depends on host load, prompt length, generation length and thread count. The product report retains its conditions and raw outputs.
- Official accuracy is zero-shot harness accuracy; normalized accuracy is retained when the task supplies it. These metrics are not interchangeable.
- WikiText token perplexity depends on the tokenizer. Word perplexity and bits per byte are also recorded, with an explicit raw-text/article/stride protocol.
- Benchmark overlap exclusion detects normalized 13-gram matches, not all semantic contamination. Short and paraphrased overlaps remain a limitation.
- Six frozen product comparisons are qualitative examples, not an independent capability benchmark. Their results never replace the official tests.

## Releasing the project

`scripts/build_source_archive.py --require-final` creates a source ZIP with the documented files and final evidence, excluding raw corpora, virtual environments and optimizer state. The final inference checkpoint is distributed separately because it is larger than GitHub's normal per-file source limit. Source publication, a hosted demo-video link and Devpost team information must each be verified independently before declaring the entry submitted.
