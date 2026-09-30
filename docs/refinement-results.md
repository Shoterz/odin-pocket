# ODIN refinement: measured results

Candidate **B** was selected using development data only. Final checkpoint SHA256: `dab13b8947f5a7ac4bc5dcd0a9eb0056d69bcad0dcd4edc99f757711fd8811bb`.

The published v0.1.0 model remains unchanged. This report does not establish a winning submission; review all saved continuations before promoting the candidate.

## Controlled pilots

Each pilot processed 250,003,456 tokens, with one seed and a shared tokenizer. A: expanded educational corpus, original architecture. B: same architecture, 20% public TinyStories mixture. C: same mixture, deeper/narrower architecture. The short pilot and single seed limit inference about long-run architecture quality.

| Candidate | Parameters | Web NLL | Story NLL | Repeated 4-grams |
|---|---:|---:|---:|---:|
| A | 49,295,872 | 3.6758 | 3.1641 | 0.086 |
| B | 49,295,872 | 3.7501 | 1.5113 | 0.043 |
| C | 49,069,440 | 3.7735 | 1.5171 | 0.031 |

Selection details: `results/refinement/decision.json`. Baseline and all 32 continuations per candidate are saved in `*-development.json`; no cherry-picked samples.

## Final official evaluation

| Task | Published baseline | Candidate | Difference |
|---|---:|---:|---:|
| hellaswag | 27.21% | 27.58% | +0.38 pp |
| arc_easy | 41.20% | 43.22% | +2.02 pp |
| piqa | 57.94% | 58.65% | +0.71 pp |
| winogrande | 50.04% | 48.15% | -1.89 pp |

WikiText-103 token perplexity: 19.463 → 21.037, with the same tokenizer and scoring protocol.

Final development web/story NLL: 3.2429 / 1.1782. Baseline: 3.2893 / 2.8373.
Final repeated 4-gram fraction: 0.051; baseline 0.044. This is a repetition metric, not a factuality or coherence score.

## Compute and provenance

Local RTX4070 12GB. All three runs combined: 3,500,015,616 processed tokens, 8.857 recorded training hours, approximate dense training compute 1.035e+18 FLOPs. These include losing pilots and selected pilot once; exclude preprocessing, throughput probes, smoke tests and evaluations, which have separate logs. The published baseline's earlier training is additional historical cost.

Input verification: `results/refinement/input-audit.json`; corpus manifests: `data/common/manifest.json`, `data/edu/manifest.json`, `data/mixed/manifest.json`. TinyStories is public synthetic data generated upstream by GPT-4, explicitly allowed by the rules; no new teacher calls or pretrained weights were used. Source pin and license are in the manifests. Exact overlap filtering cannot rule out paraphrases or short overlap.

## Reproduction

Use the same pinned environment as v0.1.0. Prepare `data/common` with `python -m odin.corpus --help`, then run `python scripts/refinement.py --baseline-root /path/to/original/baseline --dataset-cache /path/to/huggingface/datasets`. The runner refuses changed recipes, data, source files, report identities or selection decisions. Cached/uncached CPU measurements and original product prompts are in `final-efficiency.json` and `final-product.json`.
