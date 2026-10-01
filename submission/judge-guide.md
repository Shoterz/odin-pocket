# Judge's guide

**ODIN Pocket — Axiom AI — Yong Li Zhong — Track 01**

## Start here

1. Follow the fresh-environment commands in the [README](../README.md). No paid account or API key is required. Download the checkpoint from [v0.1.0](https://github.com/Shoterz/odin-pocket/releases/tag/v0.1.0) and check its SHA-256 before loading it.
2. Start the workbench with `bash run.sh --checkpoint runs/pocket/submission.pt --evidence submission/evidence`. Open `http://127.0.0.1:8766`.
3. Continue a passage, inspect token probabilities, compare two endings, and open Model evidence. Every output comes from the loaded local model. The UI exposes the model's limitations as well as its measurements.

## Evidence by judging criterion

| Criterion | Where to inspect |
| --- | --- |
| Perplexity and accuracy | [README table](../README.md#official-evaluation), [full machine-readable report](../results/official.json), [evaluation implementation](../odin/evaluate.py) |
| Reasoning performance | Required zero-shot tasks above; [all frozen qualitative outputs](../results/product.json). No claim of reliable general reasoning. |
| Training efficiency | [Submitted training summary](evidence/training-summary.json), [recorded loss/VRAM metrics](evidence/training-metrics.jsonl), [broader experiment cost](../docs/research-summary.md#compute-disclosure) |
| Innovation | Inspectable local generation and evidence tied to checkpoint identity; exact-resume/provenance checks; measured KV-cache speedup. Standard Transformer components are credited rather than claimed as novel. |
| Documentation and demo | [Reproduction guide](../docs/reproduce.md), [model card](evidence/model-card.md), [English-captioned video](video/odin-pocket-demo.mp4), [screenshots](screenshots/README.md), [research results](../docs/research-summary.md) |

## Eligibility and identity

- **49,295,872 total trainable parameters**, including tied input/output embeddings counted once. [Config](../configs/pocket.json); [parameter-count command](../README.md#architecture).
- All model weights begin randomly. No pretrained initialization, private teacher training targets or distillation. [Training implementation](../odin/train.py).
- Submitted training data: public FineWeb-Edu and WikiText-103 training text. [Data card](../docs/data-card.md), [manifest with revisions/hashes](evidence/data-manifest.json).
- Submitted checkpoint SHA-256: `4634f90b2120a7a128a0a4bbd59ae70056dbeff7baee2a35bf551fe0da90ab7e`.
- Scores are self-reported runs of the required benchmark suite, not organizer-certified results. The full report is tied to the submitted checkpoint. Research checkpoints and their validation results remain separate.
- [Built With and AI disclosure](built-with.md): OpenAI Codex assisted implementation, tests, documentation and review. The model itself runs locally without hosted inference.

## Reproduction scope

Inference and inspection require only source, dependencies and the exported checkpoint. Recreating training/evaluation data requires downloading the pinned public sources. Full training requires a CUDA GPU for the recorded runtime; CPU inference is supported. The tests cover core computation, resuming training, evidence validation and HTTP behavior. Exact numerical reproduction across different hardware/software is not promised.
