# Built With

- Python 3.10; PyTorch 2.7.0+cu126; CUDA runtime 12.6.
- NumPy 2.2.6; Hugging Face Tokenizers 0.22.1, Datasets 4.1.1 and Transformers 4.56.2; Apache Arrow 21.0.0.
- EleutherAI lm-evaluation-harness 0.4.13; pytest 8.4.1; Matplotlib 3.10.3.
- Standard-library HTTP server; HTML, CSS, browser JavaScript. No external inference APIs or hosted frontend assets.
- FineWeb-Edu sample-10BT and WikiText-103 training split. Official evaluation datasets: HellaSwag, ARC-Easy, PIQA, WinoGrande and WikiText-103 test. Exact revisions, attribution and licenses are recorded in the data manifest/card.
- NVIDIA GeForce RTX 4070, 12 GB VRAM, on the team's local computer. No paid cloud compute was purchased.
- Playwright and Chromium for browser verification and screenshots; FFmpeg for the demonstration recording.
- **OpenAI Codex**: AI-assisted architecture implementation, training pipeline, evaluation adapter, tests, interface, documentation, and independent code review. This disclosure concerns development assistance. Model weights begin randomly; model training does not use Codex answers or teacher logits as training targets.

## Research-only additions

Public [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), CDLA-Sharing-1.0, was used in later research candidates, not in the submitted v0.1.0 checkpoint. The separate research code and corpus manifests identify its source, revision and filtering. Those candidates were not promoted. See the [research appendix](../docs/research-summary.md). Git and GitHub were used for version control and artifact publication. The 2-minute-46-second demo is encoded with FFmpeg/libx264 and has burned-in English subtitles.

## Devpost Built With tags

Python, PyTorch, CUDA, NumPy, Hugging Face Tokenizers, Hugging Face Datasets, Transformers, Apache Arrow, lm-evaluation-harness, FineWeb-Edu, WikiText-103, TinyStories (research only), HTML, CSS, JavaScript, pytest, Matplotlib, Playwright, Chromium, FFmpeg, Git, GitHub, OpenAI Codex, NVIDIA RTX 4070, Intel Core i5-13400F. Use the full disclosure above in the description if a technology is unavailable as a Devpost tag.
