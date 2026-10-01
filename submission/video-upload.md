# Demo upload

Upload **video/odin-pocket-demo.mp4** to YouTube, Vimeo or Youku. The video is 2 minutes 46 seconds, H.264, 1440 × 1140, with English captions burned into the image. It is intentionally silent; English subtitles satisfy the rules. Optional separate subtitle file: `video/demo.en.srt`.

Set visibility to **Unlisted** or **Public**, not Private. Wait for processing, then check playback while signed out and paste the watch URL into Devpost. A GitHub MP4 download does not replace the required video-hosting link. No hosted video URL has yet been verified.

## Suggested title

ODIN Pocket — A 49.3M-parameter language model trained from scratch | GIBC V2

## Suggested description

ODIN Pocket is a 49,295,872-parameter language model trained from random initialization on one NVIDIA RTX 4070 12 GB. This demonstration shows actual local CPU generation, token probabilities, continuation scoring and checkpoint-matched training/benchmark evidence.

Team: Axiom AI. Solo developer: Yong Li Zhong.
Track 01: Foundational LLM Development, Global Innovation Build Challenge V2.

Source and setup: https://github.com/Shoterz/odin-pocket
Model weights: https://github.com/Shoterz/odin-pocket/releases/tag/v0.1.0

The demonstrated checkpoint processed 983,040,000 tokens in 3.923 recorded training hours. This is the submitted model's training run, not the total project experiment budget. The repository documents subsequent research separately.

The model is an English base next-token predictor. Outputs can repeat or contain false statements. Likelihood preferences are not fact checks. No hosted inference API, pretrained model initialization or private teacher targets are used. OpenAI Codex assisted development, testing, documentation and review; all tools and datasets are credited in the repository.

English subtitles are embedded. The video has no spoken audio.
