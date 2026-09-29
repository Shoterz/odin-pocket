# Submission handoff — Axiom AI

Project: **ODIN Pocket**
Tagline: **A 49.3M-parameter language model, built from scratch and inspectable on a CPU.**
Track: TECH — Foundational LLM Development
Team: **Axiom AI**
Solo developer: **Yong Li Zhong**

## Ready to review locally

- Description: [project-description.md](project-description.md)
- Built With and AI disclosure: [built-with.md](built-with.md)
- Model/results: [evidence/model-card.md](evidence/model-card.md)
- Source archive: `odin-pocket-source.zip`
- Inference weights: `../runs/pocket/submission.pt` (distribute separately from source)
- English-captioned demonstration: `video/odin-pocket-demo.mp4`
- Screenshots: `screenshots/01-workbench.png`, `02-comparison.png`, `03-evidence.png`; optional mobile screenshot `04-mobile.png`

## External steps still unverified

Publish the source repository and provide a downloadable checkpoint. Upload the English demo to YouTube, Vimeo or Youku and copy its link. Confirm the Devpost team registration and associate Yong Li Zhong with Axiom AI. Add the description, complete Built With list, at least three screenshots, repository link and hosted video to Track 01. Review the entry before submitting.

No public repository, hosted video or Devpost submission is claimed by these local files. Rules: https://gibc-v2.devpost.com/rules . The rules list October 1 at 23:45 UTC+8; an update lists October 2 at 23:59. Use the earlier deadline unless the organizer confirms otherwise.

## Claims to preserve

49,295,872 parameters; 983,040,000 processed tokens (repeated sampled windows); 3.923 recorded training hours on an RTX 4070; all weights randomly initialized. The final benchmark scores are in the model card. WinoGrande is about chance and generated facts remain unreliable. CPU speed is a measured single-host workload, not a universal guarantee. Do not call this a proven winning or state-of-the-art model.
