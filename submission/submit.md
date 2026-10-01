# Submit ODIN Pocket — Axiom AI

**Deadline:** October 1, 2026, **23:45 Singapore/Taipei time (UTC+8)**, or 15:45 UTC, according to the [official rules](https://gibc-v2.devpost.com/rules), checked October 1. Use this deadline; do not rely on an unconfirmed extension.

**Project:** ODIN Pocket
**Tagline:** A 49.3M-parameter language model, trained from scratch and inspectable on a CPU.
**Track:** Track 01: TECH — Foundational LLM Development
**Team:** Axiom AI
**Member:** Yong Li Zhong — solo developer

## Copy into Devpost

| Required field or upload | Ready material |
| --- | --- |
| Project description | Copy [project-description.md](project-description.md). It includes approach, results, challenges, lessons and AI disclosure. |
| Public source repository | https://github.com/Shoterz/odin-pocket |
| Model download | https://github.com/Shoterz/odin-pocket/releases/tag/v0.1.0 |
| Documentation/source release | https://github.com/Shoterz/odin-pocket/releases/tag/v0.1.1-submission |
| Built With | Copy the tags and full attribution in [built-with.md](built-with.md). Include OpenAI Codex. |
| Demo video | Upload [video/odin-pocket-demo.mp4](video/odin-pocket-demo.mp4), then paste the YouTube/Vimeo/Youku **watch URL**. [Title, description and upload instructions](video-upload.md). |
| Screenshots | Upload `screenshots/01-workbench.png`, `02-comparison.png` and `03-evidence.png`. [Captions](screenshots/README.md). `04-mobile.png` is optional. |
| Team member | Associate **Yong Li Zhong's actual Devpost account** with the entry; writing the name in the description alone does not add a member. |

## Remaining account steps

- [ ] Confirm registration for GIBC V2 and associate the solo member's Devpost account.
- [ ] Check the student's eligibility and the organizer's declarations in the actual form. These personal declarations must come from the participant.
- [ ] Upload the finished MP4 to an accepted host, choose Unlisted/Public, wait for processing, and verify signed-out playback.
- [ ] Create or update the project with the fields and three screenshots above; choose Track 01.
- [ ] Preview the public description, source link and embedded video.
- [ ] Submit the entry before the deadline and retain the confirmation/project URL.

**Preparation does not mean submission.** Registration, hosted video and final Devpost submission are not yet verified. GitHub-hosted video is useful for review but does not replace the required accepted-host URL. This checklist stays unchecked until those actions are confirmed.

## Claims and artifacts to preserve

The submitted model is **v0.1.0**, with SHA-256 `4634f90b2120a7a128a0a4bbd59ae70056dbeff7baee2a35bf551fe0da90ab7e`, 49,295,872 parameters and 983,040,000 processed training tokens. The video and README benchmark table describe those exact weights. Later candidate results belong in the [research appendix](../docs/research-summary.md), not in that benchmark table.

The model's recorded training loop took 3.923 hours on an RTX 4070. The broader project's available training summaries total 22.355 recorded hours; additional evaluation, preparation and unrecorded partial work are excluded. This distinction is documented for the efficiency criterion.

The demo is an English-captioned, silent recording of actual model operation, 2 minutes 46 seconds long. No instruction-following, factual reliability, state-of-the-art or competition-placement claim is made. The contribution is a complete scratch-trained compact model and a reproducible local inspection system.

For technical questions from judges, use [judge-guide.md](judge-guide.md), [the model card](evidence/model-card.md) and [the reproduction guide](../docs/reproduce.md).
