# GIBC final submission preparation

> **For agentic workers:** Use superpowers:executing-plans for this documentation and release task.

**Goal:** Prepare and publish the complete, verified ODIN Pocket submission package for Axiom AI / Yong Li Zhong.

**Architecture:** Preserve the public v0.1.0 checkpoint and runtime. Update documentation, include later research evidence separately, verify downloadable artifacts, and publish a final source/evidence release. Account-bound video hosting and Devpost submission remain pending until their URLs and completion are verified.

**Tech stack:** Existing Python/PyTorch project, Git/GitHub, JSON evidence, FFmpeg and Markdown.

**Spec:** User request to prepare documentation, repository and everything required by the [official rules](https://gibc-v2.devpost.com/rules), checked October 1, 2026.

## Constraints and review focus

- Submitted checkpoint SHA-256 remains `4634f90b2120a7a128a0a4bbd59ae70056dbeff7baee2a35bf551fe0da90ab7e`; 49,295,872 parameters; randomly initialized training.
- Published model scores must never be replaced by research scores from other weights or development splits.
- Distinguish final-model training cost from the additional experiment budget.
- Required assets: public source/setup, description, hosted 2–5-minute English demo, Built With/AI disclosure, real team identity, at least three screenshots.
- Repository/video access, checkpoint hashes, relative links and archive contents must be checked; no unverified submission claims.
- No training, runtime changes, benchmark selection or new capability claims. Documentation-only changes do not need new unit tests; use existing tests and direct artifact checks.

## Tasks

- [x] Audit existing assets and validate the published checkpoint, full reports, tests and representative video frames.
- [x] Update README, model/data/reproduction documentation, Devpost text, upload instructions and research/compute appendix.
- [ ] Independently review the assembled package; verify archive contents, three screenshots, English captions, model loading and actual inference.
- [ ] Publish source/evidence to the authorized GitHub repository and verify links; assemble an upload bundle and record any account-bound steps still pending.

Execution uses an isolated worktree based on `f251662`. Publishing to Shoterz/odin-pocket is already authorized by the user. The immutable older release is retained. No permission checkpoint is required for routine preparation or the already-authorized publication.
