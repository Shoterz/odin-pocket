# ODIN Pocket — demo recording outline

Completed recording: [video/odin-pocket-demo.mp4](video/odin-pocket-demo.mp4), 2 minutes 46 seconds, with burned-in English captions. It shows the actual submitted v0.1.0 checkpoint, including unsuccessful output. [Hosted YouTube version](https://youtu.be/OxnweVSt72s): 2m02s, unlisted; [metadata verification](evidence/hosted-video-verification.json). The timings below are the original recording outline, not exact chapter boundaries.

0:00–0:25 — Explain the constraint: an entire language model, including its vocabulary and output head, under 50 million trainable parameters. ODIN Pocket has 49,295,872. It starts from random weights and runs locally.

0:25–1:05 — Continue a real passage using the default science prompt. Show the generated text, latency and actual token probabilities. Identify it as a base model, not a chatbot. Use the identical interaction a judge can reproduce.

1:05–1:40 — Compare two endings for “Water left in a freezer will eventually”. Show both scores, explain length normalization and that likelihood does not prove factual correctness. If the model chooses incorrectly, explain that failure rather than replacing the result.

1:40–2:25 — Open evidence. Show measured parameter count, training tokens, time and hardware, the real development loss curve, and all five official benchmark results. Say the exact measured values, distinguish accuracy from perplexity, and show full coverage. Never claim smoke results are final.

2:25–3:00 — Show checkpoint identity, public-data provenance, random initialization, no distillation and local inference. Explain the technical contribution: a complete reproducible small-model training and inspection system under a real consumer-GPU budget. State limits: model scale, lack of instruction training and imperfect contamination detection. Finish on source/reproduction instructions and the actual public link when available.
