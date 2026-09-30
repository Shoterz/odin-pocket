# Continuation review rubric

Frozen before any new candidate generation was examined. Apply to every item in `results/controlled/blind-review.json` before reading its mapping. Report the reviewer identity: an assessment by Codex is not independent human validation.

Score each applicable dimension 0 (clear failure), 1 (mixed or ambiguous), or 2 (coherent and correct). Use null where a dimension does not apply. Report dimension means and clear-failure counts separately; do not hide a factual failure inside a combined fluency score. Empty or immediately terminated continuations fail relevance; an absent error in an empty answer is not a successful continuation.

- **Relevance:** Continues the supplied passage and develops its situation or explanation. Repeated filler, a disconnected topic, or an unexplained conversational format loses credit.
- **Entity consistency:** Maintains who did what, objects, locations, and narrative perspective. New characters or a clearly signaled perspective change are allowed. Unexplained name swaps, impossible ownership changes, and treating two different people as the same person are failures.
- **Causal/factual consistency:** Events follow coherently from the premise. Fiction may introduce fantasy if the narrative establishes it; anthropomorphism alone is not an error. Scientific explanations must preserve the relevant physical facts. Plausible grammar is not evidence of correctness.

## Prompt-specific checks

| Beginning | Checks |
|---|---|
| I saw my friend on the road today | Continue the encounter; maintain participants and perspective unless a transition explains the change. |
| Mira and her red umbrella | Preserve Mira, the umbrella, and its established location; a search or retrieval should respect the premise. |
| Sam and the leaking boat | Distinguish a hole/leak from unrelated problems such as a boat being stuck; any proposed remedy should address water entering. |
| Two friends sharing one apple | Preserve the two friends and the shared apple; use of a knife should lead to a coherent event, not unexplained duplication. |
| Nora looking for her dog | Preserve Nora and the lost dog; sounds or discoveries should follow coherently. |
| Missing the bus and leaving earlier | Connect leaving earlier with having more time to catch the bus. |
| Closed shop with a light inside | A coherent response to knocking; the prompt does not determine who is inside. |
| Seed in a pot | A coherent care or growth sequence. Avoid unexplained identity or object changes. |
| Ice melting | Warming can transfer energy to ice; melting is a phase change. Do not reward invented chemical explanations. |
| Bicycle brakes | Brakes oppose motion through friction; energy is dissipated, chiefly as heat. |
| Plants and sunlight | Light supplies energy for photosynthesis. Avoid reversing the basic process or treating sunlight as an ingested material. |
| Metal and wooden spoons | At the same room temperature, metal usually feels colder because it transfers heat from the hand faster. Do not claim the metal must have a lower temperature. |
| Bright Moon | Moonlight is principally reflected sunlight. |
| Lightning before thunder | Light travels much faster than sound through air. |
| Cup near a table edge | Move or support the cup so it is less likely to fall; do not reverse the goal. |
| Water freezing in a full glass bottle | Water expands on freezing and may break a constrained container. |

## Interpretation

These 16 locally authored prompts have already been used in earlier development. They are a diagnostic review set, not a new independent benchmark. Review all two-seed continuations from both confirmation models. Preserve verbatim text, neutral item IDs, scores, explanations, and mapping. Explicitly report ties and uncertainty. A few attractive samples do not establish general reasoning ability or justify a release replacement.
