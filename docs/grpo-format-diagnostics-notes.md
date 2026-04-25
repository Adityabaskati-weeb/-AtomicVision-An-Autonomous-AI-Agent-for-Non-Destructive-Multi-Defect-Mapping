# GRPO Format Diagnostics Notes

These notes capture the two short hard-only GRPO probes run after we stopped guessing and added explicit completion-format diagnostics.

## V10b: Strip-think on Main Baseline

- Job: [69ec8952d70108f37acde3a5](https://huggingface.co/jobs/prodigyhuh/69ec8952d70108f37acde3a5)
- Commit: `42b25cf919fedd57e3522fbc5262d04a47c0ef01`
- Metrics: [hard-only-grpo-reference-probe-stripthink-main-v10b-metrics.json](./hard-only-grpo-reference-probe-stripthink-main-v10b-metrics.json)

Key result:

- The parser fix preserved the healthier baseline behavior:
  - `done_rate = 0.75`
  - `submit_tool_rate = 0.25`
  - `normalized_tool_call_pass_rate = 1.0`
  - `strict_tool_call_pass_rate = 0.0`

Diagnostic result:

- `raw_tool_call_tag_rate = 0.0`
- `repaired_without_tool_tags_rate = 1.0`
- `stripped_think_wrapper_rate = 0.0`

Interpretation:

- The dominant strict-format failure is **not** empty Qwen `<think>` wrappers.
- The model is usually emitting **tagless repaired outputs** such as a bare tool name plus arguments.

## V11b: Stronger Penalty for Tagless Repaired Outputs

- Job: [69ec8b0fd70108f37acde3b9](https://huggingface.co/jobs/prodigyhuh/69ec8b0fd70108f37acde3b9)
- Commit: `8e2aa2b1191539c697baf914a98690a9d8c40b64`
- Metrics: [hard-only-grpo-reference-probe-tagless-penalty-v11b-metrics.json](./hard-only-grpo-reference-probe-tagless-penalty-v11b-metrics.json)

Key result:

- The stronger tagless penalty changed the diagnostic mix, but not the main gate metrics:
  - `done_rate = 0.75`
  - `submit_tool_rate = 0.25`
  - `normalized_tool_call_pass_rate = 1.0`
  - `strict_tool_call_pass_rate = 0.0`

Diagnostic result:

- `raw_tool_call_tag_rate = 0.25`
- `repaired_without_tool_tags_rate = 0.75`
- `repaired_with_tool_tags_rate = 0.25`

Interpretation:

- Penalizing tagless repaired outputs more strongly is **not enough by itself** in a 4-step probe.
- The model begins to show some wrapper usage, but it does not convert into strict final submissions yet.

## Current Conclusion

The next meaningful strict-format intervention should target **generation behavior**, not just parsing or reward accounting. The evidence so far says:

1. wrapper stripping was a correct parser hardening fix
2. the live failure mode is mostly tagless repaired output
3. a stronger penalty alone does not produce strict tool calls in a short GRPO continuation

## V12: Tiny Format-Refresh SFT Warmup Then Same GRPO Probe

- Job: [69ec8e88d70108f37acde3e5](https://huggingface.co/jobs/prodigyhuh/69ec8e88d70108f37acde3e5)
- Commit: `7b0580ef558469f139b19968717ecbb796db74f6`
- Metrics: [hard-only-grpo-after-format-refresh-v12-metrics.json](./hard-only-grpo-after-format-refresh-v12-metrics.json)

Setup:

- Added a reproducible `format_refresh` SFT profile.
- Generated a tiny hard-only strict-envelope warmup set at `seed_start=3600`.
- Ran a short SFT continuation from `atomicvision-medium-fidelity-boost-lora`.
- Ran the same 4-step hard GRPO probe from the warmed-up adapter.

Key result compared with V11b:

- `reward`: `2.58 -> 2.90`
- `reward_std`: `0.51 -> 1.02`
- `done_rate`: `0.75 -> 0.78125`
- `normalized_tool_call_pass_rate`: stayed `1.0`
- `strict_tool_call_pass_rate`: stayed `0.0`
- `submit_tool_rate`: `0.25 -> 0.21875`

Interpretation:

- The tiny SFT warmup improved the **usefulness of the GRPO signal** and the total reward.
- It did **not** fix the strict XML-wrapped final tool-call issue yet.
- The failure mode still looks generation-side: the model remains mostly in the repairable, non-strict regime.

Practical takeaway:

- The `format_refresh` SFT warmup is directionally helpful as a staging step before GRPO.
- It is not sufficient by itself to greenlight a longer GRPO continuation.

## V13e: Generation-Side XML Sequence Bias

- Job: [69ec9546d70108f37acde439](https://huggingface.co/jobs/prodigyhuh/69ec9546d70108f37acde439)
- Commit: `dff47640662027b9fac40f78f70a5964140da359`
- Metrics: [hard-only-grpo-sequence-bias-v13e-metrics.json](./hard-only-grpo-sequence-bias-v13e-metrics.json)

Setup:

- Kept the same tiny `format_refresh` warmup recipe as V12.
- Added an opt-in GRPO generation constraint using `GenerationConfig.sequence_bias`.
- Biased prefixes of `<tool_call>`, `<tool_call>{"name":"`, and `</tool_call>` with a positive bias of `2.0`.
- This follows the official Transformers guidance that multi-token sequence biasing works better when prefixes are biased too, and uses TRL's `GRPOConfig.generation_kwargs` pass-through to reach generation.

Key result compared with V12:

- `submit_tool_rate`: `0.21875 -> 0.25`
- `raw_tool_call_tag_rate`: `0.21875 -> 0.25`
- `strict_tool_call_pass_rate`: stayed `0.0`
- `done_rate`: `0.78125 -> 0.75`
- `reward`: `2.90 -> 2.71`
- `reward_std`: `1.02 -> 0.45`

Interpretation:

- The XML sequence bias slightly improved wrapper usage and submit frequency.
- It did **not** convert those gains into strict final tool calls.
- It also reduced reward diversity and total reward relative to the better V12 baseline.

Practical takeaway:

- Generation-side sequence bias alone is **not** the missing strict-XML fix for this model/setup.
- The best baseline remains V12: tiny `format_refresh` warmup plus the un-biased short GRPO probe.

## V15: Structured Tool-Call Strict Refresh Warmup

- Job: [69eca7e4d2c8bd8662bcd8c5](https://huggingface.co/jobs/prodigyhuh/69eca7e4d2c8bd8662bcd8c5)
- Commit: `8871df85331e27444b4e6c117f8c5fecb49d122e`
- Metrics: [hard-only-grpo-structured-toolcall-v15-metrics.json](./hard-only-grpo-structured-toolcall-v15-metrics.json)

Setup:

- Reworked the strict refresh warmup so the assistant target is stored as a
  structured `tool_calls` message instead of a literal `<tool_call>...</tool_call>`
  string.
- Updated SFT validation and target rendering to let the tokenizer chat template
  produce the final assistant tool envelope.
- Ran the same tiny hard-only strict refresh warmup and the same 4-step hard
  GRPO probe as V12, but with a pre-downloaded local base-model snapshot so the
  run could not fail on late Hub auth.

Key result compared with V12:

- `reward`: `2.90 -> 2.46`
- `reward_std`: `1.02 -> 0.52`
- `done_rate`: `0.78125 -> 0.75`
- `submit_tool_rate`: `0.21875 -> 0.25`
- `raw_tool_call_tag_rate`: `0.21875 -> 0.25`
- `normalized_tool_call_pass_rate`: stayed `1.0`
- `strict_tool_call_pass_rate`: stayed `0.0`

Interpretation:

- The local-snapshot fix solved the HF Jobs infrastructure problem cleanly: the
  run completed end to end and produced a persistent summary.
- The structured-tool-call warmup did **not** improve strict XML submission on
  the short GRPO probe.
- It slightly increased raw wrapper usage and submit frequency, but at the cost
  of lower reward and lower reward variance than the better V12 baseline.

Practical takeaway:

- Teaching structured assistant tool calls in the warmup is compatible with the
  training stack, but it is **not** the missing strict-XML fix by itself.
- V12 remains the better behavioral baseline for the next iteration.
