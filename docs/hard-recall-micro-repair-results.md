# Hard Recall Micro Repair

This page records the first targeted hard-recall micro-repair continuation that
actually improved the held-out hard slice without regressing medium quality.

## Goal

Take the hard-error mining result seriously:

- focus only on the missed-defect recall bottleneck
- keep strict execution perfect
- use a tiny continuation so we do not wash out the already-good medium policy

## Run Setup

- Base adapter:
  [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)
- HF Jobs run:
  [69ed269fd70108f37acdef6d](https://huggingface.co/jobs/prodigyhuh/69ed269fd70108f37acdef6d)
- Hardware: `a10g-large`
- Profile: `hard_recall_micro_repair`
- Dataset mix:
  - `submit_after_reference`: `12`
  - `submit_prior`: `4`
- Updates: `4`
- Learning rate: `1e-6`
- Held-out eval band: `10000-10031`

## Final Comparison Vs Previous Best

The previous best published adapter was:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)

The winning checkpoint from this run was:

- `checkpoint-1`

| Metric | Previous best | `checkpoint-1` | Delta |
| --- | ---: | ---: | ---: |
| `medium_reward` | `4.5065` | `4.5065` | `0.0000` |
| `medium_f1` | `0.7891` | `0.7891` | `0.0000` |
| `hard_reward` | `4.6917` | `4.7148` | `+0.0231` |
| `hard_f1` | `0.8162` | `0.8207` | `+0.0045` |
| `strict_tool_call_pass_rate` | `1.00` | `1.00` | `0.00` |
| `normalized_tool_call_pass_rate` | `1.00` | `1.00` | `0.00` |
| `tool_failure_rate` | `0.00` | `0.00` | `0.00` |
| `done_rate` | `1.00` | `1.00` | `0.00` |

## Checkpoint Sweep

This run also showed that the gain is localized and early:

- `checkpoint-1`: improved hard quality without hurting medium
- `checkpoint-2`: regressed medium
- `checkpoint-4`: fell back to base

So the useful lesson is not "more training helps." It is:

- a *very small*, *very targeted* hard-recall continuation can help
- the sweet spot is early

## Interpretation

This is the first continuation in the recent series that meets the promotion
bar:

- strict execution stayed perfect
- normalized execution stayed perfect
- medium quality did not regress
- hard reward improved
- hard F1 improved

That makes `checkpoint-1` the new best measured result from this training family.

## Promotion Decision

- Run status: **completed**
- Run quality: **improved**
- Winning checkpoint: **`checkpoint-1`**
- Publication status: **pending**
- Target model repo:
  [prodigyhuh/atomicvision-hard-recall-micro-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-hard-recall-micro-boost-lora)
- Publish blocker: Hugging Face LFS upload returned `403 Forbidden` for the
  current token, so the adapter could not be materialized on the Hub yet

## Stored Artifact

The machine-readable summary for this run is committed at:

- [hard-recall-micro-repair-metrics.json](hard-recall-micro-repair-metrics.json)
