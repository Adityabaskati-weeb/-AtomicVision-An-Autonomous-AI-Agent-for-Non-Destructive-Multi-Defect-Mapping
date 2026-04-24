# Phase 14 Held-Out Evaluation And GRPO Roadmap

## Decision

AtomicVision should now focus on GRPO, but only after one guardrail step:
prove the promoted SFT adapter generalizes on held-out seeds and harder
difficulties.

If a fresh SFT run logs `loss nan`, this roadmap is paused. Discard that
adapter and run the NaN-safe recovery path in
[`phase-15-nan-safe-sft-recovery.md`](phase-15-nan-safe-sft-recovery.md)
before evaluating or starting GRPO.

The current adapter is good enough for the demo package:

- Base model: `Qwen/Qwen3-1.7B`
- Method: cost-aware assistant-masked QLoRA SFT
- Dataset size: `512`
- LoRA rank: `16`
- LoRA alpha: `32`
- Optimizer updates: `80`
- Best checkpoint: `checkpoint-40`
- Medium reward: `4.47530128125`
- Medium F1: `0.79107153125`
- Scan cost: `1.5`
- Tool failure rate: `0.0`

The next model-improvement phase is not more SFT by default. It is:

1. Held-out eval.
2. GRPO variance probe.
3. Short GRPO continuation.
4. Promote only if held-out metrics improve without increasing tool failures.

## Why This Order

The earlier GRPO continuation failed for a useful reason: grouped completions
collapsed to the same reward, so GRPO had no relative signal. TRL logs this as
`reward_std=0`, `frac_reward_zero_std=1`, `loss=0`, and `grad_norm=0`.

The SFT-vs-GRPO lesson is simple: use SFT for examples that can be filtered or
demonstrated cleanly, then use GRPO only where the model must explore its own
actions. Feeding GRPO easy deterministic cases is inefficient and can regress
the model by optimizing format instead of correctness.

The promoted SFT adapter fixed tool formatting and cheap prior copying. GRPO
should now be used to improve the cases where the prior is weak or a single
reference comparison genuinely helps. It should not be allowed to destroy the
cheap reliable behavior that already works.

The GRPO script now supports this directly with:

```bash
--prompt-focus grpo-frontier
```

That mode scans deterministic AtomicVision seeds and keeps only cases where the
prior confidence is borderline or where an oracle map after one reference
comparison would beat prior-copy by the configured reward margin. The
cost-aware GRPO presets use this mode automatically, starting at seed `2000` so
they do not replay the original 512 SFT seeds.

## Held-Out Gate

Before spending HF credits, evaluate the promoted adapter on:

- `medium`, seeds `1000-1031`
- `hard`, seeds `1000-1031`
- low-confidence prior cases
- reference-improvement cases where one `compare_reference` improves F1 enough
  to pay for the `0.5` cost

Pass criteria:

- `tool_failure_rate <= 0.02`
- `done_rate >= 0.98`
- `mean_scan_cost <= 1.75`
- `strict_tool_call_pass_rate` is healthy, or normalized evaluation proves the
  remaining gap is formatting rather than policy
- `first_action_ask_prior_rate` stays high
- `submit_action_rate` stays high
- medium held-out reward does not fall below the prior-submit baseline
- hard held-out reward is competitive with prior-submit

If this fails, do not run GRPO yet. Generate a better cost-aware SFT dataset
with more held-out and hard examples, then rerun the official evaluator:

```bash
python training/evaluate_atomicvision_adapter.py \
  --adapter-dir /kaggle/working/atomicvision-format-submit-merged-lora \
  --base-model Qwen/Qwen3-1.7B \
  --difficulties medium hard \
  --episodes 32 \
  --seed-start 1000 \
  --output-json /kaggle/working/atomicvision_adapter_eval.json
```

## GRPO Focus

Use the promoted SFT adapter as the initialization. GRPO should target only the
decision boundary:

- submit high-confidence prior immediately
- use at most one cheap extra evidence call for borderline priors
- avoid high-cost scans unless the prior is missing or clearly unreliable
- stop after `submit_defect_map`

Recommended first probe:

```bash
python training/train_grpo_atomicvision.py \
  --preset cost-aware-variance-probe \
  --adapter-model-id /kaggle/working/atomicvision-best-cost-aware-masked-sft-lora \
  --report-to none
```

Equivalent explicit form:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id /kaggle/working/atomicvision-best-cost-aware-masked-sft-lora \
  --samples 32 \
  --seed-start 2000 \
  --prompt-focus grpo-frontier \
  --max-seed-candidates 1024 \
  --max-steps 3 \
  --num-generations 8 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-6 \
  --scale-rewards batch \
  --loss-type dapo \
  --report-to none
```

Continue only if:

- `reward_std > 0`
- `frac_reward_zero_std < 1`
- `grad_norm > 0`
- `atomicvision/post_terminal_tool_calls_mean = 0`
- `atomicvision/done_rate` stays near `1`

Recommended short continuation:

```bash
python training/train_grpo_atomicvision.py \
  --preset cost-aware-grpo-20 \
  --adapter-model-id /kaggle/working/atomicvision-best-cost-aware-masked-sft-lora \
  --report-to trackio \
  --output-dir /kaggle/working/atomicvision-cost-aware-grpo-20-lora
```

Recommended HF-credit continuation after the 20-step run passes:

```bash
python training/train_grpo_atomicvision.py \
  --preset cost-aware-grpo-100 \
  --adapter-model-id prodigyhuh/atomicvision-best-cost-aware-masked-sft-lora \
  --report-to trackio \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-cost-aware-grpo-lora
```

## Reward Monitoring

The GRPO script now logs AtomicVision-specific metrics when TRL provides
`log_metric`:

- `atomicvision/env_reward_mean`
- `atomicvision/format_reward_mean`
- `atomicvision/prior_copy_reward_mean`
- `atomicvision/done_rate`
- `atomicvision/post_terminal_tool_calls_mean`
- `atomicvision/strict_tool_call_pass_rate`
- `atomicvision/normalized_tool_call_pass_rate`
- `atomicvision/normalized_tool_call_repair_rate`
- `atomicvision/ask_prior_tool_rate`
- `atomicvision/submit_tool_rate`
- `atomicvision/total_reward_mean`

These should be read together with TRL's own GRPO metrics:

- `reward`
- `reward_std`
- `frac_reward_zero_std`
- `grad_norm`
- `completions/mean_length`
- per-reward-function means

Also watch the reward-source means now exposed by the training stack:

- `atomicvision/identity_reward_mean`
- `atomicvision/concentration_reward_mean`
- `atomicvision/confidence_reward_mean`
- `atomicvision/outcome_reward_mean`
- `atomicvision/false_positive_penalty_mean`
- `atomicvision/missed_defect_penalty_mean`
- `atomicvision/scan_cost_penalty_mean`
- `atomicvision/timeout_penalty_mean`
- `atomicvision/penalty_total_mean`
- `atomicvision/process_shaping_reward_mean`

These make it easier to tell whether a run is improving the real scientific
outcome, merely getting cheaper, or just over-optimizing process shaping.

## Stop Rules

Stop the run immediately if any of these happen:

- any SFT loss is `nan` or `inf`
- `frac_reward_zero_std` stays at `1`
- `grad_norm` stays at `0`
- tool failures exceed `5%`
- scan cost rises above `2.0`
- the model calls tools after terminal submission
- held-out reward drops below the promoted SFT adapter

## Promotion Rule

Promote a GRPO adapter only if it beats the promoted SFT checkpoint on held-out
evaluation, not just training reward.

Minimum promotion bar:

- higher mean reward than `4.47530128125` on comparable medium eval, or clear
  hard-difficulty improvement
- no increase in tool failure rate
- no meaningful scan-cost increase
- better behavior on weak-prior/reference-improvement cases

If GRPO improves training reward but fails this gate, keep it as an ablation and
ship the promoted SFT adapter.
