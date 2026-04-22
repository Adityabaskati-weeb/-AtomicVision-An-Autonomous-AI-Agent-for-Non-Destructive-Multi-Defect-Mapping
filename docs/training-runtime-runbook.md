# Training Runtime Runbook

This runbook is the operational guide for AtomicVision GRPO training across
Colab, Kaggle, and Hugging Face runtimes.

## Current Safe Smoke Configuration

Use this first everywhere:

```bash
python training/train_grpo_atomicvision.py --preset smoke --report-to trackio
```

Why this is conservative:

- `num-generations=4` gives GRPO a larger comparison group while keeping
  rollout concurrency manageable.
- `per-device-train-batch-size=1` and `gradient-accumulation-steps=4` make
  `generation_batch_size=4`, which is required to be divisible by
  `num_generations=4`.
- `temperature=1.2`, `top_p=0.95`, and `top_k=50` add enough sampling diversity
  to detect collapsed zero-advantage runs.
- The hosted Space supports 32 concurrent WebSocket sessions; first smoke runs
  should stay far below that.

## Next Colab Run

After the 5-step smoke run reaches `100% 5/5`, run a longer Qwen3-0.6B job:

```bash
python training/train_grpo_atomicvision.py --preset colab-20 --report-to trackio
```

Watch these metrics:

- `tools/failure_frequency`: should decrease as invalid tool calls reduce.
- `rewards/reward_func/mean`: should move upward toward zero, then positive.
- `reward_std`: must be nonzero for GRPO to update the policy.
- `frac_reward_zero_std`: should be below `1.0`; `1.0` means every group has no
  relative reward signal.
- `grad_norm`: should become nonzero once advantages are nonzero.
- `tools/call_frequency`: should become less chaotic after the model learns to
  call `ask_prior` and submit.

## Larger Model Path

The Qwen model ladder:

- `Qwen/Qwen3-0.6B`: about 751.6M parameters. Best for smoke and Colab.
- `Qwen/Qwen3-1.7B`: about 2.03B parameters. Best next step for hackathon GPU
  credit.
- `Qwen/Qwen3-4B`: stronger, use only after 1.7B is stable on paid/HF GPU.
- `Qwen/Qwen3-8B`: about 8.19B parameters. Use A100/H100-class hardware only.

Recommended 1.7B run:

```bash
python training/train_grpo_atomicvision.py --preset qwen-1p7b-50 --report-to trackio
```

Recommended SFT-copy variance probe:

```bash
python training/train_grpo_atomicvision.py \
  --preset variance-probe \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --report-to none \
  --run-name atomicvision-sft-copy-grpo-variance-probe
```

Continue only if the probe produces nonzero `reward_std`, nonzero `grad_norm`,
and `frac_reward_zero_std < 1`.

Recommended variance-aware SFT-copy continuation run:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --samples 128 \
  --max-steps 50 \
  --num-generations 8 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-completion-length 768 \
  --temperature 1.2 \
  --top-p 0.95 \
  --top-k 50 \
  --learning-rate 1e-6 \
  --report-to trackio \
  --run-name atomicvision-sft-copy-grpo-variance-50step \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-variance-lora
```

Use a smaller learning rate than fresh LoRA GRPO because the SFT-copy adapter
already matches the prior-submit behavior. The continuation goal is to improve
weak-prior decisions without damaging exact tool-copy reliability. The script
now includes the tool-system prompt by default, shapes reward for valid tool
JSON, and uses only weak exact-copy shaping for high-confidence priors.

Recommended HF-credit 4B run, only after 1.7B works:

```bash
python training/train_grpo_atomicvision.py --preset hf-4b-100 --report-to trackio
```

## Colab

Use a clean GPU runtime:

```bash
git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision
cd AtomicVision
pip install -r training/requirements-grpo.txt
python training/train_grpo_atomicvision.py --dry-run --difficulty easy
```

If you already cloned:

```bash
cd /content/AtomicVision
git pull
pip install -r training/requirements-grpo.txt
```

Do not install Torch manually. Colab provides a CUDA-matched Torch build.
Replacing it can break torchvision/torchaudio.

## Kaggle

Enable GPU in notebook settings, then:

```bash
git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision
cd AtomicVision
pip install -r training/requirements-grpo.txt
python training/train_grpo_atomicvision.py --dry-run --difficulty easy
```

Kaggle often has stricter internet/session behavior. If a Space connection
fails during training, rerun with the safe smoke configuration. If
`reward_std=0`, do not extend the run; increase sampling diversity first.

Kaggle SFT-copy variance probe:

```bash
python training/train_grpo_atomicvision.py \
  --preset variance-probe \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --report-to none \
  --run-name atomicvision-kaggle-sft-copy-grpo-variance-probe
```

Kaggle variance-aware SFT-copy continuation smoke:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --samples 64 \
  --max-steps 20 \
  --num-generations 8 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-completion-length 768 \
  --temperature 1.2 \
  --top-p 0.95 \
  --top-k 50 \
  --learning-rate 1e-6 \
  --report-to none \
  --run-name atomicvision-kaggle-sft-copy-grpo-variance-20step \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-variance-smoke-lora
```

Your latest mid-exploration smoke produced valid tools and nonzero train-time
variance, but the direct eval still collapsed to the prior-submit baseline:
reward `3.7664260625` in eval-sum style, F1 `0.773214375`, MAE
`0.0317914375`, scan cost `1.5`, and tool failures `0.0`. Do not promote that
adapter. The next move is a supervised bridge that teaches useful one-scan
behavior before another GRPO continuation.

## Curated SFT Bridge

The earlier SFT-copy dataset only taught `ask_prior -> submit_defect_map`, so
the model learned to copy the prior even when exploration could help. The tool
wrapper now exposes compact spectral summaries:

- `current_peak_freqs`
- `candidate_signature_bands`
- `spectrum_delta_top_abs` after `compare_reference`
- `candidate_signature_scores` after `compare_reference`

The first mixed reference-heavy SFT run fixed much of the tool-call collapse,
but it over-used extra evidence tools. The next dataset should therefore be
cost-aware: mostly cheap `ask_prior -> submit_defect_map` rows, with only a
small number of curated `compare_reference` rows.

Recommended Kaggle data generation command:

```bash
python training/generate_atomicvision_sft_data.py \
  --profile cost_aware \
  --episodes-per-difficulty 512 \
  --difficulties medium \
  --submit-prior-ratio 0.85 \
  --reference-ratio 0.10 \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl
```

Expected sample mix for 512 medium rows:

- `submit_prior`: about 435 rows
- `submit_after_reference`: about 51 rows
- `ask_prior`: about 26 rows

Use this dataset for the next assistant-masked SFT LoRA. Keep the first run
local/no-push. Save checkpoints around steps 40, 60, and 80, then directly
evaluate each checkpoint before any GRPO continuation.

Generate reference-improvement SFT rows on Kaggle:

```bash
python training/generate_atomicvision_sft_data.py \
  --episodes-per-difficulty 256 \
  --difficulties medium \
  --sample-types submit_after_reference \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_reference_improvement_sft.jsonl
```

For a reference-heavy ablation dataset that keeps exact prior-copy behavior
while adding many one-reference examples:

```bash
python training/generate_atomicvision_sft_data.py \
  --episodes-per-difficulty 256 \
  --difficulties medium \
  --sample-types submit_prior submit_after_reference \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_mixed_copy_reference_sft.jsonl
```

Do not use this reference-heavy mix as the default next run; it is useful only
as an ablation if the cost-aware profile fails to improve. Only after direct
eval beats the current SFT-copy adapter should you run another variance-aware
GRPO continuation.

## Hugging Face Training

Use Hugging Face Jobs only when a paid Jobs-capable account/token is available.
The same script can run there, but results are ephemeral unless pushed to the
Hub. For final runs, add:

```bash
--push-to-hub \
--hub-model-id prodigyhuh/atomicvision-qwen3-0.6b-grpo-lora
```

Do not paste tokens into notebooks or chat. Use notebook secrets or HF Jobs
secrets.

Example HF Jobs command for hackathon credits:

```bash
hf jobs run \
  --flavor a10g-small \
  --timeout 2h \
  --secrets HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc "git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision && cd AtomicVision && pip install -r training/requirements-grpo.txt && python training/train_grpo_atomicvision.py --preset qwen-1p7b-50 --report-to trackio --run-name atomicvision-hfjob-1p7b-50step --push-to-hub --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-grpo-lora"
```

Example HF Jobs command for SFT-copy GRPO continuation:

```bash
hf jobs run \
  --flavor a10g-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc "git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision && cd AtomicVision && pip install -r training/requirements-grpo.txt && python training/train_grpo_atomicvision.py --model Qwen/Qwen3-1.7B --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora --samples 256 --max-steps 200 --num-generations 8 --per-device-train-batch-size 1 --gradient-accumulation-steps 8 --max-completion-length 768 --temperature 1.2 --top-p 0.95 --top-k 50 --learning-rate 1e-6 --report-to trackio --run-name atomicvision-hf-sft-copy-grpo-variance-200step --push-to-hub --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-variance-lora"
```

For 4B, use `a10g-large` or better and a longer timeout.

## Scaling Ladder

Only scale after the previous run reaches at least one completed training step.

1. Smoke: `num-generations=4`, batch `1`, accumulation `4`, steps `5`.
2. Variance probe: 1.7B continuation, `num-generations=8`, batch `1`,
   accumulation `8`, steps `3`.
3. Demo: 1.7B, `num-generations=8`, batch `1`, accumulation `8`, steps `50`.
4. Stronger HF-credit run: 1.7B or 4B, `num-generations=8`, batch `1`,
   accumulation `8`, steps `100-300`.

## Common Failures

- `transformers version 5.2.0 or higher`: rerun
  `pip install -r training/requirements-grpo.txt`, restart runtime, rerun.
- CUDA mismatch between Torch and torchvision: delete runtime, reconnect GPU,
  reinstall requirements. Do not manually install Torch.
- `jmespath` missing: run `pip install jmespath` or reinstall requirements.
- `ConnectionClosedOK`: pull latest code and use the safe smoke configuration.
  It lowers rollout concurrency and retries reset connections.
- `loss=0`, `grad_norm=0`, `reward_std=0`, `frac_reward_zero_std=1`: the model
  produced identical-reward generations. Stop the run, increase sampling
  diversity, and do not promote that adapter.
