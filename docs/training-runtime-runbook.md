# Training Runtime Runbook

This runbook is the operational guide for AtomicVision GRPO training across
Colab, Kaggle, and Hugging Face runtimes.

## Current Safe Smoke Configuration

Use this first everywhere:

```bash
python training/train_grpo_atomicvision.py --preset smoke --report-to trackio
```

Why this is conservative:

- `num-generations=2` gives GRPO a comparison group while keeping rollout
  concurrency low.
- `per-device-train-batch-size=2` makes `generation_batch_size=2`, which is
  required to be divisible by `num_generations=2`.
- `gradient-accumulation-steps=1` avoids opening many simultaneous OpenEnv
  sessions during the first gate.
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

Recommended SFT-copy continuation run:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --samples 128 \
  --max-steps 50 \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --max-completion-length 768 \
  --learning-rate 5e-6 \
  --report-to trackio \
  --run-name atomicvision-sft-copy-grpo-50step \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-lora
```

Use a smaller learning rate than fresh LoRA GRPO because the SFT-copy adapter
already matches the prior-submit behavior. The continuation goal is to improve
weak-prior decisions without damaging exact tool-copy reliability.

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
fails during training, rerun with the safe smoke configuration and keep
`num-generations=2`.

Kaggle SFT-copy continuation smoke:

```bash
python training/train_grpo_atomicvision.py \
  --model Qwen/Qwen3-1.7B \
  --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora \
  --samples 64 \
  --max-steps 20 \
  --num-generations 2 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --max-completion-length 768 \
  --learning-rate 5e-6 \
  --report-to none \
  --run-name atomicvision-kaggle-sft-copy-grpo-20step \
  --push-to-hub \
  --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-smoke-lora
```

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
  bash -lc "git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision && cd AtomicVision && pip install -r training/requirements-grpo.txt && python training/train_grpo_atomicvision.py --model Qwen/Qwen3-1.7B --adapter-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-lora --samples 256 --max-steps 200 --num-generations 4 --per-device-train-batch-size 4 --gradient-accumulation-steps 1 --max-completion-length 768 --learning-rate 5e-6 --report-to trackio --run-name atomicvision-hf-sft-copy-grpo-200step --push-to-hub --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-sft-copy-grpo-lora"
```

For 4B, use `a10g-large` or better and a longer timeout.

## Scaling Ladder

Only scale after the previous run reaches at least one completed training step.

1. Smoke: `num-generations=2`, batch `2`, accumulation `1`, steps `5`.
2. Colab learning run: 0.6B, `num-generations=2`, batch `2`, steps `20`.
3. Demo: 1.7B, `num-generations=2`, batch `2`, steps `50`.
4. Stronger HF-credit run: 1.7B or 4B, `num-generations=4`, batch `4`,
   steps `100-300`.

## Common Failures

- `transformers version 5.2.0 or higher`: rerun
  `pip install -r training/requirements-grpo.txt`, restart runtime, rerun.
- CUDA mismatch between Torch and torchvision: delete runtime, reconnect GPU,
  reinstall requirements. Do not manually install Torch.
- `jmespath` missing: run `pip install jmespath` or reinstall requirements.
- `ConnectionClosedOK`: pull latest code and use the safe smoke configuration.
  It lowers rollout concurrency and retries reset connections.
