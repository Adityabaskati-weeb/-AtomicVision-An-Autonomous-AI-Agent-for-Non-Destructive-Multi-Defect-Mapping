# Training Runtime Runbook

This runbook is the operational guide for AtomicVision GRPO training across
Colab, Kaggle, and Hugging Face runtimes.

## Permanent Seed Policy

Use one seed split everywhere so held-out evaluation stays honest:

- SFT data generation: `1000-3999`
- GRPO prompt selection: `4000-7999`
- held-out evaluation only: `10000-10999`

If you need to compare against older overlapping runs, mark them as historical
debugging runs and do not use them for promotion.

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

## NaN SFT Failure Protocol

If any SFT run logs `loss nan`, that run is invalid even if it saved
checkpoints. Do not promote that adapter and do not start GRPO from it.

Use the NaN-safe trainer instead:

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl \
  --validate-only
```

Then run a tiny sanity train:

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --output-dir /kaggle/working/atomicvision-sft-sanity-lora \
  --max-examples 64 \
  --max-updates 5 \
  --checkpoint-steps 5 \
  --learning-rate 2e-5 \
  --max-grad-norm 1.0 \
  --overwrite-output-dir
```

Only if that prints finite losses should you run the full safe SFT:

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_cost_aware_masked_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --output-dir /kaggle/working/atomicvision-cost-aware-masked-sft-lora \
  --max-updates 80 \
  --grad-accum 8 \
  --batch-size 1 \
  --max-length 1536 \
  --learning-rate 2e-5 \
  --max-grad-norm 1.0 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --checkpoint-steps 40 60 80 \
  --overwrite-output-dir
```

The safe trainer aborts on malformed JSONL, zero assistant-label tokens,
non-finite loss, or non-finite gradient norm.

## Persist Adapters Immediately

Kaggle and Colab working directories are transient. After any adapter train you
want to keep, publish it to a Hugging Face **model repo** immediately. Do not
rely on the Space itself as adapter storage.

Recommended command:

```bash
python training/publish_adapter_to_hub.py \
  --adapter-dir /kaggle/working/atomicvision-format-submit-merged-lora \
  --repo-id prodigyhuh/atomicvision-format-submit-merged-lora \
  --base-model Qwen/Qwen3-1.7B \
  --include-zip
```

Why a model repo instead of a Space:

- model repos are the natural storage target for adapter artifacts,
- they are easier to download from Kaggle with one URL,
- the Space can still reference the model repo in its README or runtime config.

Before any GRPO continuation, run the official held-out evaluator in both
strict and normalized modes:

## Targeted Post-Recovery Booster

Once strict tool reliability is fixed, do not restart from the base model just
to chase a small held-out reward gap. Continue from the saved good adapter and
target the remaining quality pocket directly.

For AtomicVision, the current remaining gap is mostly medium-difficulty outcome
quality. A conservative next pass is:

1. build a medium-only cheap-prior dataset with a very high `submit_prior`
   ratio,
2. continue SFT from the saved reliable adapter,
3. keep the learning rate small,
4. re-run held-out eval immediately.

Example:

```bash
python training/generate_atomicvision_sft_data.py \
  --profile cost_aware \
  --episodes-per-difficulty 256 \
  --seed-start 2000 \
  --difficulties medium \
  --submit-prior-ratio 0.95 \
  --reference-ratio 0.02 \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_medium_prior_fidelity_sft.jsonl
```

Then continue from the saved adapter instead of from the base model:

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_medium_prior_fidelity_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --init-adapter-dir /kaggle/working/atomicvision-format-submit-merged-lora \
  --output-dir /kaggle/working/atomicvision-medium-fidelity-boost-lora \
  --max-updates 12 \
  --grad-accum 8 \
  --batch-size 1 \
  --max-length 1536 \
  --learning-rate 5e-6 \
  --max-grad-norm 1.0 \
  --checkpoint-steps 6 12 \
  --overwrite-output-dir
```

This follows the reward-engineering papers more closely:

- keep the good process policy,
- target the remaining outcome gap,
- do not over-correct with broad reward or format changes once strict execution
  already works.

This medium-only boost now exists as a published checkpoint:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)

Held-out result:

- medium baseline `4.5115` -> boost `4.5707`
- hard baseline `4.8883` -> boost `4.6466`
- strict tool-call pass rate `1.00`
- normalized tool-call pass rate `1.00`

So the medium intervention worked, but it did not move the hard frontier.
That matches the dataset mix: the boost run found only a handful of
`submit_after_reference` rows and therefore mostly strengthened prior fidelity
instead of harder reference-improvable cases.

## Next Hard-Frontier Booster

The next careful pass should continue from the promoted boost adapter and widen
the scan-improvement search on `hard` only.

Recommended data generation command:

```bash
python training/generate_atomicvision_sft_data.py \
  --profile hard_frontier_boost \
  --episodes-per-difficulty 256 \
  --seed-start 3000 \
  --difficulties hard \
  --max-scan-candidates-per-difficulty 1024 \
  --output-jsonl outputs/sft/atomicvision_hard_frontier_boost_sft.jsonl
```

Recommended continuation command:

```bash
python training/train_sft_atomicvision_safe.py \
  --dataset-jsonl outputs/sft/atomicvision_hard_frontier_boost_sft.jsonl \
  --model Qwen/Qwen3-1.7B \
  --init-adapter-dir /kaggle/working/atomicvision-medium-fidelity-boost-lora \
  --output-dir /kaggle/working/atomicvision-hard-frontier-boost-lora \
  --max-updates 12 \
  --grad-accum 8 \
  --batch-size 1 \
  --max-length 1536 \
  --learning-rate 5e-6 \
  --max-grad-norm 1.0 \
  --checkpoint-steps 6 12 \
  --overwrite-output-dir
```

This keeps the new medium win while concentrating fresh capacity on the
remaining hard slice instead of reopening the already-solved formatting path.

```bash
python training/evaluate_atomicvision_adapter.py \
  --adapter-dir /kaggle/working/atomicvision-format-submit-merged-lora \
  --base-model Qwen/Qwen3-1.7B \
  --difficulties medium hard \
  --episodes 32 \
  --seed-start 10000 \
  --output-json /kaggle/working/atomicvision_adapter_eval.json
```

Watch these verifier columns first:

- `strict_tool_call_pass_rate`
- `normalized_tool_call_pass_rate`
- `first_action_valid_rate`
- `first_action_ask_prior_rate`
- `submit_action_rate`
- `done_rate`
- `tool_failure_rate`

Then inspect the grouped reward-source columns in the JSON report:

- `mean_outcome_reward_total`
- `mean_penalty_total`
- `mean_identity_reward`
- `mean_concentration_reward`
- `mean_confidence_reward`

Only continue to GRPO if strict execution is healthy or normalized execution
shows that the policy is correct and the remaining gap is purely formatting.

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
  --profile two_step_curriculum \
  --episodes-per-difficulty 256 \
  --seed-start 1000 \
  --difficulties medium hard \
  --min-scan-improvement 0.25 \
  --output-jsonl outputs/sft/atomicvision_two_step_curriculum_sft.jsonl
```

This profile concatenates two reproducible phases:

- `format_repair`: many more first-step `ask_prior` rows
- `submit_bridge`: many more second-step `submit_defect_map` rows

Expected sample mix for 512 medium rows:

- `submit_prior`: about 435 rows
- `submit_after_reference`: about 51 rows
- `ask_prior`: about 26 rows

This cost-aware run completed on Kaggle. Promote checkpoint 40:

`/kaggle/working/atomicvision-cost-aware-masked-sft-lora/checkpoint-40`

Medium direct rollout result:

- Reward: `4.47530128125`
- F1: `0.79107153125`
- MAE: `0.0288233125`
- Mean steps: `2.0`
- Mean scan cost: `1.5`
- Tool failure rate: `0.0`
- Done rate: `1.0`

This is the current best checkpoint. The next improvement focus is GRPO, but
only behind a held-out and variance gate. Do not start with a long run. First
run the cost-aware variance probe from the promoted adapter:

```bash
python training/train_grpo_atomicvision.py \
  --preset cost-aware-variance-probe \
  --adapter-model-id /kaggle/working/atomicvision-best-cost-aware-masked-sft-lora \
  --report-to none \
  --run-name atomicvision-cost-aware-grpo-variance-probe
```

This preset uses `--prompt-focus grpo-frontier`, `--seed-start 4000`, and a
frontier seed scan. That means GRPO trains on uncertain or reference-improvable
episodes instead of replaying the easy deterministic SFT cases.

Continue only if `reward_std > 0`, `frac_reward_zero_std < 1`, and `grad_norm`
is nonzero. The first continuation should be short:

```bash
python training/train_grpo_atomicvision.py \
  --preset cost-aware-grpo-20 \
  --adapter-model-id /kaggle/working/atomicvision-best-cost-aware-masked-sft-lora \
  --report-to trackio \
  --output-dir /kaggle/working/atomicvision-cost-aware-grpo-20-lora \
  --run-name atomicvision-cost-aware-grpo-20step
```

Promote a GRPO adapter only after held-out evaluation beats the promoted SFT
checkpoint without raising scan cost or tool failures.

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

Example HF Jobs command for the promoted cost-aware GRPO focus run, after the
adapter has been pushed to a Hugging Face model repo:

```bash
hf jobs run \
  --flavor a10g-large \
  --timeout 3h \
  --secrets HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -lc "git clone https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv AtomicVision && cd AtomicVision && pip install -r training/requirements-grpo.txt && python training/train_grpo_atomicvision.py --preset cost-aware-grpo-100 --adapter-model-id prodigyhuh/atomicvision-best-cost-aware-masked-sft-lora --report-to trackio --run-name atomicvision-hf-cost-aware-grpo-100step --push-to-hub --hub-model-id prodigyhuh/atomicvision-qwen3-1p7b-cost-aware-grpo-lora"
```

If the promoted adapter is still only local in Kaggle, use the local
`/kaggle/working/atomicvision-best-cost-aware-masked-sft-lora` path instead of
the Hub model id.

## Scaling Ladder

Only scale after the previous run reaches at least one completed training step.

1. Smoke: `num-generations=4`, batch `1`, accumulation `4`, steps `5`.
2. Held-out eval: promoted SFT adapter on `medium` and `hard`, seeds
   `1000-1031`.
3. Variance probe: `cost-aware-variance-probe`, `num-generations=8`, batch `1`,
   accumulation `8`, steps `3`.
4. Demo continuation: `cost-aware-grpo-20`, `num-generations=8`, batch `1`,
   accumulation `8`, steps `20`.
5. Stronger HF-credit run: `cost-aware-grpo-100`, `num-generations=8`, batch
   `1`, accumulation `8`, steps `100`.

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
