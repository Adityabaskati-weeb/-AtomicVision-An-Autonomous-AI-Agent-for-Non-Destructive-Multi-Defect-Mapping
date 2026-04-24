# Lecture 91 OpenEnv Method Notes

Source transcript:
`C:\Users\baska\OneDrive\Desktop\NoteGPT_TRANSCRIPT_Mega Lecture 91 Reinforcement Learning, Agents & OpenEnv.txt`

These notes extract the methods from the lecture that matter for AtomicVision.
Use this as the reference when deciding whether to do more SFT, start GRPO,
change rewards, debug Kaggle runs, or prepare the Hugging Face training phase.

## Core Stack

AtomicVision should keep the same practical stack:

```text
OpenEnv environment -> verifier/reward functions -> TRL GRPO trainer -> Unsloth/QLoRA efficiency -> Hugging Face Space demo
```

The environment is the first-class artifact. The model is trained only after the
environment, reward logic, and rollout audit loop are stable.

## Main Methods From The Lecture

### 1. RL As Efficient Trial-And-Error

The lecture frames RL as repeated in-context search made weight-based. Instead
of sampling many completions forever and stuffing good/bad examples into the
prompt, the model samples outputs, the environment scores them, and gradient
updates move probability mass toward higher-reward behavior.

AtomicVision use:

- keep the cheap successful behavior learned by SFT
- use GRPO only where the model still needs exploration
- do not run GRPO on easy deterministic examples that all get the same reward

### 2. Success Probability Must Be Nonzero

RL fails when the model never reaches reward. If every rollout scores zero,
there is no useful learning signal.

AtomicVision use:

- start from the promoted SFT adapter, not raw base model
- use `--prompt-focus grpo-frontier` to select borderline or reference-improvable seeds
- keep max episode length short at first
- add curriculum only after the variance probe proves learning signal exists

### 3. SFT Versus GRPO

Use SFT when good demonstrations can be generated or filtered. Use GRPO when
the answer is not known in advance but success can be verified.

AtomicVision use:

- SFT already solved formatting and prior-copy reliability
- GRPO should now target when to spend extra evidence cost
- more SFT is useful only if held-out or hard eval shows the current adapter is brittle

### 4. Environment Before Trainer

OpenEnv standardizes the agent boundary:

- `reset`: start a new episode
- `step`: apply an action and produce observation/reward/done
- `state` or observation model: define what the agent can see
- actions: define what the agent can do

AtomicVision use:

- the hidden defect map must stay hidden
- observations should expose only prior, scan summaries, reference deltas, and costs
- terminal `submit_defect_map` must end the episode
- any post-terminal tool call should be penalized and logged

### 5. Reward Engineering Is The Task Specification

The reward is what the model optimizes. If the reward is incomplete, the model
will optimize the loophole.

AtomicVision reward components should stay layered:

- environment reward for defect-map quality and scan cost
- format reward for valid tool calls
- prior-copy reward for cheap high-confidence submission
- terminal discipline penalty for calling tools after submit
- failure penalty for malformed or unsupported tools
- separate metrics for done rate, scan cost, F1, MAE, and tool failure rate

### 6. Reward Hacking Prevention

The lecture repeatedly warned that reward hacking is not rare. It is normal
optimization pressure finding gaps in the spec.

AtomicVision safeguards:

- use multiple independent reward components
- log component rewards, not only total reward
- inspect sampled rollouts during training
- watch rollout length and repeated action patterns
- stop if reward rises but real F1 or held-out quality drops
- keep hidden state out of observations
- avoid letting generated code or tools mutate global state
- keep timeouts on environment steps

### 7. Process Supervision And Rubric Metrics

Final reward assigned to every token is inefficient. Better signals can come
from step-level checks, trace analysis, or rubric metrics.

AtomicVision use:

- keep final outcome reward as the main optimization target
- use process/rubric metrics mostly for logging first
- do not over-shape rewards until sparse reward becomes the real bottleneck
- a repetition or unnecessary-scan metric can be logged with zero reward weight
  before making it part of the optimizer

### 8. Dynamic Reward Weights

Some shaping rewards help early and hurt later. The lecture suggested reducing
weights over time when a reward becomes regressive.

AtomicVision use:

- early format rewards are useful
- once valid tool calls are stable, keep format shaping small
- cost penalties should remain active because the project goal is cost-aware mapping
- if GRPO starts over-optimizing cheap submission, increase reference-improvement sampling

### 9. GRPO And RLVR

GRPO compares multiple completions for the same prompt. It needs reward
variance inside each group. RLVR means using a verifier or environment instead
of a learned reward model.

AtomicVision use:

- `reward_std > 0` is required
- `frac_reward_zero_std < 1` is required
- `grad_norm > 0` is required
- if all completions get identical reward, stop and change the prompt pool or reward

### 10. OpenEnv Deployment Modes

The lecture described several OpenEnv usage patterns:

- run the FastAPI app locally
- deploy as a Hugging Face Space
- use the Space as a remote environment
- install the Space repo as Python client code
- pull/run the Space container locally
- scale with providers, workers, websockets, or horizontal services

AtomicVision use:

- Hugging Face Space remains the demo source of truth
- Kaggle can train against local code when artifacts are missing
- HF Jobs can later train against the published Space or local package
- keep one environment interface so training and demo do not diverge

### 11. Unsloth, LoRA, And QLoRA

The lecture's practical LoRA guidance matches our current setup:

- use LoRA for task specialization
- target attention and MLP modules, not only attention
- QLoRA is for memory-efficient training
- during RL, training and inference precision should match
- when saving, do not naively upcast a 4-bit model and merge into it
- save adapters directly or merge LoRA into the original 16-bit base through the proper path

AtomicVision current setup is aligned:

- base model: `Qwen/Qwen3-1.7B`
- LoRA rank: `16`
- LoRA alpha: `32`
- target modules include attention plus MLP projections
- promoted method: cost-aware assistant-masked QLoRA SFT

### 12. Monitoring

The lecture emphasized that one scalar reward is not enough.

AtomicVision must monitor:

- mean reward
- reward standard deviation
- fraction of zero-variance reward groups
- gradient norm
- completion length
- environment reward
- format reward
- prior-copy reward
- done rate
- scan cost
- tool failure rate
- post-terminal tool calls
- sampled rollouts

### 13. Scaling And Sandboxes

For small hackathon work, the GPU is not the only bottleneck. Environment CPU
cost, container startup, concurrency, and remote latency matter.

AtomicVision use:

- keep the environment lightweight
- avoid heavy per-step dependencies
- do small variance probes before expensive training
- run local or Kaggle first, HF Jobs later
- scale only after reward and rollout behavior are stable

## AtomicVision Decision Rules

### If The Question Is "More SFT Or GRPO?"

Answer:

1. Run held-out evaluation first.
2. If format or basic action reliability fails, do more SFT.
3. If format is stable but reference/cost decisions are weak, do GRPO.
4. If GRPO has zero reward variance, change prompt selection or reward before continuing.

### If The Question Is "Is The Model Overfitting?"

Check:

- train loss versus held-out reward
- medium seeds not used in SFT
- hard seeds not used in SFT
- whether all checkpoints have identical eval scores
- whether the model simply learned one deterministic policy

The current checkpoint may be narrow because all evaluated checkpoints matched
on 32 medium episodes, but it is not proven overfit until held-out and hard
eval fail.

### If The Question Is "What Should We Train Next?"

Use this order:

1. Held-out eval of promoted SFT.
2. GRPO variance probe on frontier seeds.
3. 20-step guarded GRPO continuation.
4. Held-out eval versus SFT.
5. 100-step HF-credit run only if the 20-step run improves.

### If Kaggle Files Are Missing

Do not assume `/kaggle/working` still has old artifacts. Kaggle sessions reset.

Recovery order:

1. If adapter zip exists, unzip and eval.
2. If final submission package exists, extract adapter zip from it.
3. If no adapter exists, regenerate data and retrain from the recipe.
4. Then promote best checkpoint again.

### If Asked For Current Model Facts

Use:

- Dataset size: `512`
- Model: `Qwen/Qwen3-1.7B`
- LoRA rank: `16`
- LoRA alpha: `32`
- Optimizer updates: `80`
- Best checkpoint: `checkpoint-40`
- Best reward: `4.4753`
- Best F1: `0.7911`
- Best MAE: `0.02882`
- Scan cost: `1.50`

If asked one-word iteration answer:

```text
80
```

## Do Not Do

- Do not run GRPO from scratch.
- Do not train on hidden evaluation seeds.
- Do not optimize only total reward.
- Do not trust rising reward without rollout inspection.
- Do not add many dense shaping rewards before testing the simple reward.
- Do not let generated outputs access hidden state.
- Do not merge QLoRA by upcasting the 4-bit model.
- Do not spend HF compute before the variance probe passes.

## References Checked

- Local lecture transcript listed above.
- OpenEnv GitHub: https://github.com/meta-pytorch/OpenEnv
- TRL OpenEnv docs: https://huggingface.co/docs/trl/openenv
- TRL GitHub: https://github.com/huggingface/trl
- Unsloth GRPO guide: https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo
- Unsloth save/merge docs: https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-vllm

No StackOverflow lookup was needed for this pass because there was no concrete
runtime error to debug. For future implementation errors, prefer official docs
first, then StackOverflow only for a specific traceback or environment issue.
