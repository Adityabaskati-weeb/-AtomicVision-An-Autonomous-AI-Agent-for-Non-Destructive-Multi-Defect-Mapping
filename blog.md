# AtomicVision: Training A Small Model To Do Cost-Aware Defect Mapping

AtomicVision is our OpenEnv environment for non-destructive atomic defect
mapping. Instead of giving a model one static prompt and asking for a guess, we
turn the task into a small scientific workflow:

- the agent starts with compact spectral evidence,
- chooses characterization tools,
- pays explicit scan costs,
- and submits a final defect map under budget pressure.

That makes it a much better fit for real tool-using model training than a
single-turn benchmark.

## The Problem

We wanted to train a model to do something that feels closer to real materials
analysis than ordinary chat completion:

- reason under partial observability,
- decide when more evidence is worth the cost,
- use tools in the right order,
- and finish with a verifiable answer.

In AtomicVision, the model does not get destructive access to the sample. It
has to work from non-invasive spectral evidence and a small tool set:

- `ask_prior`
- `compare_reference`
- `request_scan`
- `zoom_band`
- `submit_defect_map`

That combination of uncertainty, cost, and structure is exactly why we built it
as an OpenEnv environment.

## What Made The Project Hard

The first big surprise was that the hardest problem was not pure scientific
reasoning. It was interface reliability.

Early on, the model often had the right intent but expressed it in the wrong
tool-call format. So before we could trust any reward improvement, we had to
stabilize the execution layer:

- NaN-safe SFT training
- strict and normalized held-out evaluation
- schema-aware two-step curriculum
- saved adapters on the Hub instead of ephemeral notebook checkpoints

That groundwork mattered. Once the tool layer became reliable, we could finally
measure whether the model was actually getting better at the task itself.

## What Training Actually Worked

The strongest stable base we built first was:

- [prodigyhuh/atomicvision-medium-fidelity-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-medium-fidelity-boost-lora)

That adapter gave us a reliable, submission-worthy policy with perfect strict
tool execution.

From there, we mined held-out hard failures and found a very specific remaining
weakness:

- the model was mostly failing by **missing defects** on hard seeds
- not by malformed XML
- not by broken tool use
- and not by random action collapse

So instead of doing a broad continuation, we ran a tiny targeted repair pass
focused on that missed-defect recall bottleneck.

## Final Model Result

The winning model is now published here:

- [prodigyhuh/atomicvision-hard-recall-micro-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-hard-recall-micro-boost-lora)

It came from the `checkpoint-1` winner of the hard-recall micro-repair run:

- [hard-recall-micro-repair-results.md](docs/hard-recall-micro-repair-results.md)

Held-out strict comparison versus the previous best published adapter:

| Metric | Previous best | Current best | Delta |
| --- | ---: | ---: | ---: |
| medium reward | 4.5065 | 4.5065 | 0.0000 |
| medium F1 | 0.7891 | 0.7891 | 0.0000 |
| hard reward | 4.6917 | 4.7148 | +0.0231 |
| hard F1 | 0.8162 | 0.8207 | +0.0045 |
| strict tool-call pass | 1.00 | 1.00 | 0.00 |
| normalized tool-call pass | 1.00 | 1.00 | 0.00 |
| tool failure rate | 0.00 | 0.00 | 0.00 |
| done rate | 1.00 | 1.00 | 0.00 |

That is the shape of improvement we wanted:

- better hard-case quality,
- no regression on medium,
- and no loss of strict execution reliability.

Here is the cleanest view of the final comparison:

![Final model comparison](docs/final-model-comparison.png)

The story is simple: the final adapter does not win by getting flashier or
looser. It wins by improving the **hard** slice while keeping the rest of the
system steady.

That became even clearer when we plotted the checkpoint sweep from the targeted
repair run:

![Checkpoint sweep](docs/final-checkpoint-sweep.png)

This second graph is important because it shows that our final gain did not come
from "more training" in the generic sense. The improvement appeared early at
`checkpoint-1`, and longer continuation gave it back. In other words, the
winning move was **narrow training plus early stopping**, not brute force.

## Why This Matters

AtomicVision is not a toy grid world and not a static classification prompt. It
is a professional-task environment where the model has to:

- gather evidence,
- spend budget wisely,
- act through tools,
- and finish correctly.

That makes it useful in two ways:

1. as a hackathon submission that demonstrates real end-to-end training on an
   OpenEnv environment
2. as a research sandbox for studying how small models improve on structured,
   partially observable scientific tasks

## Try It

- Space: [prodigyhuh/atomicvision-openenv](https://huggingface.co/spaces/prodigyhuh/atomicvision-openenv)
- Live app: [prodigyhuh-atomicvision-openenv.hf.space](https://prodigyhuh-atomicvision-openenv.hf.space)
- Final model: [prodigyhuh/atomicvision-hard-recall-micro-boost-lora](https://huggingface.co/prodigyhuh/atomicvision-hard-recall-micro-boost-lora)
- Judge notebook: [notebooks/AtomicVision_Judge_Repro_Colab.ipynb](notebooks/AtomicVision_Judge_Repro_Colab.ipynb)
- Writeup: [docs/judge-writeup.md](docs/judge-writeup.md)

The nicest part of this final result is that we did not need a giant model to
show meaningful improvement. We needed a reliable environment, honest held-out
evaluation, and one very targeted training step in the right place.
