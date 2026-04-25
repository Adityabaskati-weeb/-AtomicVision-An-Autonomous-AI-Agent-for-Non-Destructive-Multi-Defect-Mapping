# Training Data Architecture

AtomicVision uses three different kinds of data. Keeping them separate is what
protects policy quality while still letting us use public materials corpora.

## 1. Policy Training Episodes

These are the only rows that should go directly into SFT or GRPO policy
training.

They must look like AtomicVision episodes:

- chat-style `messages`
- exact tool protocol
- final assistant output as an AtomicVision tool call
- metadata tied to an environment seed, difficulty, and reward structure

Current generators and validators live in:

- [generate_atomicvision_sft_data.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/generate_atomicvision_sft_data.py:53>)
- [generate_atomicvision_sft_data.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/generate_atomicvision_sft_data.py:209>)
- [train_sft_atomicvision_safe.py](</C:/Users/baska/OneDrive/Documents/New project/AtomicVision/training/train_sft_atomicvision_safe.py:99>)

Allowed examples:

- `ask_prior`
- `ask_prior -> submit_defect_map`
- `ask_prior -> compare_reference -> submit_defect_map`

Not allowed:

- raw Raman tables
- raw DOS tables
- raw defect-property CSV rows
- free-form Q&A that does not map to the tool contract

## 2. Prior / Reference Corpora

This bucket is for data that should improve upstream scientific knowledge
without directly teaching the model how to emit tool calls.

Examples:

- JARVIS vacancy / DOS data
- Computational Raman Database
- Foundry semiconductor defect levels
- curated reference spectra

This data should feed:

- `ask_prior`
- reference retrieval
- confidence calibration
- simulator parameter calibration

It should not be concatenated raw into policy SFT JSONL.

## 3. Simulator Calibration Data

This bucket tunes the synthetic world itself.

Examples:

- defect prevalence assumptions
- concentration distributions
- confidence heuristics
- scan-improvement heuristics
- reward-calibration studies

The output of this bucket is not training rows directly. It is a better
environment that can then produce better AtomicVision-native episodes.

## Allowed Data Flow

The safe direction is:

`public materials data -> prior/reference/simulator improvements -> regenerated AtomicVision episodes -> policy training`

The unsafe shortcut is:

`public materials data -> raw policy SFT`

## Practical Rule

Before any new dataset is used for model improvement, classify it into one of
these buckets first:

1. `policy_training_episode`
2. `prior_or_reference_corpus`
3. `simulator_calibration_data`

If it is not bucket 1, it does not go directly into SFT or GRPO.
