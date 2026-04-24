# AtomicVision Notebooks

## AtomicVision_Judge_Repro_Colab.ipynb

Judge-facing reproducible Colab notebook for the current held-out recovery path.

It runs:

- branch clone from GitHub
- dependency installation
- `two_step_curriculum` SFT dataset generation
- NaN-safe SFT validation
- 5-update sanity train
- 40-update rebuild
- optional `publish_adapter_to_hub.py`
- strict + normalized held-out adapter eval

## AtomicVision_GRPO_Colab.ipynb

Phase 10 bridge notebook plus Phase 11 GRPO launch notes for the deployed OpenEnv Space.

It verifies:

- public Space health
- remote environment reset
- persistent WebSocket session
- `ask_prior` action
- `submit_defect_map` action
- reward breakdown visibility
- training dependency installation
- dry-run command for `training/train_grpo_atomicvision.py`
- minimal GRPO smoke command for Colab/Kaggle
