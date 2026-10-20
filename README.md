# TAIRAS

A **Decision-Oriented User State (DOUS)** and Thought Pattern framework for
multi-turn interactive recommendation. A state compiler updates the user state
from public dialogue. The state and historical features form a planner view:
SCTP matches procedures, CIV selects a legal action, and the executor retrieves
products and generates a response.

## Repository contents

- `src/taira_mt/state_compiler/`: state schemas, evidence binding, state updates, and model inference.
- `src/taira_mt/state_training/`: data construction and Curriculum Open-SFT.
- `src/taira_mt/mt_agent/`: Thought Patterns, SCTP, CIV, executors, and baselines.
- `src/taira_mt/multiturn_benchmark/`: benchmark construction and the multi-turn user simulator.
- `data/`: 300 frozen tasks, development samples, SFT splits, and product catalogs for three domains.
- `configs/`: training/inference configurations and lightweight CIV parameters.

This is the **SFT-only** release. It does not include experimental results, API
logs, process documents, or large model weights. Historical identifiers such as
`DSUS` and `taira_mt` are retained for compatibility and do not imply a claim of
statistical sufficiency.

## Installation and quick check

Use Python 3.10-3.12. From the repository root, run:

```bash
python -m pip install -e '.[ml,llm,dev]'
python -m pytest -q tests
```

Run a rule-based control example against the full product catalogs without API calls:

```bash
python -m taira_mt.cli.build_mt_catalog_index
python -m taira_mt.cli.run_mt_baseline \
  --policy taira-mt-rule --max-cases 1 --maximum-turns 2
```

The first command builds reusable retrieval indexes and only needs to run once.
This control example checks the execution pipeline; the full method requires a
trained state compiler and an LLM executor. Generated indexes and outputs are
written to the Git-ignored `artifacts/` directory.

## Running the full agent

Provide an OpenAI-compatible service and a trained adapter. Supply API keys only
through environment variables. Place the final adapter in `models/dous-open-sft/`,
then run:

```bash
export TAIRA_MT_API_KEY='your-key'
export TAIRA_MT_BASE_URL='https://your-provider.example/v1'
export TAIRAS_EXECUTOR_MODEL='your-model'

python -m taira_mt.cli.run_mt_baseline \
  --policy taira-mt-compiler \
  --state-compiler-profile configs/compiler_local.json \
  --civ-model configs/civ_ranker.joblib --selector-mode learned \
  --model "$TAIRAS_EXECUTOR_MODEL" --provider-profile-id user-configured \
  --max-completion-tokens 4096 --max-cases 1 --maximum-turns 2
```

Local model inference also requires the `.[state-compiler-train]` dependencies.
The simulator does not require LLM calls at runtime.

## Supervised fine-tuning

```bash
python -m pip install -e '.[state-compiler-train]'
python -m taira_mt.cli.train_dsus_state_compiler_sft \
  --config configs/train_sft.json
```

The configuration retains Qwen3.5-4B and two Open-SFT epochs. Before running,
place the opening-calibrated adapter from the controlled stage in
`models/opening-calibrated/`. These weights are not included. Training directly
from the base model is a different setup from the original training pipeline.
Relative model and training-data paths in configuration files are resolved
relative to the configuration file's directory.

## Data and evaluation notes

Product data originates from Amazon. SFT data includes synthetic supervision
and samples derived from external dialogues; use and redistribution are subject
to the upstream data terms. Task files contain simulator-private goals. At
runtime, the recommendation agent receives only public dialogue and catalog
interfaces.

Included runtime statistics are intended for debugging. This repository does
not include final paper summary tables or offline evaluation results.
