# Experimenter — Usage Guide

Ask Claude a scientific question. The platform runs an ML experiment and answers it.

## Quick Start

```bash
experimenter run "Does exercise frequency predict academic performance?"
experimenter run "Does income predict life satisfaction?" --compute local --time-budget 10
```

## How It Works

```
User Question
     ↓ plan_experiment  (Claude designs hypothesis, model, compute)
     ↓ search_datasets  (Kaggle + HuggingFace + web)
     ↓ download_dataset (cache locally)
     ↓ prepare_data     (impute, encode, split)
     ↓ run_experiment   (local or Vertex AI)
     ↓ answer_hypothesis (verdict + confidence + W&B link)
```

## Using with Claude (MCP)

Once configured (see SETUP.md), ask Claude any question:

> "Does sleep duration predict academic performance?"

Claude will present a plan, ask for confirmation (plan mode), then execute.

## CLI Commands

```bash
experimenter plan   "Does X predict Y?"          # design without running
experimenter search "student GPA dataset"        # find datasets
experimenter run    "Does X cause Y?"            # full pipeline
experimenter status <experiment-id>              # check status
experimenter results <experiment-id>             # view results
experimenter list                                # list all runs
```

## Compute Backends

| Backend    | When to use              | Cost     |
|------------|--------------------------|----------|
| local      | small data, prototyping  | Free     |
| vertex_ai  | large data, GPU needed   | ~$0.02+  |
| auto       | platform decides         | cheapest |

## Experiment Storage

All runs at `~/.experimenter/runs/{id}/run.json`.
