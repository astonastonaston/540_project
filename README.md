# Red-Teaming the Robot: Automated Violation Synthesis for Safety-Critical Robotic Systems

**CMSC 540 Final Project — Spring 2026**
**Team:** Nan Xiao · Tom Olesch · Qipan Xu 

---

## Overview

This project automatically generates **adversarial test scenarios** for robot manipulation
controllers using large language models (LLMs). An LLM is prompted to act as an adversarial
red-team agent: given a task description and parameter schema, it produces environment
configurations (object positions, obstacle geometry, sensor noise, lighting, mass) that
are specifically chosen to maximise the probability that a safety constraint is violated.

We compare four LLM back-ends (GPT-4o, GPT-4o-mini, Claude Sonnet 4.6, Claude Haiku 4.5),
three prompt engineering strategies (standard, enhanced, self-correction), and a
random-uniform baseline across nine evaluation metrics.

**Key result:** Claude Sonnet 4.6 achieves 100% scenario validity and the highest Adversarial
Quality Score (AQS = 0.789). LLM-generated scenarios trigger real safety violations at
~30% rate versus 0% for random baselines in closed-loop simulation.

---

## Repository Structure

```
540_project/
│
├── llm_generator.py          # LLM adversarial scenario generator (core)
│                             #   ScenarioGenerator (GPT-4o / GPT-4o-mini)
│                             #   ClaudeScenarioGenerator (Sonnet / Haiku)
│                             #   Standard, Enhanced, Self-correction prompts
│
├── validator.py              # Schema + geometric constraint validator
│                             #   checks parameter bounds and 3-D object separation
│
├── metrics.py                # 9 adversarial quality metrics (no simulation required)
│                             #   AQS, proximity, path obstruction, diversity, ...
│
├── tasks_config.py           # ManiSkill3 task schemas and parameter bounds
│                             #   PickCube, StackCube, PushCube, PlaceSphere, PegInsert
│
├── random_baseline.py        # Random-uniform scenario generator (baseline)
│
├── evaluate.py               # Full evaluation pipeline: generate → validate → score
│
├── visualize_scenarios.py    # Matplotlib figures for scenario analysis
│
├── render_comparison.py      # Side-by-side render comparisons (uses render_env/)
│
├── demo.py                   # Quick demo: generate + validate + show 1 scenario
│
├── render_env/               # ManiSkill3 rendering subpackage
│   ├── adv_tasks.py          #   adversarial task wrappers
│   └── render_scenarios.py   #   render a batch of scenarios to PNG
│
├── run_generation.py         # One-off generation (single config, N scenarios)
│
├── run_full_experiment.py    # Experiment A: one LLM vs random baseline (30 each)
│                             #   outputs → figures/eval_expanded/  results/llm_30/
│
├── run_multi_llm_experiment.py  # Experiment B: all 9 configs (main experiment)
│                             #   outputs → figures/multi_llm/  results/multi_llm/
│
├── run_all.sh                # Convenience wrapper: runs both experiments end-to-end
│
├── controller/               # Tom's controller code (see controller/README.md)
│
├── data/                     # Committed dataset — validated scenario JSONs
│   ├── multi_llm/            #   one file per LLM config (7 configs × 30 scenarios)
│   └── random_baseline_scenarios.json
│
├── report/                   # LaTeX final report (Overleaf-ready, compiles standalone)
│   ├── main.tex
│   ├── references.bib
│   ├── IEEEtran.cls          # bundled — compiles on any machine without extra install
│   ├── IEEEtran.bst
│   └── figures/              # report figures (committed to git)
│
├── slides/                   # Presentation slides (LaTeX Beamer)
│   └── presentation_slides.tex
│
├── figures/                  # Auto-generated figures (gitignored — run scripts to regenerate)
├── results/                  # Experiment outputs — JSON + renders (gitignored)
│
├── api_key.py                # YOUR API keys (gitignored — never committed)
├── api_key.py.example        # Template — copy to api_key.py and fill in
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repo
```bash
git clone <repo-url>
cd 540_project
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API keys
```bash
cp api_key.py.example api_key.py
# Edit api_key.py and fill in your real credentials:
#   OPENAI_API_KEY    = "sk-..."
#   ANTHROPIC_API_KEY = "sk-ant-..."
```

The scripts read keys from `api_key.py` automatically — no environment variable
setup needed (environment variables also work if you prefer to use them).

---

## Quick Start

### Run the demo (no API call needed)
```bash
python demo.py
```

### Generate scenarios for one config
```bash
python run_generation.py --model gpt4o --task PickCube-v1 --n 5
```

### Experiment A: GPT-4o vs Random Baseline (30 scenarios each)
```bash
python run_full_experiment.py
# Outputs: figures/eval_expanded/   results/llm_30/   results/random_30/
```

### Experiment B: All 9 configs (GPT-4o, GPT-4o-mini, Claude, and Random Baseline) compared (main experiment)
```bash
python run_multi_llm_experiment.py
# Outputs: figures/multi_llm/   results/multi_llm/<config>/
```

### Re-draw all figures without making API calls
```bash
python run_multi_llm_experiment.py --figs-only
```

### Run everything (experiments A and B) end-to-end
```bash
bash run_all.sh
```

---

## Experiment Scripts in Detail

### `run_multi_llm_experiment.py` — Main Experiment

Generates 30 scenarios per configuration across 5 ManiSkill3 tasks using
8 LLM/prompt configurations + 1 random baseline.

**Configurations:**

| Config name        | Description |
|--------------------|-------------|
| `gpt4o_standard`   | GPT-4o with standard system prompt |
| `gpt4o_enhanced`   | GPT-4o + enhanced (explicit separation rules + worked examples) |
| `gpt4o_selfcorrect`| GPT-4o + self-correction loop (≤2 repair rounds) |
| `gpt4omini_standard` | GPT-4o-mini standard |
| `gpt4omini_enhanced` | GPT-4o-mini enhanced  |
| `claude_sonnet`    | Claude Sonnet 4.6 standard |
| `claude_haiku`     | Claude Haiku 4.5 standard |
| `claude_sonnet_enh`| Claude Sonnet 4.6 enhanced |
| `random_baseline`  | Independent uniform sampling (no LLM) |

**CLI flags:**
```bash
python run_multi_llm_experiment.py [OPTIONS]

  --n N              Scenarios per config per task (default: 6 → 30 total)
  --figs-only        Skip generation; re-draw figures from saved results
  --no-skip          Re-generate even if saved results already exist
  --configs A B ...  Run only selected configs (names from table above)
```

**Example — regenerate only Claude configs:**
```bash
python run_multi_llm_experiment.py --configs claude_sonnet claude_haiku --n 6
```

### `run_full_experiment.py` — Single LLM vs Baseline

```bash
python run_full_experiment.py [--skip-gen] [--model gpt4o|claude_sonnet]
```

Generates 30 LLM + 30 random scenarios and produces 8 comparison figures
in `figures/eval_expanded/`.

### `run_generation.py` — One-Off Generator

```bash
python run_generation.py --model gpt4o --task PickCube-v1 \
                         --n 10 --prompt enhanced --self-correct
```

Prints validated JSON to stdout and saves to `results/quick_gen.json`.

---

## Module Reference

| Module | Key exports | Purpose |
|--------|-------------|---------|
| `llm_generator.py` | `ScenarioGenerator`, `ClaudeScenarioGenerator` | LLM back-end wrappers, prompt building, self-correction loop |
| `validator.py` | `validate_scenario(scenario)` | Returns `(valid: bool, errors: list[str])` |
| `metrics.py` | `compute_adversarial_quality(scenario)`, `compute_full_metrics(scenarios, results)` | Returns dict of all 9 metrics |
| `tasks_config.py` | `TASK_SCHEMAS`, `TASK_DESCRIPTIONS`, `SAFETY_CONSTRAINTS` | Task enum and parameter bounds |
| `random_baseline.py` | `generate_random_scenario(task_name)` | Single uniform-random scenario |
| `evaluate.py` | `run_evaluation(scenarios, validation_results)` | Full eval pipeline, summary dict |

### The 9 Adversarial Quality Metrics

| ID | Name | What it measures |
|----|------|-----------------|
| M1 | Obstacle Proximity | How close the obstacle is to the nearest object (`1 − dist/d_max`) |
| M2 | Workspace Extremity | How far the primary object is from the safe workspace centre |
| M3 | AQS | Weighted composite: 0.40·M1 + 0.25·M2 + 0.20·noise + 0.10·lighting + 0.05·mass |
| M4 | Time-to-Failure Proxy | `M1 / 0.3` — estimated steps before collision |
| M5 | Path Obstruction | How much the obstacle occludes the robot→target straight-line path |
| M6 | Multi-Hazard Density | Fraction of 6 simultaneous binary hazard conditions active |
| M7 | Boundary Push | Distance of object from midpoint of its parameter range |
| M8 | Predicted Violation Rate | Heuristic fraction of 5 safety constraints predicted to be violated |
| M9 | Batch Diversity | Mean pairwise parameter distance; penalises mode collapse |

---

## Reproducing Paper Results

All generated scenario JSON files land in `results/multi_llm/<config>/validated.json`
after running the experiments. To fully reproduce from scratch:

```bash
# 1. Install
pip install -r requirements.txt

# 2. API keys
cp api_key.py.example api_key.py   # fill in real keys

# 3. Run main experiment (~10–20 min, ~270 API calls total)
python run_multi_llm_experiment.py --n 6

# 4. Re-draw figures only (no API calls, uses saved results)
python run_multi_llm_experiment.py --figs-only

# 5. Compile report PDF
cd report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Controller Integration

See `controller/README.md` for full details on adding your code.

Each validated scenario JSON has this structure:
```json
{
  "scenario_id": "abc123",
  "task": "PickCube-v1",
  "target_constraint": "collision_avoidance",
  "parameters": {
    "object_pose_xyz": [0.44, 0.18, 0.02],
    "obstacle_pose_xyz": [0.24, 0.09, 0.15],
    "obstacle_size": 0.12,
    "object_mass": 1.8,
    "sensor_noise": 0.14,
    "lighting": "dim"
  },
  "valid": true,
  "aqs": 0.812
}
```

---

## Citation

```bibtex
@misc{xiao2026redteaming,
  title  = {Red-Teaming the Robot: Automated Violation Synthesis
            for Safety-Critical Robotic Systems},
  author = {Xiao, Nan and Xu, Qipan and Olesch, Tom},
  year   = {2026},
  note   = {CMSC 540 Final Project, University of Maryland}
}
```
