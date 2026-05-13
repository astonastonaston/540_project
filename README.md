# Red-Teaming the Robot: Automated Violation Synthesis for Safety-Critical Robotic Systems

**COSC 540 Final Project — Spring 2026**
**Team:** Nan Xiao · Tom Olesch · Qipan Xu 

---

## Overview

This project automatically generates **adversarial test scenarios** for robot manipulation
controllers using large language models (LLMs). An LLM is prompted to act as an adversarial
red-team agent: given a task description and parameter schema, it produces environment
configurations (object positions, obstacle geometry, sensor noise, lighting, mass) that
are specifically chosen to maximise the probability that a safety constraint is violated.

The framework has two evaluation arms.
An **offline arm** (this directory) scores generated scenarios analytically across five
ManiSkill3 tabletop tasks — comparing four LLM back-ends (GPT-4o, GPT-4o-mini,
Claude Sonnet 4.6, Claude Haiku 4.5), three prompt engineering strategies (standard,
enhanced, self-correction), and a random-uniform baseline across nine evaluation metrics.
A **closed-loop arm** ([`closed_loop/`](closed_loop/)) rolls validated scenarios out
against a Franka Panda controller in ManiSkill3 simulation and records ground-truth
safety violations.

**Key results:**
- *Offline arm:* Claude Sonnet 4.6 achieves 100% scenario validity and the highest
  Adversarial Quality Score (AQS = 0.789).
- *Closed-loop arm:* LLM-generated scenarios trigger real safety violations in **34.3%**
  of episodes (95% CI [0.200, 0.486]) versus **0%** for the random baseline over 100
  episodes on ReachGoal.


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

# 5. (Optional) Reproduce the closed-loop arm — ManiSkill3 rollouts
cd closed_loop
pip install -r requirements.txt          # adds ManiSkill3 + gymnasium
python run_comparison_study.py           # 35 LLM + 100 random ReachGoal rollouts
cd ..

# 6. Compile report PDF
cd report
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Closed-Loop Arm

[`closed_loop/`](closed_loop/) contains the second evaluation arm of the
framework: a custom **ReachGoal** task in ManiSkill3, a PD + waypoint +
potential-field controller as the system under test, a runtime safety monitor
that checks seven constraints every simulation step, and a deterministic
root-cause analyser that attributes each violation to perception, planning,
or control. The arm exposes six controller-specific adversarial knobs (EE
start, EE-position noise, joint-encoder noise, control delay, gain scale,
potential-field influence distance) and rolls validated scenarios out
against the controller to record ground-truth violations.

**Headline numbers** (35 LLM scenarios vs. 100 random, on ReachGoal —
full data in [`closed_loop/study_output/comparison_summary.csv`](closed_loop/study_output/comparison_summary.csv)):

| Metric                    | LLM (n=35)                       | Random (n=100) |
|---------------------------|----------------------------------|----------------|
| Violation rate            | **34.3 %** (95 % CI [0.20, 0.49])| 0.0 %          |
| Goal-reach rate           | 45.7 %                           | 96.0 %         |
| Mean violations / episode | 14.46                            | 0.00           |

Root-cause attribution across LLM-induced violations: control 50 %,
planning 38 %, perception 12 %.

![Clean baseline rollout of ReachGoal](assets/reachgoal_task_successful.gif)

*Clean baseline rollout of ReachGoal — the controller climbs the wall and
reaches the goal with zero safety violations
([MP4 download](assets/reachgoal_task_successful.mp4)).*

See [`closed_loop/README.md`](closed_loop/README.md) for the scenario YAML
format and how to run the experiments.

---

## Citation

```bibtex
@misc{xiao2026redteaming,
  title  = {Red-Teaming the Robot: Automated Violation Synthesis
            for Safety-Critical Robotic Systems},
  author = {Xiao, Nan and Xu, Qipan and Olesch, Tom},
  year   = {2026},
  note   = {COSC 540 Final Project, University of Tennessee, Knoxville}
}
```
