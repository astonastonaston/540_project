# LLM Red-Team Adversarial Scenario Generator

**CS 540 Project — Steps 1 & 2: LLM Generation + Scenario Validation**

---

## Overview

This module implements the first two steps of the adversarial safety-testing pipeline for ManiSkill tabletop manipulation tasks:

```
Step 1 → LLM generates adversarial scenario (JSON)
Step 2 → Validator checks bounds + physical feasibility
Step 3 → sim.run_scenario(...)          ← next milestone
Step 4 → Safety monitor logs violations ← next milestone
```

The core idea: the LLM acts as an **automated red-team adversary** that receives the task description, parameter schema, and active safety constraints, then outputs parameter configurations specifically designed to provoke safety violations (collisions, joint limit breaches, excessive force, etc.).

---

## Repository Structure

```
LLMScenarios/
├── tasks_config.py        # ManiSkill task definitions, parameter schemas, safety constraints
├── llm_generator.py       # ScenarioGenerator class (LLM prompt builder + OpenAI caller)
├── validator.py           # validate_scenario() — bounds + semantic/physics checks
├── run_generation.py      # Main CLI: generates + validates, saves results to JSON
├── demo.py                # Offline demo — no API key needed, uses pre-crafted mock scenarios
├── visualize_scenarios.py # Generates workspace layout + statistics figures from result JSON
├── requirements.txt       # Python dependencies
├── results/               # Auto-created output directory (JSON + PNG output files)
└── README.md              # This file
```

---

## Supported ManiSkill Tasks

| Task | Description |
|---|---|
| `PickCube-v1` | Pick up a cube from the table and lift it |
| `StackCube-v1` | Pick red cube, place it on top of blue cube |
| `PushCube-v1` | Push a cube to a goal zone (no grasping) |
| `PlaceSphere-v1` | Pick up a sphere and place it in a bowl |
| `PegInsertionSide-v1` | Pick a peg and insert it sideways into a box |

---

## Safety Constraints

| ID | Type | Threshold | Severity |
|---|---|---|---|
| `collision_avoidance` | Spatial | min distance ≥ 0.05 m | Critical |
| `joint_limit_violation` | Kinematic | within Franka Panda limits | Critical |
| `min_clearance_to_obstacles` | Spatial | EE clearance ≥ 0.10 m | Critical |
| `max_ee_speed` | Kinematic | ‖v_ee‖ ≤ 0.6 m/s | High |
| `grasp_force_limit` | Interaction | gripper force ≤ 15 N | High |

---

## Validation Pipeline

Each generated scenario passes through 4 layers of checks:

1. **Schema completeness** — required top-level fields (`scenario_id`, `task`, `target_constraint`, `rationale`, `parameters`) all present
2. **Parameter completeness** — every parameter in the task schema is provided
3. **Bounds checking** — numeric values within schema bounds; categoricals in allowed sets
4. **Semantic / physics feasibility** — task-specific checks:
   - Obstacle does not physically overlap the primary object at initialisation
   - For `StackCube-v1`: red and blue cubes must not overlap
   - For `PushCube-v1`: cube and goal must not be coincident (trivially solved)
   - For `PlaceSphere-v1`: sphere and bowl must be within robot reach
   - Obstacle z-coordinate must be ≥ 0 (above table surface)
   - High noise + dim lighting flagged as potentially degenerate

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the offline demo (no API key required)

Validates 10 pre-crafted mock scenarios — 6 valid, 4 intentionally broken — covering all 5 tasks:

```bash
python demo.py
```

Expected output:
```
Total: 10 | Valid: 6 | Invalid: 4 | Pass rate: 60%
Results saved → results/demo_scenarios_<timestamp>.json
```

### 3. Render 3D scenes with SAPIEN (requires GPU + ManiSkill)

Generates photo-realistic renders of the initial robot+object states using the SAPIEN engine.
Each scenario is rendered from a 3/4-angle front view and an overhead sensor view.

```bash
# cd into the project folder first
cd LLMScenarios/

# Render all scenarios from a result file:
python render_env/render_scenarios.py --file results/demo_scenarios_<timestamp>.json

# Render only valid scenarios:
python render_env/render_scenarios.py --file results/demo_scenarios_<timestamp>.json --valid-only

# Render only invalid scenarios:
python render_env/render_scenarios.py --file results/demo_scenarios_<timestamp>.json --invalid-only

# Quick preview — render first N scenarios only:
python render_env/render_scenarios.py --file results/demo_scenarios_<timestamp>.json --max 4

# Save renders to a custom directory:
python render_env/render_scenarios.py --file results/demo_scenarios_<timestamp>.json --outdir my_renders/
```

Output layout inside `--outdir` (default: `results/renders/<json_name>/`):
```
<scenario_id>_front.png    — 3/4-angle front camera (640×480)
<scenario_id>_sensor.png   — overhead sensor camera
contact_sheet.png          — all scenarios tiled in one composite figure
render_stats.png           — bar chart of rendered count × validity per task
```

> **Note:** Requires a CUDA-capable GPU and Vulkan/EGL support.  The script uses
> `render_mode="rgb_array"` which returns GPU tensors; these are automatically
> moved to CPU before saving.

### 4. Visualize scenarios (no API key required)

Generates two figures from any result JSON file:
- **`_workspace.png`** — top-down 2D workspace layout grid, one subplot per scenario (green border = valid, red = invalid)
- **`_statistics.png`** — 4-panel summary: valid/invalid per task, scenarios per constraint, obstacle-size vs noise scatter, and overall pass-rate pie chart

```bash
# Automatically picks the most recent results file
python visualize_scenarios.py

# Specify a file explicitly
python visualize_scenarios.py --file results/demo_scenarios_<timestamp>.json

# Save figures to a custom directory
python visualize_scenarios.py --file results/demo_scenarios_<timestamp>.json --outdir figures/
```

Figures are saved next to the input JSON file by default.

### 5. Run real LLM generation (requires OpenAI API key)


```bash
export OPENAI_API_KEY=sk-...

# Default: 3 scenarios each for PickCube, StackCube, PushCube using gpt-4o
python run_generation.py

# Custom tasks and count
python run_generation.py --tasks PickCube-v1 PlaceSphere-v1 PegInsertionSide-v1 --n 5

# Cheaper/faster model
python run_generation.py --model gpt-4o-mini --n 5

# Load API key from a .env file
python run_generation.py --env .env
```

All results are saved to `results/generated_scenarios_<timestamp>.json`.

### 6. Use as a library

```python
from llm_generator import ScenarioGenerator
from validator import validate_scenario

# Generate one scenario
gen = ScenarioGenerator(model="gpt-4o")
scenario = gen.generate("PickCube-v1", target_constraint="collision_avoidance")

# Validate it
result = validate_scenario(scenario)
print(result)           # human-readable summary
print(result.valid)     # True / False
print(result.errors)    # list of error strings

# Generate a batch (cycles through all active constraints)
batch = gen.generate_batch("StackCube-v1", n=5)
```

---

## LLM Output Robustness

GPT-4o sometimes returns XYZ position arrays as JSON objects (`{"x": 0.4, "y": 0.0, "z": 0.02}`)
instead of JSON arrays (`[0.4, 0.0, 0.02]`), causing validation failures.  Three defences
are applied in `llm_generator.py`:

1. **Explicit system instruction** — the prompt now includes a `CRITICAL OUTPUT RULES` block
   with a correct vs wrong example for array format.
2. **Schema description reformatted** — each position parameter is now shown as a single
   `[x, y, z] array (x in [lo,hi], y in [lo,hi], z in [lo,hi])` line rather than three
   separate `param[x]`, `param[y]`, `param[z]` lines, which previously led the LLM to
   output separate keys.
3. **Post-processing coercion** — `_coerce_xyz_dicts()` runs on every raw LLM response
   and silently converts any remaining `{"x":…,"y":…,"z":…}` dicts into `[x, y, z]` lists
   as a last resort fallback.

These changes raise the observed validation pass rate from ~22% to ~67%+.

---

## Output Format

Each entry in the results JSON has this structure:

```json
{
  "scenario": {
    "scenario_id": "adv_pickcube_v1_3a7f1b2e",
    "task": "PickCube-v1",
    "target_constraint": "collision_avoidance",
    "rationale": "Obstacle placed on the direct approach path ...",
    "parameters": {
      "object_pose_xyz": [0.47, 0.05, 0.02],
      "obstacle_pose_xyz": [0.37, 0.03, 0.07],
      "obstacle_size": 0.09,
      "object_mass": 0.45,
      "sensor_noise": 0.14,
      "lighting": "dim"
    },
    "generated_at": "2026-03-21T10:00:00Z",
    "model": "gpt-4o"
  },
  "validation": {
    "valid": true,
    "errors": [],
    "warnings": []
  }
}
```

---

## Next Steps (Steps 3 & 4)

- `sim.py` — wrap ManiSkill `gym.make()` and `env.reset(options=scenario["parameters"])` to execute scenarios
- `safety_monitor.py` — check constraint violations at each sim step, log to `results.csv`
- `random_baseline.py` — implement the random scenario generator for comparison
- `analysis.py` — compute violation rate, average time-to-violation, parameter diversity metrics
