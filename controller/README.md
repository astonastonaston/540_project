# Controller Code — Tom Olesch

This directory contains the controller under test:
**PD + Waypoint Navigation + Potential Field** applied to the
**ReachGoal** task in ManiSkill3.

---

## How to integrate

1. Clone the repo if you haven't already:
   ```bash
   git clone <repo-url>
   cd 540_project
   ```

2. Add your controller files to this directory. Suggested layout:
   ```
   controller/
   ├── README.md          ← this file
   ├── reach_goal.py      ← main controller class
   ├── pd_controller.py   ← PD control loop
   ├── potential_field.py ← potential field safety module
   ├── evaluate_controller.py  ← run controller on generated scenarios
   └── safety_monitor.py  ← constraint violation detector
   ```

3. The scenario JSON files that the LLM generator produces are stored in
   `results/multi_llm/<config>/validated.json`.
   Load them in your evaluation script like:

   ```python
   import json, sys, os
   sys.path.insert(0, os.path.dirname(__file__) + '/..')  # reach root

   with open('../results/multi_llm/claude_sonnet/validated.json') as f:
       scenarios = json.load(f)

   for scenario in scenarios:
       params = scenario['parameters']
       task   = scenario['task']
       # ... run your controller with params ...
   ```

4. Six adversarial knobs the LLM generates values for (ReachGoal task):

   | Parameter           | Role                                      |
   |---------------------|-------------------------------------------|
   | `ee_start_xyz`      | End-effector start position               |
   | `ee_sensor_noise`   | Noise on EE position sensor               |
   | `joint_sensor_noise`| Noise on joint angle sensors              |
   | `control_delay`     | Latency injected into control loop (s)    |
   | `gain_scale`        | Multiplier on PD gains                    |
   | `influence_distance`| Potential-field activation radius (m)     |

5. Safety constraints monitored during evaluation:

   | Constraint                  | Threshold       |
   |-----------------------------|-----------------|
   | EE-to-wall clearance        | ≥ 5 cm          |
   | Joint angle limits          | Within ±limit   |
   | Max EE speed                | ≤ v_max         |
   | Collision avoidance         | No contact      |
   | Grasp force limit           | ≤ F_max         |

---

## Contact

- **Tom Olesch** — controller implementation
- **Nan Xiao** — scenario generator (`llm_generator.py`, `run_multi_llm_experiment.py`)
- **Qipan Xu** — \[add your role here\]
