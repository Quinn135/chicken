# chicken

Isaac Lab 3 environment for training the chicken robot to walk, as a uv project.

Registered task: **`Chicken-Walk-Direct`** (it was `Isaac-Chicken-Robot-v0` in the Isaac Lab 2.x / Docker version).

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/) and an Isaac Lab 3 source checkout
**next to this repo**. `pyproject.toml` points at it with relative paths:

```
thin/
├── IsaacLab/      # Isaac Lab 3 checkout
└── chickengit/    # this repo
```

If either folder moves, update the paths in `[tool.uv.sources]` in `pyproject.toml`.

Create the environment (once, and again after pulling changes to `pyproject.toml` or `uv.lock`):

```bash
uv sync
```

That installs Isaac Lab, skrl and standalone PhysX (OvPhysX). No Docker or conda environment is needed;
every command below runs through `uv run`, which uses the project's `.venv`.

## Train

```bash
uv run isaaclab train --rl_library skrl --task Chicken-Walk-Direct --num_envs 4096
```

Runs are headless unless you pass `--viz` (there is no `--headless` flag in 3.x). Logs and checkpoints are
written to `logs/skrl/chicken2/<timestamp>_ppo_torch/`. To resume, pass `--checkpoint <path/to/agent_N.pt>`.

Watch training progress:

```bash
uv run tensorboard --logdir logs/skrl/chicken2
```

## Play a trained policy

```bash
PYGLFW_LIBRARY_VARIANT=x11 uv run isaaclab play --rl_library skrl --task Chicken-Walk-Direct --num_envs 9 --checkpoint latest --viz newton_gl
```

`--checkpoint latest` picks the newest run in `logs/skrl/chicken2`. `PYGLFW_LIBRARY_VARIANT=x11` is needed for the
`newton_gl` viewer on Wayland desktops; without it the viewer crashes with `undefined symbol: glfwGetX11Window`.
To make it permanent, add `export PYGLFW_LIBRARY_VARIANT=x11` to `~/.bashrc`.

Quick checks without a trained policy:

```bash
uv run isaaclab list_envs
uv run isaaclab random_agent --task Chicken-Walk-Direct --num_envs 16 --max_steps 600
```

## Physics backend

The robot settings (solver iterations, depenetration velocity, sleep threshold) are PhysX-specific, so the
task defaults to PhysX: Isaac Sim's PhysX if it is installed, otherwise standalone OvPhysX. To pick one
explicitly, add `physics=<name>` to any command:

| `physics=`       | Needs                                                          |
| ---------------- | -------------------------------------------------------------- |
| `physx`          | Default; uses whichever PhysX below is installed               |
| `ovphysx`        | Nothing extra (installed by `uv sync`)                         |
| `isaacsim_physx` | `--extra isaacsim` on every command (full Isaac Sim, large download) |

There is no Newton preset yet; that needs Newton solver settings tuned for the chicken.

## Changing the robot

The task loads `src/chicken/assets/data/robots/usd/chicken/chicken.usda`, which is generated from the URDF in
`src/chicken/assets/data/robots/chicken_v6_5/`. After editing the URDF, regenerate it and commit the result:

```bash
uv run --extra importers python scripts/convert_robot.py
```

If you add, remove or rename links, also update `CHICKEN_BODY_PATHS` in `src/chicken/assets/chicken.py`. The 3.x
importer nests each link under its parent, so the sensors need each link's full path.

## Project layout

| Path                                                       | What it is                                          |
| ---------------------------------------------------------- | --------------------------------------------------- |
| `src/chicken/tasks/walk_direct/config/chicken/env.py`      | The environment: observations, rewards, resets      |
| `src/chicken/tasks/walk_direct/config/chicken/env_cfg.py`  | Scene (ground, robot, sensors, lights) and physics  |
| `src/chicken/tasks/walk_direct/config/chicken/agents/`     | skrl PPO settings                                   |
| `src/chicken/tasks/walk_direct/config/chicken/__init__.py` | Task registration                                   |
| `src/chicken/assets/chicken.py`                            | Robot config: actuators, physics properties         |
| `src/chicken/assets/data/`                                 | URDF, generated robot USD and ground USD            |
| `scripts/convert_robot.py`                                 | URDF to USD conversion                              |
| `models/`                                                  | Older robot USDs from the 2.x project (unused)      |

## Not ported yet

`scripts/control.py` (keyboard teleop) and `scripts/skrl/export.py` (ONNX export) still use the Isaac Lab 2.x
launcher (`AppLauncher`) and won't run as they are.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run pre-commit run --all-files
```

To set up VS Code or Cursor with the project's interpreter:

```bash
uv run isaaclab --editor
```
