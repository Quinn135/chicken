# chicken

Isaac Lab 3 environment for training the chicken robot to walk, as a uv project.

Registered task: **`Chicken-Walk-Direct`** (it was `Isaac-Chicken-Robot-v0` in the Isaac Lab 2.x / Docker version).

## Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), [Git LFS](https://git-lfs.com/) and an
Isaac Lab 3 source checkout **next to this repo**. `pyproject.toml` points at it with relative paths:

```
thin/
├── IsaacLab/      # Isaac Lab 3 checkout
└── chickengit/    # this repo
```

If either folder moves, update the paths in `[tool.uv.sources]` in `pyproject.toml`.

On a fresh machine, clone both side by side. The robot and terrain USD files are stored in Git LFS, so run
`git lfs install` once first, or they arrive as small text pointers and the scene fails to load:

```bash
git lfs install
git clone https://github.com/isaac-sim/IsaacLab.git
git clone https://github.com/Quinn135/chicken.git chickengit
```

This project was ported and tested against Isaac Lab `main` at commit `eceac1ac1` (version 3.0.0, 2026-11-15).
`main` changes often; if something breaks after updating Isaac Lab, check out that commit in `IsaacLab/`:

```bash
cd IsaacLab && git checkout eceac1ac1
```

`onnx2c/` is a git submodule used for compiling exported policies to C. It is not needed for training; fetch it
with `git submodule update --init` when you need it.

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

The config defaults to 4096 environments, so pass a smaller `--num_envs` to `play` and `control.py`. Runs are headless unless you pass `--viz`
(there is no `--headless` flag in 3.x). Logs and checkpoints are written to
`logs/skrl/chicken2/<timestamp>_ppo_torch/`. To resume, pass `--checkpoint <path/to/agent_N.pt>`.

Checkpoints from the Isaac Lab 2.x / Docker version won't load: the observation size and network changed since
then, so train a new policy.

Watch training progress:

```bash
uv run tensorboard --logdir logs/skrl/chicken2
```

## Play a trained policy

```bash
PYGLFW_LIBRARY_VARIANT=x11 uv run isaaclab play --rl_library skrl --task Chicken-Walk-Direct --num_envs 9 --checkpoint latest --viz newton_gl --real-time
```

`--checkpoint latest` picks the newest run in `logs/skrl/chicken2`. `--real-time` makes it play at actual speed;
without it the sim runs as fast as the GPU allows, which looks sped up. Recorded videos are always at the right speed. `PYGLFW_LIBRARY_VARIANT=x11` is needed for the
`newton_gl` viewer on Wayland desktops; without it the viewer crashes with `undefined symbol: glfwGetX11Window`.
To make it permanent, add `export PYGLFW_LIBRARY_VARIANT=x11` to `~/.bashrc`.

Quick checks without a trained policy:

```bash
uv run isaaclab list_envs
uv run isaaclab random_agent --task Chicken-Walk-Direct --num_envs 16 --max_steps 600
```

### Recording video

Add `--video` to `play` or `random_agent` to record an MP4 without opening a window. It needs the `video` extra:

```bash
PYGLFW_LIBRARY_VARIANT=x11 uv run --extra video isaaclab play --rl_library skrl --task Chicken-Walk-Direct --num_envs 9 --checkpoint latest --video --video_length 600
```

Videos are saved under the run's `videos/` folder (for `random_agent`, `logs/random_agent/Chicken-Walk-Direct/videos/`).

## Drive it with the keyboard

```bash
PYGLFW_LIBRARY_VARIANT=x11 uv run python scripts/control.py --checkpoint latest --num_envs 1 --viz newton_gl --real-time
```

Click the viewer window so it has keyboard focus, then:

| Keys                     | Does                  |
| ------------------------ | --------------------- |
| I / K (or numpad 8 / 2)  | Walk forward / back   |
| J / L (or numpad 4 / 6)  | Side-step left / right|
| U / O (or numpad 7 / 9)  | Turn left / right     |
| R                        | Reset the robots      |

WASD, the arrow keys and Q/E still move the viewer camera. Commands ramp toward the held keys. Speeds are set with
`--v_x_sensitivity`, `--v_y_sensitivity` (m/s) and `--yaw_sensitivity` (rad/s), and the gait frequency with
`--step_frequency`. `--checkpoint` and `--video` work as in `isaaclab play`.

While driving, the robot isn't reset on a timer, and only gets reset if it falls in its first 60 steps (or when
you press R).

How it works: the script loads the policy the same way `isaaclab play` does, then sets `is_teleop` on the env so
it stops sampling random velocity commands. Every step it reads the held keys from the viewer window, moves the
command `--smoothing` (default 20%) of the way toward them, passes it to the env with `set_teleop_commands`, and
runs the policy's mean action. The policy sees your command exactly where it saw the training commands.

## Export for the robot

```bash
uv run --extra export python scripts/export.py --checkpoint latest --onnx2c ~/onnx2c/build/onnx2c
```

This writes `exports/<run>_<checkpoint>/` with:

- `policy.onnx`: input `(1, 838)` raw observations, output `(1, 10)` actions (one per joint, in `CHICKEN_JOINTS` order)
- `nn.c` and `nn.h`: `void entry(const float tensor_input[1][838], float tensor_output[1][10])`, generated by
  [onnx2c](https://github.com/kraiskil/onnx2c). Copy both into the mainboard firmware's `lib/nn/`.

The script checks that the exported model matches what `isaaclab play` runs, and that ONNX matches PyTorch.

**Observation normalization is now inside the exported model.** During training and play, skrl normalizes
observations (`RunningStandardScaler`) before the policy sees them. The 2.x export left that step out, so a
firmware feeding raw sensor values gave the policy inputs on a different scale than in training. The new
`nn.c` expects the same raw observations the firmware already builds. Pass `--no_normalization` only if your
firmware normalizes them itself.

`--onnx2c` can be left out if `onnx2c/build/onnx2c` exists in this repo (run `git submodule update --init`,
then follow the build steps in `onnx2c/README.md`) or `onnx2c` is on your `PATH`. Without it, only
`policy.onnx` is written.

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

## Robot settings

Everything about the simulated robot is in `src/chicken/assets/chicken.py`:

- **Actuators** (`actuators=` in `CHICKEN_CFG`): one `DelayedPDActuatorCfg` for all 10 joints, with the command
  delay, torque limits (`actuator_effort_limit` is what the actuator model clips to, `joint_effort_limit` is the
  PhysX limit), velocity limits, PD gains (`stiffness`, `damping`), armature and friction. These override the
  drive gains written into the USD, so the gains in `scripts/convert_robot.py` don't matter.
- **Joint position limits** come from the URDF; change them there and reconvert.
- **Spawn pose** (`init_state`): root height and starting joint angles.
- **`CHICKEN_JOINTS`**: the actuated joints, in the policy's action order: the left leg (turn hip, side hip, hip,
  knee, ankle), then the right leg in the same order. The env splits the list in half, so keep both legs
  the same length and order. The ankle joints are named after the feet (`lfoot`, `rfoot`). The action and
  observation sizes follow from its length.

The reward targets are in `src/chicken/tasks/walk_direct/config/chicken/env.py`. `target_height` is the IMU's
height above the lower foot (both measured at their centers of mass); the straight-leg height of the current
robot is 0.595 m, so the target sits a little below that for a slight knee bend.

## Changing the robot

The robot is generated from the URDF in `src/chicken/assets/data/robots/chicken_v4/urdf/chicken_v4.urdf` (the
v7 robot; the old v6.5 one is retired). After editing the URDF, regenerate the USD:

```bash
uv run --extra importers python scripts/convert_robot.py
```

The importer writes `src/chicken/assets/data/robots/usd/chicken_v4/chicken_v4.usda`, or `chicken_v4_1`,
`chicken_v4_2`, ... if that folder already exists. Point `CHICKEN_USD_PATH` in `src/chicken/assets/chicken.py` at
the new file, delete the old folders you don't need, and commit the result.

The script also corrects each body's center of mass after converting. The importer re-anchors some links (feet,
IMU, battery, ...) at their mesh offset but leaves the URDF's inertial origins unchanged, which puts their mass in
the wrong place (each foot's mass ended up ~0.31 m above the foot). To apply only that correction to a USD you
converted some other way:

```bash
uv run python scripts/convert_robot.py --fix_only src/chicken/assets/data/robots/usd/<folder>/chicken_v4.usda
```

Then check `src/chicken/assets/chicken.py`:

- **`CHICKEN_BODY_PATHS`**: update it if you add, remove or rename links. The 3.x importer nests each link under
  its parent, so the sensors need each link's full path.
- **`CHICKEN_JOINTS`**: update it if joint names change.
- **Spawn height (`init_state.pos`)**: the root body (`hip`) sits at the URDF origin. Set the height so the feet
  start about 1 cm above the terrain (which is at z = -0.02 at the spawn points) with the legs at `joint_pos`.

Also check `GROUND_CONTACT_BODIES` in `env_cfg.py` (bodies that end the episode when they touch the ground) and
the foot and IMU body names used in `env.py` (`lfoot`, `rfoot`, `imu`).

## Changing the ground

The terrain is `src/chicken/assets/data/terrain/ground.usda`, which loads the mesh from `env10.usd` and adds the
collider; the scale, friction and material are set in `env_cfg.py`. If you swap in a different terrain:

- Keep it a **static collider**: a collision mesh with no rigid body on it. A kinematic rigid body makes the robots fall
  through it, and the `newton_gl` viewer crashes.
- Keep `collision_group=-1` on the ground in `env_cfg.py`, or the robots won't collide with it.
- Check the terrain surface is just below z = 0 at the spawn points, the way the current one is.

## Project layout

| Path                                                       | What it is                                          |
| ---------------------------------------------------------- | --------------------------------------------------- |
| `src/chicken/tasks/walk_direct/config/chicken/env.py`      | The environment: observations, rewards, resets      |
| `src/chicken/tasks/walk_direct/config/chicken/env_cfg.py`  | Scene (ground, robot, sensors, lights) and physics  |
| `src/chicken/tasks/walk_direct/config/chicken/agents/`     | skrl PPO settings                                   |
| `src/chicken/tasks/walk_direct/config/chicken/__init__.py` | Task registration                                   |
| `src/chicken/assets/chicken.py`                            | Robot config: USD path, actuators, joints, spawn    |
| `src/chicken/assets/data/robots/`                          | URDF and meshes (`chicken_v4/`), generated USD (`usd/`) |
| `src/chicken/assets/data/terrain/`                         | Ground: `ground.usda` and the `env10.usd` mesh      |
| `scripts/convert_robot.py`                                 | URDF to USD conversion                              |
| `scripts/control.py`                                       | Keyboard driving of a trained policy                |
| `scripts/export.py`                                        | ONNX and C export for the firmware                  |
| `models/`                                                  | Older robot USDs from the 2.x project (unused)      |

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
