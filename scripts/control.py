"""Drive a trained chicken policy with the keyboard.

    PYGLFW_LIBRARY_VARIANT=x11 uv run python scripts/control.py --checkpoint latest --viz newton_gl

Click the viewer window so it has keyboard focus, then:

- I / K (or numpad 8 / 2): walk forward / backward
- J / L (or numpad 4 / 6): side-step left / right
- U / O (or numpad 7 / 9): turn left / right
- R: reset the robots

The newton_gl viewer moves the camera with WASD, the arrow keys and Q/E, so driving uses other keys.

Commands ramp toward the held keys rather than jumping, as in the 2.x script. While teleop is on, the
environment ignores its random velocity commands and episodes don't time out; a robot is only reset if
it falls within its first 60 steps (see ``ChickenEnv._get_dones``).
"""

import argparse
import contextlib
import logging
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_rl.entrypoints.backends import cli_args_skrl
from isaaclab_rl.entrypoints.common import add_common_play_args, set_hydra_args

from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
add_common_play_args(parser, agent_default=None, agent_help="Agent configuration entry point.")
parser.add_argument("--checkpoint", type=str, default="latest", help="'latest', 'best', or a checkpoint path.")
parser.add_argument("--v_x_sensitivity", type=float, default=0.5, help="Forward/backward speed at full key [m/s].")
parser.add_argument("--v_y_sensitivity", type=float, default=0.5, help="Side-step speed at full key [m/s].")
parser.add_argument("--yaw_sensitivity", type=float, default=2.0, help="Turn rate at full key [rad/s].")
parser.add_argument("--smoothing", type=float, default=0.2, help="Fraction of the way to the target per step.")
parser.add_argument("--step_frequency", type=float, default=1.8, help="Gait frequency passed to the env [Hz].")
cli_args_skrl.add_skrl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
set_hydra_args(hydra_args)
if args_cli.task is None:
    args_cli.task = "Chicken-Walk-Direct"

import gymnasium as gym  # noqa: E402

from isaaclab_rl.entrypoints.common import (  # noqa: E402
    CHECKPOINT_SELECTORS,
    apply_env_overrides,
    apply_video_recording,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    run_playback,
)
from isaaclab_rl.skrl import (  # noqa: E402
    SkrlVecEnvWrapper,
    import_skrl_runner,
    resolve_skrl_agent_cfg_entry_point,
)

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import chicken.tasks  # noqa: E402, F401

logger = logging.getLogger(__name__)

# W3C key codes reported by the viewer window -> (command index, sign). The numpad layout matches 2.x
# Se2Keyboard; the letter keys avoid the newton_gl camera keys (WASD, arrows, Q/E) and its Space/H/F/Esc.
KEY_BINDINGS = {
    "KeyI": (0, +1.0),
    "Numpad8": (0, +1.0),
    "KeyK": (0, -1.0),
    "Numpad2": (0, -1.0),
    "KeyJ": (1, +1.0),
    "Numpad4": (1, +1.0),
    "KeyL": (1, -1.0),
    "Numpad6": (1, -1.0),
    "KeyU": (2, +1.0),
    "Numpad7": (2, +1.0),
    "KeyO": (2, -1.0),
    "Numpad9": (2, -1.0),
}
RESET_KEY = "KeyR"


class Keyboard:
    """Held keys from every visualizer window that reports them, read as an SE(2) command."""

    def __init__(self, visualizers):
        self.held: set[str] = set()
        self.reset_requested = False
        self.subscriptions = []
        for visualizer in visualizers:
            source = visualizer.key_event_source
            if source is not None:
                self.subscriptions.append(source.add_key_listener(self._on_key, self.held.clear))
        if not self.subscriptions:
            logger.warning("No visualizer window reports keys. Run with --viz newton_gl to drive the robot.")

    def _on_key(self, code: str, pressed: bool) -> None:
        if code == RESET_KEY and pressed:
            self.reset_requested = True
        elif code in KEY_BINDINGS:
            (self.held.add if pressed else self.held.discard)(code)

    def command(self) -> tuple[float, float, float]:
        cmd = [0.0, 0.0, 0.0]
        for code in self.held:
            index, sign = KEY_BINDINGS[code]
            cmd[index] += sign
        scale = (args_cli.v_x_sensitivity, args_cli.v_y_sensitivity, args_cli.yaw_sensitivity)
        return tuple(max(-1.0, min(1.0, c)) * s for c, s in zip(cmd, scale, strict=True))

    def close(self) -> None:
        for subscription in self.subscriptions:
            subscription.close()


def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b


def main():
    env_cfg, agent_cfg = resolve_task_config(
        args_cli.task, resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm), play_mode=True
    )
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        apply_env_overrides(args_cli, env_cfg)
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = agent_cfg["seed"]

        experiment_cfg = agent_cfg["agent"]["experiment"]
        log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["directory"]))
        if args_cli.checkpoint in CHECKPOINT_SELECTORS:
            checkpoint = resolve_checkpoint_selector(
                log_root_path,
                args_cli.checkpoint,
                library="skrl",
                task=args_cli.task,
                checkpoint_pattern=r".*",
                other_dirs=["checkpoints"],
                latest_checkpoint_pattern=r"agent_.*",
            )
        else:
            checkpoint = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
        log_dir = os.path.dirname(os.path.dirname(checkpoint))
        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="control")

        env = gym.make(args_cli.task, cfg=env_cfg)
        cleanup.callback(env.close)
        chicken = env.unwrapped
        chicken.is_teleop = True
        dt = chicken.step_dt

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["write_interval"] = 0
        experiment_cfg["checkpoint_interval"] = 0
        runner = import_skrl_runner(args_cli.ml_framework)(env, agent_cfg)
        logger.info(f"Loading model checkpoint from: {checkpoint}")
        print(f"[control] Loading checkpoint: {checkpoint}")
        runner.agent.load(checkpoint)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        keyboard = Keyboard(chicken.sim.visualizers)
        cleanup.callback(keyboard.close)
        print("[control] Click the viewer window, then: I/K walk, J/L side-step, U/O turn, R reset.")

        obs, _ = env.reset()
        states = env.state()
        v_x = v_y = yaw_rate = 0.0

        def step() -> None:
            nonlocal obs, states, v_x, v_y, yaw_rate
            if keyboard.reset_requested:
                keyboard.reset_requested = False
                # skrl's wrapper only resets the env on its first reset() call and returns cached
                # observations after that, so re-arm it.
                env._reset_once = True
                obs, _ = env.reset()
                states = env.state()
                v_x = v_y = yaw_rate = 0.0
            cmd = keyboard.command()
            t = args_cli.smoothing
            # The chicken env's forward command is negative target_vel (as in the 2.x control script).
            v_x = lerp(v_x, -cmd[0], t)
            v_y = lerp(v_y, cmd[1], t)
            yaw_rate = lerp(yaw_rate, cmd[2], t)
            chicken.set_teleop_commands(v_x=v_x, v_y=v_y, yaw_rate=yaw_rate, freq_t=args_cli.step_frequency)

            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
            states = env.state()

        run_playback(step, dt=dt, args_cli=args_cli, env_cfg=env_cfg)


if __name__ == "__main__":
    main()
