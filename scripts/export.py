"""Export a trained skrl policy to ONNX, and to C with onnx2c for the robot firmware.

    uv run --extra export python scripts/export.py --checkpoint latest --onnx2c ~/onnx2c/build/onnx2c

Writes to ``exports/<run>_<checkpoint>/``:

- ``policy.onnx``: input ``(1, num_obs)`` raw observations, output ``(1, 8)`` mean actions.
- ``nn.c`` / ``nn.h``: ``void entry(const float tensor_input[1][num_obs], float tensor_output[1][8])``,
  a drop-in replacement for ``lib/nn`` in the mainboard firmware. Skipped when onnx2c isn't found.

Observation normalization is part of the exported model. During training and ``isaaclab play``, skrl's
``RunningStandardScaler`` normalizes observations before the policy sees them; the 2.x export left that out,
so a firmware feeding raw sensor values got inputs on a different scale. Pass ``--no_normalization`` only
if your firmware normalizes the observations itself.
"""

import argparse
import os
import shutil
import subprocess
import sys

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--task", type=str, default="Chicken-Walk-Direct", help="Name of the task.")
parser.add_argument(
    "--checkpoint",
    type=str,
    default="latest",
    help="'latest', 'best', or a path to a checkpoint .pt file (default: latest).",
)
parser.add_argument("--output_dir", type=str, default="exports", help="Parent folder for exports.")
parser.add_argument(
    "--onnx2c",
    type=str,
    default=None,
    help="Path to the onnx2c binary. Defaults to onnx2c/build/onnx2c in this repo, then onnx2c on PATH.",
)
parser.add_argument("--precision", type=int, default=9, help="Digits for weights in the C code (onnx2c -P).")
parser.add_argument(
    "--no_normalization",
    action="store_true",
    help="Leave observation normalization out of the exported model (the 2.x behavior).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser, sys.argv[1:])
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from isaaclab_rl.entrypoints.common import CHECKPOINT_SELECTORS, resolve_checkpoint_selector  # noqa: E402
from isaaclab_rl.skrl import (  # noqa: E402
    SkrlVecEnvWrapper,
    import_skrl_runner,
    resolve_skrl_agent_cfg_entry_point,
)

from isaaclab_tasks.utils import resolve_task_config  # noqa: E402

import chicken.tasks  # noqa: E402, F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ExportedPolicy(torch.nn.Module):
    """Raw observations -> normalized observations -> deterministic (mean) actions."""

    def __init__(self, agent, normalize: bool):
        super().__init__()
        self.policy = agent.policy
        self.normalize = False
        scaler = agent._observation_preprocessor
        if normalize and isinstance(scaler, torch.nn.Module):
            # Same math as skrl's RunningStandardScaler at play time, baked in as float32 constants.
            self.normalize = True
            self.register_buffer("obs_mean", scaler.running_mean.float().clone())
            self.register_buffer("obs_std", (torch.sqrt(scaler.running_variance.float()) + scaler.epsilon).clone())
            self.clip = float(scaler.clip_threshold)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            obs = torch.clamp((obs - self.obs_mean) / self.obs_std, -self.clip, self.clip)
        # compute() is the deterministic network pass; act() would also sample from the action distribution.
        mean_actions, _ = self.policy.compute({"observations": obs, "states": None}, role="policy")
        return mean_actions


def find_onnx2c() -> str | None:
    if args_cli.onnx2c:
        path = os.path.expanduser(args_cli.onnx2c)
        if not os.path.isfile(path):
            raise SystemExit(f"onnx2c not found at {path}")
        return path
    in_repo = os.path.join(REPO_ROOT, "onnx2c", "build", "onnx2c")
    return in_repo if os.path.isfile(in_repo) else shutil.which("onnx2c")


def resolve_checkpoint(log_root_path: str) -> str:
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="skrl",
            task=args_cli.task,
            checkpoint_pattern=r".*",
            other_dirs=["checkpoints"],
            latest_checkpoint_pattern=r"agent_.*",
        )
    return os.path.abspath(os.path.expanduser(args_cli.checkpoint))


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, resolve_skrl_agent_cfg_entry_point(None, None))
    env_cfg.scene.num_envs = 1

    with launch_simulation(env_cfg, args_cli):
        log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
        checkpoint = resolve_checkpoint(log_root_path)
        run_name = os.path.basename(os.path.dirname(os.path.dirname(checkpoint)))
        out_dir = os.path.join(args_cli.output_dir, f"{run_name}_{os.path.splitext(os.path.basename(checkpoint))[0]}")
        os.makedirs(out_dir, exist_ok=True)

        # The environment is only needed so skrl can build the models with the right input/output sizes.
        env = SkrlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), ml_framework="torch")
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        agent_cfg["agent"]["experiment"]["write_interval"] = 0
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        runner = import_skrl_runner("torch")(env, agent_cfg)
        print(f"[export] Loading checkpoint: {checkpoint}")
        runner.agent.load(checkpoint)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        model = ExportedPolicy(runner.agent, normalize=not args_cli.no_normalization).eval()
        num_obs = env.observation_space.shape[-1]
        print(f"[export] Observations: {num_obs}, normalization in model: {model.normalize}")

        # Check the wrapper against skrl's own inference path on a real observation from the env.
        obs, _ = env.reset()
        with torch.inference_mode():
            reference = runner.agent.act(obs, None, timestep=0, timesteps=0)[-1]["mean_actions"]
            ours = model(obs)
        if not args_cli.no_normalization and not torch.allclose(reference, ours, atol=1e-5):
            raise SystemExit(f"Exported model disagrees with skrl's play output: {(reference - ours).abs().max():.3g}")
        sample = obs[:1].detach().clone()
        env.close()

    # Export and check outside the simulation context; only CPU tensors from here on.
    model = model.cpu()
    sample = sample.cpu()
    onnx_path = os.path.join(out_dir, "policy.onnx")
    torch.onnx.export(
        model, (sample,), onnx_path, input_names=["input"], output_names=["output"], opset_version=17, dynamo=False
    )
    onnx.checker.check_model(onnx.load(onnx_path))

    with torch.inference_mode():
        expected = model(sample).numpy()
    got = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"]).run(None, {"input": sample.numpy()})[0]
    err = float(np.abs(expected - got).max())
    print(f"[export] ONNX vs PyTorch max difference: {err:.3g}")
    if err > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")
    print(f"[export] Saved {onnx_path}")

    onnx2c = find_onnx2c()
    if onnx2c is None:
        print("[export] onnx2c not found; skipping C generation. Pass --onnx2c <path> to generate nn.c.")
        return
    c_code = subprocess.run(
        [onnx2c, f"-P{args_cli.precision}", onnx_path], capture_output=True, text=True, check=True
    ).stdout
    with open(os.path.join(out_dir, "nn.c"), "w", encoding="utf-8") as f:
        f.write(c_code)
    num_actions = got.shape[-1]
    with open(os.path.join(out_dir, "nn.h"), "w", encoding="utf-8") as f:
        f.write(
            "#pragma once\n\n#ifdef __cplusplus\nextern \"C\"\n{\n#endif\n\n"
            f"    void entry(const float tensor_input[1][{num_obs}], float tensor_output[1][{num_actions}]);\n\n"
            "#ifdef __cplusplus\n}\n#endif\n"
        )
    print(f"[export] Saved {out_dir}/nn.c and nn.h (copy both into the firmware's lib/nn/)")


if __name__ == "__main__":
    main()
