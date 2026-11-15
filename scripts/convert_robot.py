"""Convert the chicken URDF into the USD that the task loads.

The 2.x project re-ran this conversion every time the robot config was imported. In 3.x it is a
one-off step: run it again only after changing the URDF, then commit the regenerated USD.

    uv run --extra importers python scripts/convert_robot.py

The settings below match the 2.x ``UrdfConverterCfg`` used by ``assets/robots/chicken.py``.
"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.version import standalone_importers_available

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
add_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.require_kit = not standalone_importers_available()
args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402

from chicken.assets import CHICKEN_ASSETS_DIR  # noqa: E402

URDF_PATH = CHICKEN_ASSETS_DIR / "robots" / "chicken_v6_5" / "urdf" / "chicken.urdf"
USD_DIR = CHICKEN_ASSETS_DIR / "robots" / "usd"


def main():
    cfg = UrdfConverterCfg(
        asset_path=str(URDF_PATH),
        usd_dir=str(USD_DIR),
        fix_base=False,
        force_usd_conversion=True,
        merge_fixed_joints=False,
        link_density=1250,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=20.0, damping=0.5),
        ),
        collision_from_visuals=True,
        self_collision=False,
        collision_type="Convex Hull",
    )
    with launch_simulation(cfg=sim_utils.SimulationCfg(), launcher_args=args_cli):
        converter = UrdfConverter(cfg)
        print(f"Generated USD file: {converter.usd_path}")


if __name__ == "__main__":
    main()
