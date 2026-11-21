"""Convert the chicken URDF into the USD that the task loads.

The 2.x project re-ran this conversion every time the robot config was imported. In 3.x it is a
one-off step: run it again only after changing the URDF, then commit the regenerated USD.

    uv run --extra importers python scripts/convert_robot.py

The settings below match the 2.x ``UrdfConverterCfg`` used by ``assets/robots/chicken.py``.

After converting, the script corrects each body's center of mass (see ``fix_centers_of_mass``). To apply
only that correction to a USD converted earlier:

    uv run python scripts/convert_robot.py --fix_only src/chicken/assets/data/robots/usd/<dir>/<robot>.usda
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--fix_only", metavar="USD", help="Only correct the centers of mass in an existing USD.")


def _origin(element) -> np.ndarray:
    """4x4 transform of a URDF <origin> element (xyz + fixed-axis rpy)."""
    xyz = [float(v) for v in (element.get("xyz", "0 0 0") if element is not None else "0 0 0").split()]
    roll, pitch, yaw = [float(v) for v in (element.get("rpy", "0 0 0") if element is not None else "0 0 0").split()]
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rot = (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    )
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = rot, xyz
    return out


def fix_centers_of_mass(urdf_path: Path, usd_path: Path) -> None:
    """Express each body's center of mass in the frame the converter gave that body.

    Onshape exports place each mesh with a <visual> origin offset. The converter re-anchors some links at
    that offset (feet, imu, battery, ...) but copies the <inertial> origin unchanged, which is relative to
    the original link frame. Without this fix those bodies' masses sit in the wrong place, e.g. each foot's
    mass ~0.31 m above the foot. The frames are only shifted, not rotated, so inertia is left as is.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    parents = {
        j.find("child").get("link"): (j.find("parent").get("link"), _origin(j.find("origin")))
        for j in root.findall("joint")
    }

    def link_frame(name: str) -> np.ndarray:
        if name not in parents:
            return np.eye(4)
        parent, joint = parents[name]
        return link_frame(parent) @ joint

    stage = Usd.Stage.Open(str(usd_path))
    xforms = UsdGeom.XformCache(0)
    fixes = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or name not in links or links[name].find("inertial") is None:
            continue
        body = np.array(xforms.GetLocalToWorldTransform(prim)).T  # Gf matrices are row-vector
        to_body = np.linalg.inv(body) @ link_frame(name)
        if not np.allclose(to_body[:3, :3], np.eye(3), atol=1e-6):
            raise SystemExit(f"{name}: body frame is rotated relative to the URDF link frame; fix not supported")
        com_link = _origin(links[name].find("inertial/origin"))[:3, 3]
        com_body = to_body[:3, 3] + com_link
        if not np.allclose(com_body, com_link, atol=1e-7):
            fixes[name] = Gf.Vec3f(*com_body.tolist())

    # The mass properties are authored in the Physics payload layer.
    layer = Sdf.Layer.FindOrOpen(str(usd_path.parent / "payloads" / "Physics" / "physics.usda"))
    done = set()

    def visit(spec):
        if spec.name in fixes and "physics:centerOfMass" in spec.attributes:
            spec.attributes["physics:centerOfMass"].default = fixes[spec.name]
            done.add(spec.name)
        for child in spec.nameChildren:
            visit(child)

    for prim_spec in layer.rootPrims:
        visit(prim_spec)
    if done != set(fixes):
        raise SystemExit(f"Could not find centerOfMass for: {sorted(set(fixes) - done)}")
    layer.Save()
    for name in sorted(fixes):
        print(f"Fixed center of mass: {name} -> {tuple(round(v, 4) for v in fixes[name])}")


URDF_NAME = ("robots", "chicken_v4", "urdf", "chicken_v4.urdf")


def main():
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab.utils.version import standalone_importers_available

    add_launcher_args(parser)
    args_cli = parser.parse_args()
    from chicken.assets import CHICKEN_ASSETS_DIR

    urdf_path = CHICKEN_ASSETS_DIR.joinpath(*URDF_NAME)
    if args_cli.fix_only:
        fix_centers_of_mass(urdf_path, Path(args_cli.fix_only).resolve())
        return
    args_cli.require_kit = not standalone_importers_available()
    args_cli.physics = "isaacsim_physx" if args_cli.require_kit else "newton_mjwarp"

    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        # The importer writes <usd_dir>/<robot name>/<robot name>.usda, adding _1, _2 ... if that folder exists.
        usd_dir=str(CHICKEN_ASSETS_DIR / "robots" / "usd"),
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
    fix_centers_of_mass(urdf_path, Path(converter.usd_path))
    print("Point CHICKEN_USD_PATH in src/chicken/assets/chicken.py at the generated file.")


if __name__ == "__main__":
    sys.exit(main())
