"""Convert an Isaac/UsdPhysics USD into a Newton-friendly model snapshot.

This script fixes common USD issues (CollisionAPI on non-GPrim, zero mass),
then loads the stage into a Newton ModelBuilder and resets joint drive
parameters before saving a Newton model/state snapshot.
"""

from __future__ import annotations

import argparse

import warp as wp

import newton

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except Exception as exc:  # pragma: no cover
    raise RuntimeError("pxr (usd-core) is required to process USD files.") from exc


def _fix_collision_api(stage: Usd.Stage) -> tuple[int, int, int]:
    moved = 0
    removed = 0
    orphaned = 0

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdGeom.Gprim(prim):
            continue

        mesh_child = None
        for child in prim.GetChildren():
            if UsdGeom.Mesh(child):
                mesh_child = child
                break

        if mesh_child is not None:
            UsdPhysics.CollisionAPI.Apply(mesh_child)
            moved += 1
        else:
            orphaned += 1

        UsdPhysics.CollisionAPI(prim).ApplyAPI(False)
        removed += 1

    return moved, removed, orphaned


def _fix_zero_mass(stage: Usd.Stage, min_mass: float) -> int:
    fixed = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.MassAPI):
            continue
        mass_attr = UsdPhysics.MassAPI(prim).GetMassAttr()
        if mass_attr.HasValue() and mass_attr.Get() == 0.0:
            mass_attr.Set(min_mass)
            fixed += 1
    return fixed


def _reset_joint_drives(builder: newton.ModelBuilder, ke: float, kd: float) -> None:
    if builder.joint_dof_count == 0:
        return
    for i in range(builder.joint_dof_count):
        builder.joint_target_ke[i] = ke
        builder.joint_target_kd[i] = kd
        builder.joint_target_pos[i] = builder.joint_q[i]
    if ke > 0.0:
        for i in range(builder.joint_dof_count):
            builder.joint_act_mode[i] = int(newton.ActuatorMode.POSITION)


def convert_usd(
    src: str,
    dst_usd: str | None,
    out_model: str | None,
    min_mass: float,
    joint_ke: float,
    joint_kd: float,
    add_ground: bool,
    device: str,
) -> None:
    wp.set_device(device)
    stage = Usd.Stage.Open(src)
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {src}")

    moved, removed, orphaned = _fix_collision_api(stage)
    mass_fixed = _fix_zero_mass(stage, min_mass)

    if dst_usd:
        stage.GetRootLayer().Export(dst_usd)

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_usd(
        stage,
        apply_up_axis_from_stage=True,
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        load_visual_shapes=True,
        hide_collision_shapes=True,
    )
    if add_ground:
        builder.add_ground_plane()

    _reset_joint_drives(builder, joint_ke, joint_kd)

    model = builder.finalize(device=device)
    state_0 = model.state()

    if out_model:
        recorder = newton.utils.RecorderModelAndState()
        recorder.record_model(model)
        recorder.record(state_0)
        recorder.save_to_file(out_model)

    print("USD conversion summary")
    print(f"- collision_moved: {moved}")
    print(f"- collision_removed: {removed}")
    print(f"- collision_orphaned: {orphaned}")
    print(f"- mass_fixed: {mass_fixed}")
    if dst_usd:
        print(f"- saved_fixed_usd: {dst_usd}")
    if out_model:
        print(f"- saved_model: {out_model}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Input USD/USDZ path")
    parser.add_argument("--dst-usd", default="", help="Output fixed USD path")
    parser.add_argument("--out-model", default="", help="Output Newton model (.json/.bin)")
    parser.add_argument("--min-mass", type=float, default=1.0e-3, help="Minimum mass to clamp")
    parser.add_argument("--joint-ke", type=float, default=2.0e4, help="Joint stiffness")
    parser.add_argument("--joint-kd", type=float, default=1.0e2, help="Joint damping")
    parser.add_argument("--no-ground", action="store_true", help="Do not add a ground plane")
    parser.add_argument("--device", type=str, default="cpu", help="Warp device for conversion")
    args = parser.parse_args()

    convert_usd(
        src=args.src,
        dst_usd=args.dst_usd or None,
        out_model=args.out_model or None,
        min_mass=args.min_mass,
        joint_ke=args.joint_ke,
        joint_kd=args.joint_kd,
        add_ground=not args.no_ground,
        device=args.device,
    )


if __name__ == "__main__":
    main()
