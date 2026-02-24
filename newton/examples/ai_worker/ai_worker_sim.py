"""Load a URDF and simulate it."""

from __future__ import annotations

from pathlib import Path

import warp as wp

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_urdf(
            args.urdf,
            collapse_fixed_joints=False,
            enable_self_collisions=True,
        )
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collisions_enabled = True
        self.collision_pipeline = None
        self.contacts = None
        if self.collisions_enabled:
            if args is None or not hasattr(args, "collision_pipeline"):
                self.collision_pipeline = newton.examples.create_collision_pipeline(
                    self.model, collision_pipeline_type="standard"
                )
            else:
                self.collision_pipeline = newton.examples.create_collision_pipeline(self.model, args)
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)

        self.viewer.set_model(self.model)

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collisions_enabled:
                self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.collisions_enabled and self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def main():
    parser = newton.examples.create_parser()
    default_urdf = str(Path(__file__).resolve().parent / "ai_worker.urdf")
    parser.add_argument(
        "--urdf",
        type=str,
        default=default_urdf,
        help="Path to a URDF file.",
    )
    parser.add_argument(
        "--enable-collisions",
        action="store_true",
        help="Enable collision pipeline.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)


if __name__ == "__main__":
    main()
