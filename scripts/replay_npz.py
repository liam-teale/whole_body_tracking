"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py --registry_name <wandb-registry-name>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib
import numpy as np
import torch
import wandb

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# download motion file before launching Isaac Sim so the timeline doesn't stall
registry_name = args_cli.registry_name
if ":" not in registry_name:
    registry_name += ":latest"
api = wandb.Api()
artifact = api.artifact(registry_name)
motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

# IsaacLab v6 launches headless unless a Kit visualizer is requested. This is a replay
# tool, so default to the GUI viewport unless the user opted out (--headless or --viz).
if getattr(args_cli, "visualizer", None) is None and not args_cli.headless:
    args_cli.visualizer = ["kit"]
    args_cli.visualizer_explicit = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_pos = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_quat = motion.body_quat_w[time_steps][:, 0]
        root_lin_vel = motion.body_lin_vel_w[time_steps][:, 0]
        root_ang_vel = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_pose_to_sim(root_pose=torch.cat([root_pos, root_quat], dim=-1))
        robot.write_root_velocity_to_sim(root_velocity=torch.cat([root_lin_vel, root_ang_vel], dim=-1))
        robot.write_joint_position_to_sim(position=motion.joint_pos[time_steps])
        robot.write_joint_velocity_to_sim(velocity=motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.step(render=True)  # kinematic replay — physics output is overwritten each frame
        scene.update(sim_dt)

        pos_lookat = root_pos[0].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
