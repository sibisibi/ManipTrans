"""Observation terms matching compute_observations_side, right-hand block then left-hand block."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from . import jit_math as JM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env):
    return env.command_manager.get_term("tracking")


def proprioception(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd = _cmd(env)
    outs = []
    for s in ("rh", "lh"):
        st = cmd.side_states(s)
        base = torch.cat([torch.zeros_like(st["base_state"][:, :3]), st["base_state"][:, 3:]], dim=-1)
        outs.append(torch.cat([st["q"], st["cos_q"], st["sin_q"], base], dim=-1))
    return torch.cat(outs, dim=-1)


def privileged(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd = _cmd(env)
    outs = []
    grav = 9.8
    for s in ("rh", "lh"):
        st = cmd.side_states(s)
        obj = cmd.objects[s]
        tip_force = cmd.tip_object_force(s)  # own-object force, same definition as reward
        tip_force = torch.cat([tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1)

        com_b = obj.data.com_pos_b[:, 0, :]  # body-frame COM offset
        cur_com = (
            JM.quat_to_rotmat(st["manip_obj_quat"][:, [3, 0, 1, 2]]) @ com_b.unsqueeze(-1)
        ).squeeze(-1) + st["manip_obj_pos"]

        mass = obj.data.default_mass.to(env.device)[:, 0]
        outs.append(
            torch.cat(
                [
                    st["dq"],
                    st["manip_obj_pos"] - st["base_state"][:, :3],
                    st["manip_obj_quat"],
                    st["manip_obj_vel"],
                    st["manip_obj_ang_vel"],
                    tip_force.reshape(env.num_envs, -1),
                    cur_com - st["base_state"][:, :3],
                    (mass * grav).unsqueeze(-1),
                ],
                dim=-1,
            )
        )
    return torch.cat(outs, dim=-1)


def target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd = _cmd(env)
    nE = env.num_envs
    idx = cmd.obs_idx
    outs = []
    for s in ("rh", "lh"):
        st = cmd.side_states(s)
        demo = cmd.demo[s]
        t = {}

        target_wrist_pos = demo["wrist_pos"][idx]
        t["delta_wrist_pos"] = target_wrist_pos - st["base_state"][:, :3]

        target_wrist_vel = demo["wrist_velocity"][idx]
        t["wrist_vel"] = target_wrist_vel
        t["delta_wrist_vel"] = target_wrist_vel - st["base_state"][:, 7:10]

        target_wrist_quat = JM.aa_to_quat(demo["wrist_rot"][idx])[:, [1, 2, 3, 0]]
        t["wrist_quat"] = target_wrist_quat
        t["delta_wrist_quat"] = JM.quat_mul(st["base_state"][:, 3:7], JM.quat_conjugate(target_wrist_quat))

        target_wrist_ang_vel = demo["wrist_angular_velocity"][idx]
        t["wrist_ang_vel"] = target_wrist_ang_vel
        t["delta_wrist_ang_vel"] = target_wrist_ang_vel - st["base_state"][:, 10:13]

        target_joints_pos = demo["mano_joints"][idx].reshape(nE, -1, 3)
        t["delta_joints_pos"] = (target_joints_pos - st["joints_state"][:, 1:, :3]).reshape(nE, -1)

        target_joints_vel = demo["mano_joints_velocity"][idx].reshape(nE, -1, 3)
        t["joints_vel"] = target_joints_vel.reshape(nE, -1)
        t["delta_joints_vel"] = (target_joints_vel - st["joints_state"][:, 1:, 7:10]).reshape(nE, -1)

        transf = demo["obj_trajectory"][idx]
        t["delta_manip_obj_pos"] = transf[:, :3, 3] - st["manip_obj_pos"]

        target_obj_vel = demo["obj_velocity"][idx]
        t["manip_obj_vel"] = target_obj_vel
        t["delta_manip_obj_vel"] = target_obj_vel - st["manip_obj_vel"]

        target_obj_quat = JM.rotmat_to_quat(transf[:, :3, :3])[:, [1, 2, 3, 0]]
        t["manip_obj_quat"] = target_obj_quat
        t["delta_manip_obj_quat"] = JM.quat_mul(st["manip_obj_quat"], JM.quat_conjugate(target_obj_quat))

        target_obj_ang_vel = demo["obj_angular_velocity"][idx]
        t["manip_obj_ang_vel"] = target_obj_ang_vel
        t["delta_manip_obj_ang_vel"] = target_obj_ang_vel - st["manip_obj_ang_vel"]

        t["obj_to_joints"] = torch.norm(
            st["manip_obj_pos"][:, None] - st["joints_state"][:, :, :3], dim=-1
        ).reshape(nE, -1)

        t["gt_tips_distance"] = demo["tips_distance"][idx]
        t["bps"] = demo["bps"].unsqueeze(0).expand(nE, -1)

        outs.append(
            torch.cat(
                [
                    t[k]
                    for k in [
                        "delta_wrist_pos", "wrist_vel", "delta_wrist_vel", "wrist_quat", "delta_wrist_quat",
                        "wrist_ang_vel", "delta_wrist_ang_vel", "delta_joints_pos", "joints_vel", "delta_joints_vel",
                        "delta_manip_obj_pos", "manip_obj_vel", "delta_manip_obj_vel", "manip_obj_quat",
                        "delta_manip_obj_quat", "manip_obj_ang_vel", "delta_manip_obj_ang_vel", "obj_to_joints",
                        "gt_tips_distance", "bps",
                    ]
                ],
                dim=-1,
            )
        )
    return torch.cat(outs, dim=-1)
