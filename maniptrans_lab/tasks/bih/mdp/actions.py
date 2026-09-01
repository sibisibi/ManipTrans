"""Action term porting dexhandmanip_bih.pre_physics_step with the base half of the action replaced from the retargeted trajectory."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import dexhand_artimano as HAND
from . import jit_math as JM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

ROOT_CONTROL_DIM = 9  # PID mode
N = HAND.N_DOFS


class BiHManipAction(ActionTerm):

    cfg: "BiHManipActionCfg"

    def __init__(self, cfg: "BiHManipActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        dev = self.device
        n = self.num_envs
        self.dt = env.step_dt  # gym's control timestep

        self.command = env.command_manager.get_term(cfg.command_name)
        self.robots = {"rh": env.scene["dexhand_r"], "lh": env.scene["dexhand_l"]}
        self.palm_id = {}
        for s in ("rh", "lh"):
            ids, _ = self.robots[s].find_bodies(["palm"])
            self.palm_id[s] = ids

        self._raw_actions = torch.zeros(n, self.action_dim, device=dev)
        self.prev_targets = {s: torch.zeros(n, N, device=dev) for s in ("rh", "lh")}
        self.apply_forces = {s: torch.zeros(n, 3, device=dev) for s in ("rh", "lh")}
        self.apply_torque = {s: torch.zeros(n, 3, device=dev) for s in ("rh", "lh")}
        self.pos_error_integral = {s: torch.zeros(n, 3, device=dev) for s in ("rh", "lh")}
        self.rot_error_integral = {s: torch.zeros(n, 3, device=dev) for s in ("rh", "lh")}
        self.prev_pos_error = {s: torch.zeros(n, 3, device=dev) for s in ("rh", "lh")}
        self.prev_rot_error = {s: torch.zeros(n, 3, device=dev) for s in ("rh", "lh")}

    # ------------------------------------------------------------- properties

    @property
    def action_dim(self) -> int:
        return 2 * (ROOT_CONTROL_DIM + N) + 2 * (6 + N)  # 118

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    # ---------------------------------------------------------------- helpers

    def _retargeted_base_side(self, s: str) -> torch.Tensor:
        cmd = self.command
        demo = cmd.demo[s]
        idx = cmd.frame_buf
        robot: Articulation = self.robots[s]
        origins = self._env.scene.env_origins

        ref_wrist_pos = demo["opt_wrist_pos"][idx]
        ref_wrist_rotmat = JM.aa_to_rotmat(demo["opt_wrist_rot"][idx])
        ref_dof_pos = demo["opt_dof_pos"][idx]

        cur_wrist_pos = robot.data.root_pos_w - origins
        cur_wrist_rotmat = JM.quat_to_rotmat(robot.data.root_quat_w)  # wxyz native

        pos_error = (ref_wrist_pos - cur_wrist_pos) / self.cfg.wrist_pos_scale
        rot_error = JM.rotmat_to_rot6d(ref_wrist_rotmat @ cur_wrist_rotmat.transpose(-1, -2))
        dof_target = JM.unscale(ref_dof_pos, demo["dof_lower"], demo["dof_upper"])
        return torch.cat([pos_error, rot_error, dof_target], dim=-1)

    # -------------------------------------------------------------- interface

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        base = torch.clamp(
            torch.cat([self._retargeted_base_side("rh"), self._retargeted_base_side("lh")], dim=-1), -1, 1
        )
        res = actions[:, 2 * (ROOT_CONTROL_DIM + N):] * 2  # gym residual doubling

        ma = self.cfg.act_moving_average
        for k, s in enumerate(("rh", "lh")):
            b = base[:, k * (ROOT_CONTROL_DIM + N): (k + 1) * (ROOT_CONTROL_DIM + N)]
            r = res[:, k * (6 + N): (k + 1) * (6 + N)]
            demo = self.command.demo[s]

            dof_pos = torch.clamp(b[:, ROOT_CONTROL_DIM:] + r[:, 6:], -1, 1)
            targets = JM.scale(dof_pos, demo["dof_lower"], demo["dof_upper"])
            targets = ma * targets + (1.0 - ma) * self.prev_targets[s]
            targets = JM.tensor_clamp(targets, demo["dof_lower"], demo["dof_upper"])
            self.prev_targets[s] = targets

            # wrist PID (gym lines, verbatim math)
            pos_err = b[:, 0:3]
            self.pos_error_integral[s] = torch.clamp(self.pos_error_integral[s] + pos_err * self.dt, -1, 1)
            pos_der = (pos_err - self.prev_pos_error[s]) / self.dt
            force = HAND.KP_POS * pos_err + HAND.KI_POS * self.pos_error_integral[s] + HAND.KD_POS * pos_der
            self.prev_pos_error[s] = pos_err
            force = force + r[:, 0:3] * self.dt * self.cfg.translation_scale * 500

            rot_err = JM.rot6d_to_aa(b[:, 3:9])
            self.rot_error_integral[s] = torch.clamp(self.rot_error_integral[s] + rot_err * self.dt, -1, 1)
            rot_der = (rot_err - self.prev_rot_error[s]) / self.dt
            torque = HAND.KP_ROT * rot_err + HAND.KI_ROT * self.rot_error_integral[s] + HAND.KD_ROT * rot_der
            self.prev_rot_error[s] = rot_err
            torque = torque + r[:, 3:6] * self.dt * self.cfg.orientation_scale * 200

            self.apply_forces[s] = ma * force + (1.0 - ma) * self.apply_forces[s]
            self.apply_torque[s] = ma * torque + (1.0 - ma) * self.apply_torque[s]

            robot: Articulation = self.robots[s]
            robot.set_joint_position_target(targets[:, self.command.lab_from_gym[s]])
            robot.set_external_force_and_torque(
                self.apply_forces[s].unsqueeze(1), self.apply_torque[s].unsqueeze(1),
                body_ids=self.palm_id[s], is_global=True,
            )

    def apply_actions(self):
        # targets and wrenches persist in the articulation buffers across the decimation loop
        pass

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        for s in ("rh", "lh"):
            self.prev_targets[s][env_ids] = 0
            self.apply_forces[s][env_ids] = 0
            self.apply_torque[s][env_ids] = 0
            self.pos_error_integral[s][env_ids] = 0
            self.rot_error_integral[s][env_ids] = 0
            self.prev_pos_error[s][env_ids] = 0
            self.prev_rot_error[s][env_ids] = 0
        self._raw_actions[env_ids] = 0


@configclass
class BiHManipActionCfg(ActionTermCfg):
    class_type: type = BiHManipAction
    asset_name: str = "dexhand_r"
    command_name: str = "tracking"
    act_moving_average: float = 0.4
    translation_scale: float = MISSING
    orientation_scale: float = MISSING
    wrist_pos_scale: float = 0.1
