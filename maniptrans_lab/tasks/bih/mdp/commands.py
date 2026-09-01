"""Command term owning the demo clip, its frame timing, reference state initialization, the curriculum, and the joint kernel pass."""

from __future__ import annotations

import math
import os
from dataclasses import MISSING
from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from . import dexhand_artimano as HAND
from . import jit_math as JM

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BiHTrackingCommand(CommandTerm):

    cfg: "BiHTrackingCommandCfg"

    def __init__(self, cfg: "BiHTrackingCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        dev = self.device
        data = torch.load(cfg.demo_path, map_location=dev)
        self.seq_len = int(data["demo_rh"]["seq_len"])
        self.max_episode_length = int(data["max_episode_length"])

        self.demo = {}
        for side in ("rh", "lh"):
            self.demo[side] = {
                k: (v.to(dev).float() if isinstance(v, torch.Tensor) else v)
                for k, v in data[f"demo_{side}"].items()
            }
            self.demo[side]["bps"] = data[f"bps_{side}"].to(dev).float()
            self.demo[side]["dof_lower"] = data[f"dof_lower_{side}"].to(dev).float()
            self.demo[side]["dof_upper"] = data[f"dof_upper_{side}"].to(dev).float()
            self.demo[side]["dof_speed"] = data[f"dof_speed_{side}"].to(dev).float()

        n = self.num_envs
        self.frame_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.running_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self._just_reset = torch.zeros(n, dtype=torch.bool, device=dev)

        # gym initializes the contact history to all true on reset
        self.tips_contact_history = {
            s: torch.ones(n, cfg.contact_history_len, 5, dtype=torch.bool, device=dev) for s in ("rh", "lh")
        }

        # caches filled per step
        self.target = {"rh": {}, "lh": {}}       # reward-frame targets
        self.obs_idx = torch.zeros(n, dtype=torch.long, device=dev)  # obs frame index
        self._obs_offset = torch.zeros(n, dtype=torch.long, device=dev)
        self.kernel_out = None                    # (rew, reset, success, failure, reward_dict, error)
        self._ss_cache = {}
        self._ss_stamp = -1
        self.scale_factor = 1.0

        self.weight_idx = {k: list(v) for k, v in HAND.WEIGHT_IDX.items()}
        self._arange = torch.arange(n, device=dev)

        # scene assets resolve lazily on first use
        self._resolved = False

    # ------------------------------------------------------------------ setup

    def _resolve(self):
        env = self._env
        self.robots = {"rh": env.scene["dexhand_r"], "lh": env.scene["dexhand_l"]}
        self.objects = {"rh": env.scene["manip_obj_rh"], "lh": env.scene["manip_obj_lh"]}
        s_prefix = {"rh": "contact_rh_", "lh": "contact_lh_"}
        self.tip_sensors = {
            s: [env.scene.sensors[s_prefix[s] + b] for b in HAND.CONTACT_BODY_NAMES] for s in ("rh", "lh")
        }
        self.body_order = {}
        for s in ("rh", "lh"):
            robot: Articulation = self.robots[s]
            names = robot.body_names
            self.body_order[s] = torch.tensor(
                [names.index(b) for b in HAND.BODY_NAMES], dtype=torch.long, device=self.device
            )
            dof_names = robot.joint_names
            assert sorted(dof_names) == sorted(HAND.DOF_NAMES), f"{s} joint set mismatch: {dof_names}"
        # IsaacLab orders joints breadth-first and gym depth-first, so permute at the boundary
        self.lab_from_gym = {}
        self.gym_from_lab = {}
        for s in ("rh", "lh"):
            lab_names = self.robots[s].joint_names
            self.lab_from_gym[s] = torch.tensor(
                [HAND.DOF_NAMES.index(nm) for nm in lab_names], dtype=torch.long, device=self.device
            )
            self.gym_from_lab[s] = torch.tensor(
                [lab_names.index(nm) for nm in HAND.DOF_NAMES], dtype=torch.long, device=self.device
            )
        self._resolved = True

    # --------------------------------------------------------------- required

    @property
    def command(self) -> torch.Tensor:
        # the manager interface requires a command vector even though the terms read the caches
        return self.frame_buf.unsqueeze(-1).float()

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Reset to a sampled demo frame, writing hand and object state, like gym _reset_default."""
        if not self._resolved:
            self._resolve()
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids)

        if self.cfg.random_state_init:
            seq_idx = (self.seq_len * 0.98 * torch.rand(n, device=self.device)).long()
        else:
            seq_idx = torch.zeros(n, dtype=torch.long, device=self.device)

        self.frame_buf[env_ids] = seq_idx
        self.running_buf[env_ids] = 0
        self._just_reset[env_ids] = True
        self._obs_offset[env_ids] = 1  # observations after a fresh reset read the next frame like gym reset_done

        origins = self._env.scene.env_origins[env_ids]

        for s in ("rh", "lh"):
            demo = self.demo[s]
            robot: Articulation = self.robots[s]

            dof_pos = JM.tensor_clamp(demo["opt_dof_pos"][seq_idx], demo["dof_lower"], demo["dof_upper"])
            dof_vel = JM.tensor_clamp(demo["opt_dof_velocity"][seq_idx], -demo["dof_speed"], demo["dof_speed"])

            wrist_pos = demo["opt_wrist_pos"][seq_idx]
            wrist_quat_wxyz = JM.aa_to_quat(demo["opt_wrist_rot"][seq_idx])
            wrist_vel = demo["opt_wrist_velocity"][seq_idx]
            wrist_ang_vel = demo["opt_wrist_angular_velocity"][seq_idx]

            root = torch.cat([wrist_pos + origins, wrist_quat_wxyz, wrist_vel, wrist_ang_vel], dim=-1)
            robot.write_root_state_to_sim(root, env_ids=env_ids)
            p = self.lab_from_gym[s]
            robot.write_joint_state_to_sim(dof_pos[:, p], dof_vel[:, p], env_ids=env_ids)
            robot.set_joint_position_target(dof_pos[:, p], env_ids=env_ids)
            robot.write_data_to_sim()

            obj: RigidObject = self.objects[s]
            obj_transf = demo["obj_trajectory"][seq_idx]
            obj_pos = obj_transf[:, :3, 3]
            obj_quat_wxyz = JM.rotmat_to_quat(obj_transf[:, :3, :3])
            obj_root = torch.cat(
                [obj_pos + origins, obj_quat_wxyz, demo["obj_velocity"][seq_idx], demo["obj_angular_velocity"][seq_idx]],
                dim=-1,
            )
            obj.write_root_state_to_sim(obj_root, env_ids=env_ids)
            obj.write_data_to_sim()

            self.tips_contact_history[s][env_ids] = True

        # the action term resets its own controller state in reset
        self._ss_cache = {}  # reset environments must observe fresh state
        self._refresh_caches(env_ids)

    def _update_command(self):
        """Advance frames for envs that acted this step, then refresh caches."""
        advance = ~self._just_reset
        self.frame_buf[advance] += 1
        self.running_buf[advance] += 1
        self._obs_offset[advance] = 0  # advanced environments read observations at the current frame, matching gym's pre-incremented counter
        self._just_reset[:] = False

        # the tighten curriculum follows gym's schedule on the policy step counter
        if self.cfg.training:
            last_step = self._env.common_step_counter
            tf, ts = self.cfg.tighten_factor, self.cfg.tighten_steps
            self.scale_factor = (math.e * 2) ** (-1 * last_step / ts) * (1 - tf) + tf
        else:
            self.scale_factor = 1.0

        self._refresh_caches(slice(None))

    # ----------------------------------------------------------------- caches

    def _refresh_caches(self, env_ids):
        """Cache demo targets at the reward frame (frame_buf) and the obs frame (frame_buf + offset)."""
        idx_r = self.frame_buf
        self.obs_idx = torch.clamp(self.frame_buf + self._obs_offset, max=self.seq_len - 1)
        for s in ("rh", "lh"):
            demo = self.demo[s]
            t = {}
            t["wrist_pos"] = demo["wrist_pos"][idx_r]
            t["wrist_quat"] = JM.aa_to_quat(demo["wrist_rot"][idx_r])[:, [1, 2, 3, 0]]
            t["wrist_vel"] = demo["wrist_velocity"][idx_r]
            t["wrist_ang_vel"] = demo["wrist_angular_velocity"][idx_r]
            t["tips_distance"] = demo["tips_distance"][idx_r]
            t["joints_pos"] = demo["mano_joints"][idx_r].reshape(self.num_envs, -1, 3)
            t["joints_vel"] = demo["mano_joints_velocity"][idx_r].reshape(self.num_envs, -1, 3)
            transf = demo["obj_trajectory"][idx_r]
            t["manip_obj_pos"] = transf[:, :3, 3]
            t["manip_obj_quat"] = JM.rotmat_to_quat(transf[:, :3, :3])[:, [1, 2, 3, 0]]
            t["manip_obj_vel"] = demo["obj_velocity"][idx_r]
            t["manip_obj_ang_vel"] = demo["obj_angular_velocity"][idx_r]
            self.target[s] = t

        self.kernel_out = None  # recomputed lazily by the termination term

    # ------------------------------------------------------------ joint kernel

    def side_states(self, s: str) -> dict:
        """Live state in gym conventions, cached per policy step."""
        if not self._resolved:
            self._resolve()
        stamp = self._env.common_step_counter
        if stamp != self._ss_stamp:
            self._ss_cache = {}
            self._ss_stamp = stamp
        if s in self._ss_cache:
            return self._ss_cache[s]
        robot: Articulation = self.robots[s]
        origins = self._env.scene.env_origins
        root = robot.data.root_state_w.clone()
        root[:, :3] -= origins
        base_state = torch.cat([root[:, :3], root[:, 3:7][:, [1, 2, 3, 0]], root[:, 7:13]], dim=-1)

        g = self.gym_from_lab[s]
        q = robot.data.joint_pos[:, g]
        dq = robot.data.joint_vel[:, g]

        body_state = robot.data.body_state_w[:, self.body_order[s], :10].clone()
        body_state[:, :, :3] -= origins.unsqueeze(1)
        body_state[:, :, 3:7] = body_state[:, :, 3:7][:, :, [1, 2, 3, 0]]

        obj: RigidObject = self.objects[s]
        oroot = obj.data.root_state_w.clone()
        oroot[:, :3] -= origins

        out = {
            "q": q,
            "cos_q": torch.cos(q),
            "sin_q": torch.sin(q),
            "dq": dq,
            "base_state": base_state,
            "joints_state": body_state,
            "manip_obj_pos": oroot[:, :3],
            "manip_obj_quat": oroot[:, 3:7][:, [1, 2, 3, 0]],
            "manip_obj_vel": oroot[:, 7:10],
            "manip_obj_ang_vel": oroot[:, 10:13],
        }
        self._ss_cache[s] = out
        return out

    def tip_object_force(self, s: str) -> torch.Tensor:
        """Per-tip force against this hand's own object, the one contact signal for reward, history, and critic obs."""
        return torch.stack([sen.data.force_matrix_w[:, 0, 0, :] for sen in self.tip_sensors[s]], dim=1)

    def run_kernel(self):
        """One joint reward/termination pass per step, both sides, gym-identical."""
        if self.kernel_out is not None:
            return self.kernel_out
        if not self._resolved:
            self._resolve()
        env = self._env
        act_term = env.action_manager.get_term("bih")

        outs = {}
        for s in ("rh", "lh"):
            states = self.side_states(s)
            t = dict(self.target[s])

            tip_force = self.tip_object_force(s)
            hist = self.tips_contact_history[s]
            hist = torch.cat([hist[:, 1:], (torch.norm(tip_force, dim=-1) > 0)[:, None]], dim=1)
            self.tips_contact_history[s] = hist
            t["tip_force"] = tip_force
            t["tip_contact_state"] = hist

            dof_torque = self.robots[s].data.applied_torque[:, self.gym_from_lab[s]]
            t["power"] = torch.abs(dof_torque * states["dq"]).sum(dim=-1)

            wrist_force = act_term.apply_forces[s]
            wrist_torque = act_term.apply_torque[s]
            wrist_power = torch.abs(torch.sum(wrist_force * states["base_state"][:, 7:10], dim=-1))
            wrist_power += torch.abs(torch.sum(wrist_torque * states["base_state"][:, 10:13], dim=-1))
            t["wrist_power"] = wrist_power

            max_length = torch.full(
                (self.num_envs,), float(min(self.seq_len, self.max_episode_length)), device=self.device
            )
            reset_in = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            outs[s] = JM.compute_imitation_reward(
                reset_in,
                self.frame_buf,
                self.running_buf,
                env.action_manager.action,
                states,
                t,
                max_length,
                float(self.scale_factor),
                self.weight_idx,
                float(self.cfg.contact_reward_range[0]),
                float(self.cfg.contact_reward_range[1]),
                float(self.cfg.contact_fail_distance),
            )

        rew = outs["rh"][0] + outs["lh"][0]
        reset = (outs["rh"][1] | outs["lh"][1]).bool()
        success = outs["rh"][2] & outs["lh"][2]
        failure = outs["rh"][3] | outs["lh"][3]
        rdict = {
            **{"rh_" + k: v for k, v in outs["rh"][4].items()},
            **{"lh_" + k: v for k, v in outs["lh"][4].items()},
        }
        error = outs["rh"][5] | outs["lh"][5]
        self.kernel_out = (rew, reset, success, failure, rdict, error)
        return self.kernel_out


@configclass
class BiHTrackingCommandCfg(CommandTermCfg):
    class_type: type = BiHTrackingCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)  # resample only on reset
    demo_path: str = MISSING
    random_state_init: bool = True
    training: bool = True
    tighten_factor: float = 0.7
    tighten_steps: int = 3200
    contact_history_len: int = 3
    # contact constants from the gym kernel
    contact_reward_range: tuple[float, float] = (0.02, 0.03)
    contact_fail_distance: float = -1.0
