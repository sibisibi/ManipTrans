from __future__ import annotations

import os
import random
import yaml
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory


from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rot6d_to_aa,
    rot6d_to_quat,
    quat_to_aa,
)
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat
import pickle


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class DexHandImitatorRHEnv(VecTask):
    side = "right"

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]

        if not hasattr(self, "dexhand"):
            self.dexhand = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand.Kp_rot
            self.Ki_rot = self.dexhand.Ki_rot
            self.Kd_rot = self.dexhand.Kd_rot

            self.Kp_pos = self.dexhand.Kp_pos
            self.Ki_pos = self.dexhand.Ki_pos
            self.Kd_pos = self.dexhand.Kd_pos

        self.cfg["env"]["numActions"] = (
            (1 + 6 + self.dexhand.n_dofs) if use_quat_rot else (6 + self.dexhand.n_dofs)
        ) + (3 if self.use_pid_control else 0)
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        # a dict containing prop obs name to dump and their dimensions
        # used for distillation
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        # Values to be filled in at runtime
        self.states = {}
        self.dexhand_handles = {}  # will be dict mapping names to relevant sim handles
        self.objs_handles = {}  # for obj handlers
        self.objs_assets = {}
        self.num_dofs = None  # Total number of DOFs per env
        self.actions = None  # Current actions to be deployed

        self.dataIndices = self.cfg["env"]["dataIndices"]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]
        self.noisy_reset_init = self.cfg["env"].get("noisyResetInit", True)
        self.wrist_power_weight = float(self.cfg["env"].get("wristPowerWeight", 0.5))
        self.wrist_pos_weight = float(self.cfg["env"].get("wristPosWeight", 0.1))
        self.wrist_ang_vel_weight = float(self.cfg["env"].get("wristAngVelWeight", 0.05))

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        # Adaptive sampling config
        self.sampling_method = self.cfg["env"].get("samplingMethod", "sonic")  # "sonic", "mastery", or "twolevel"
        self.bin_ema_alpha = self.cfg["env"].get("binEmaAlpha", 0.001)
        self.sonic_blend_alpha = self.cfg["env"].get("sonicBlendAlpha", 0.1)
        self.sonic_cap_beta = self.cfg["env"].get("sonicCapBeta", 200.0)
        self.bin_conv_lambda = self.cfg["env"].get("binConvLambda", 0.8)
        self.mastery_floor = self.cfg["env"].get("masteryFloor", 0.05)  # min weight for mastered chunks
        self.twolevel_chunk_alpha = self.cfg["env"].get("twoLevelChunkAlpha", 0.5)  # level-2 adaptive ratio

        # Tensor placeholders
        self._root_state = None  # State of root body        (n_envs, 13)
        self._dof_state = None  # State of all joints       (n_envs, n_dof)
        self._q = None  # Joint positions           (n_envs, n_dof)
        self._qd = None  # Joint velocities          (n_envs, n_dof)
        self._rigid_body_state = None  # State of all rigid bodies             (n_envs, n_bodies, 13)
        self.net_cf = None  # contact force
        self._eef_state = None  # end effector state (at grasping point)
        self._ftip_center_state = None  # center of fingertips
        self._eef_lf_state = None  # end effector state (at left fingertip)
        self._eef_rf_state = None  # end effector state (at left fingertip)
        self._j_eef = None  # Jacobian for end effector
        self._mm = None  # Mass matrix
        self._pos_control = None  # Position actions
        self._effort_control = None  # Torque actions
        self._dexhand_effort_limits = None  # Actuator effort limits for dexhand
        self._dexhand_dof_speed_limits = None  # Actuator speed limits for dexhand
        self._global_dexhand_indices = None  # Unique indices corresponding to all envs in flattened array

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        TARGET_OBS_DIM = (3 + 3 + 3 + 4 + 4 + 3 + 3 + (self.dexhand.n_bodies - 1) * 9) * self.obs_future_length
        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        default_pose = torch.ones(self.dexhand.n_dofs, device=self.device) * np.pi / 36
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)

        # load BPS model
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        if "obj_verts" in self.demo_data:
            obj_verts = self.demo_data["obj_verts"]
            self.obj_bps = self.bps_layer.encode(obj_verts, feature_type=self.bps_feat_type)[self.bps_feat_type]

        # Reset all environments
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Refresh tensors
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        # * >>> import table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4

        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self.dexhand_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        create_kwargs = dict(
            side=self.side,
            device=self.sim_device,
            mujoco2gym_transf=self.mujoco2gym_transf,
            max_seq_len=self.max_episode_length,
            dexhand=self.dexhand,
            embodiment=self.cfg["env"]["dexhand"],
        )

        index_path = self.cfg["env"].get("indexPath", "")
        self._precomputed_pt = None
        if index_path:
            data_dir = self.cfg["env"].get("dataDir", "")
            assert data_dir, "dataDir must be set when using indexPath"

            # Try pre-computed .pt fast path (supports comma-separated list)
            pt_paths = [p.strip() for p in index_path.split(",")]
            if all(p.endswith(".pt") for p in pt_paths):
                pt_list = []
                for pt_rel in pt_paths:
                    pt_path = os.path.join(data_dir, pt_rel)
                    print(f"Loading pre-computed .pt: {pt_path}")
                    pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
                    if "motion_num_frames" not in pt_data:
                        print("  Old .pt format detected, falling back to dataset")
                        pt_list = []
                        break
                    print(f"  {len(pt_data['motion_num_frames']):,} chunks, "
                          f"{pt_data['wrist_pos'].shape[0]:,} frames")
                    pt_list.append(pt_data)

                if len(pt_list) == 1:
                    self._precomputed_pt = pt_list[0]
                elif len(pt_list) > 1:
                    # Concatenate multiple .pt files
                    temporal_keys = ["wrist_pos", "wrist_rot", "wrist_velocity",
                                     "wrist_angular_velocity", "mano_joints", "mano_joints_velocity"]
                    merged = {}
                    for k in temporal_keys:
                        merged[k] = torch.cat([p[k] for p in pt_list], dim=0)
                    merged["motion_num_frames"] = torch.cat([p["motion_num_frames"] for p in pt_list])
                    # Recompute length_starts from merged motion_num_frames
                    nf = merged["motion_num_frames"]
                    ls = torch.zeros(len(nf), dtype=torch.int64)
                    if len(nf) > 1:
                        ls[1:] = torch.cumsum(nf[:-1], dim=0)
                    merged["length_starts"] = ls
                    for k in ["datasets", "filenames", "splits"]:
                        merged[k] = sum([p[k] for p in pt_list], [])
                    self._precomputed_pt = merged
                    total_chunks = len(merged["motion_num_frames"])
                    total_frames = merged["wrist_pos"].shape[0]
                    print(f"Merged {len(pt_list)} .pt files: {total_chunks:,} chunks, {total_frames:,} frames")

            if self._precomputed_pt is None:
                # index_path mode: create dataset directly, use all sequences
                self.demo_dataset = ManipDataFactory.create_data(
                    manipdata_type="handphuma",
                    data_dir=data_dir,
                    index_path=index_path,
                    **create_kwargs,
                )
        else:
            # Legacy dataIndices mode
            data_dir = self.cfg["env"].get("dataDir", "")
            if data_dir:
                create_kwargs["data_dir"] = data_dir
            dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))
            self.demo_dataset_dict = {}
            for dataset_type in dataset_list:
                self.demo_dataset_dict[dataset_type] = ManipDataFactory.create_data(
                    manipdata_type=dataset_type, **create_kwargs
                )

        # load dexhand asset
        dexhand_asset_file = self.dexhand.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_asset_file), asset_options)
        dexhand_dof_stiffness = torch.tensor(
            [500] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [30] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_props_asset)

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        print(f"Num dexhand Bodies: {self.num_dexhand_bodies}")
        print(f"Num dexhand DOFs: {self.num_dexhand_dofs}")

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)

        if self._record:
            bg_pos = gymapi.Vec3(-0.8, 0, 0.75)
            bg_pose = gymapi.Transform()
            bg_pose.p = gymapi.Vec3(bg_pos.x, bg_pos.y, bg_pos.z)

        # compute aggregate size
        num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        num_dexhand_shapes = self.gym.get_asset_rigid_shape_count(dexhand_asset)

        self.dexhands = []
        self.envs = []

        if self._precomputed_pt is not None:
            # Pre-computed .pt fast path: bypass dataset + __getitem__ entirely
            pt = self._precomputed_pt
            num_chunks = len(pt["motion_num_frames"])
            assert num_chunks > 0, "Pre-computed .pt is empty"
            assert num_chunks == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

            self.motion_num_frames = pt["motion_num_frames"].to(self.device)
            self.length_starts = pt["length_starts"].to(self.device)

            self.demo_data = {
                "wrist_pos": pt["wrist_pos"].to(self.device),
                "wrist_rot": pt["wrist_rot"].to(self.device),
                "wrist_velocity": pt["wrist_velocity"].to(self.device),
                "wrist_angular_velocity": pt["wrist_angular_velocity"].to(self.device),
                "mano_joints": pt["mano_joints"].to(self.device),
                "mano_joints_velocity": pt["mano_joints_velocity"].to(self.device),
            }

            # Lightweight proxy for demo_dataset.sequences (needed for chunk-to-seq mapping)
            class _SeqProxy:
                def __init__(self, datasets, filenames, splits):
                    self.sequences = [
                        {"dataset": d, "filename": f, "split": s}
                        for d, f, s in zip(datasets, filenames, splits)
                    ]
                def __len__(self):
                    return len(self.sequences)
            self.demo_dataset = _SeqProxy(pt["datasets"], pt["filenames"], pt["splits"])

            del self._precomputed_pt

            self.num_chunks = num_chunks
            self.sampling_weights = torch.ones(num_chunks, device=self.device)
            self.motion_ids = torch.multinomial(
                self.sampling_weights, self.num_envs, replacement=True
            )
            print(f"Pre-computed .pt loaded: {num_chunks:,} chunks, "
                  f"{self.demo_data['wrist_pos'].shape[0]:,} frames (instant)")
        elif hasattr(self, 'demo_dataset'):
            # index_path mode: load ALL chunks once, enable dynamic weighted sampling
            num_chunks = len(self.demo_dataset)
            assert num_chunks > 0, "Dataset is empty"
            assert num_chunks == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

            all_chunks = [self.demo_dataset[i] for i in tqdm(range(num_chunks))]
            self.demo_data = self.pack_data_flat(all_chunks)  # flat storage: [total_frames, ...]

            # Dynamic weighted sampling state
            self.num_chunks = num_chunks
            self.sampling_weights = torch.ones(num_chunks, device=self.device)
            self.motion_ids = torch.multinomial(
                self.sampling_weights, self.num_envs, replacement=True
            )
        else:
            # Legacy dataIndices mode
            assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"
            dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

            def segment_data(k):
                todo_list = self.dataIndices
                idx = todo_list[k % len(todo_list)]
                return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

            self.demo_data = [segment_data(i) for i in tqdm(range(self.num_envs))]
            self.demo_data = self.pack_data_flat(self.demo_data)

            # Legacy mode: identity mapping (motion_ids[i] == i), no dynamic sampling
            self.num_chunks = self.num_envs
            self.sampling_weights = None
            self.motion_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        # Per-chunk metrics (Track 1) and chunk-to-sequence mapping (Track 2)
        if self.sampling_weights is not None:
            self.chunk_success_count = torch.zeros(self.num_chunks, device=self.device)
            self.chunk_fail_count = torch.zeros(self.num_chunks, device=self.device)
            self.chunk_eval_count = torch.zeros(self.num_chunks, device=self.device)
            self.chunk_completion_sum = torch.zeros(self.num_chunks, device=self.device)
            self.chunk_reward_sum = torch.zeros(self.num_chunks, device=self.device)
            # Global counters (never reset, track cumulative stats across all of training)
            self.chunk_eval_count_global = torch.zeros(self.num_chunks, device=self.device)
            self.chunk_success_count_global = torch.zeros(self.num_chunks, device=self.device)
            self.chunk_completion_sum_global = torch.zeros(self.num_chunks, device=self.device)

        # Per-joint-group error histograms for adaptive threshold statistics
        self._err_joint_groups = [
            "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
            "level_1", "level_2",
            "wrist_pos", "wrist_rot", "wrist_vel", "wrist_ang_vel", "joints_vel",
        ]
        self._err_hist_nbins = 200
        # Different histogram ranges for different error types (all in native units)
        self._err_hist_range = {
            "thumb_tip": 0.2, "index_tip": 0.2, "middle_tip": 0.2,  # meters
            "ring_tip": 0.2, "pinky_tip": 0.2,
            "level_1": 0.2, "level_2": 0.2,
            "wrist_pos": 0.2,           # meters
            "wrist_rot": 3.15,          # radians (~pi)
            "wrist_vel": 10.0,          # m/s
            "wrist_ang_vel": 20.0,      # rad/s
            "joints_vel": 10.0,         # rad/s (mean across joints)
        }
        self._err_histograms = {
            g: torch.zeros(self._err_hist_nbins, device=self.device)
            for g in self._err_joint_groups
        }
        self._err_hist_count = 0

        # Chunk-level max error tracking: running max per env, recorded on episode end
        n_groups = len(self._err_joint_groups)
        self._err_chunk_max = torch.zeros(self.num_envs, n_groups, device=self.device)
        self._err_chunk_max_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._err_chunk_max_histograms = {
            g: torch.zeros(self._err_hist_nbins, device=self.device)
            for g in self._err_joint_groups
        }
        self._err_chunk_max_count = 0

        # Adaptive termination thresholds
        self._adaptive_alpha = float(self.cfg["env"].get("adaptiveAlpha", 0.0))
        # The 7 termination groups in JIT column order (must match failed_execute indexing)
        self._termination_groups = [
            "thumb_tip", "index_tip", "middle_tip", "pinky_tip", "ring_tip",
            "level_1", "level_2",
        ]
        self._base_thresholds = torch.tensor(
            [0.04, 0.045, 0.05, 0.06, 0.06, 0.07, 0.08], device=self.device
        )
        self._default_thresholds = self._base_thresholds / 0.7  # loosest initial thresholds

        if self.sampling_weights is not None:
            # Build chunk-to-sequence mapping from CSV data
            seq_key_to_id = {}
            chunk_to_seq_list = []
            for seq_info in self.demo_dataset.sequences:
                key = (seq_info["dataset"], seq_info["filename"])
                if key not in seq_key_to_id:
                    seq_key_to_id[key] = len(seq_key_to_id)
                chunk_to_seq_list.append(seq_key_to_id[key])
            self.chunk_to_seq = torch.tensor(chunk_to_seq_list, dtype=torch.long, device=self.device)
            self.num_sequences = len(seq_key_to_id)
            self.chunk_duration_frames = self.motion_num_frames.clone()

            # Parse split column from CSV for evaluation
            train_idx = []
            test_idx = []
            for i, seq_info in enumerate(self.demo_dataset.sequences):
                split = seq_info.get("split", "train")
                if split == "test":
                    test_idx.append(i)
                else:
                    train_idx.append(i)
            self._train_chunk_indices = torch.tensor(train_idx, dtype=torch.long, device=self.device)
            self._test_chunk_indices = torch.tensor(test_idx, dtype=torch.long, device=self.device)

            # Build chunk-to-dataset mapping for per-dataset metrics
            dataset_name_to_id = {}
            chunk_to_dataset_list = []
            for seq_info in self.demo_dataset.sequences:
                ds = seq_info["dataset"]
                if ds not in dataset_name_to_id:
                    dataset_name_to_id[ds] = len(dataset_name_to_id)
                chunk_to_dataset_list.append(dataset_name_to_id[ds])
            self.chunk_to_dataset = torch.tensor(chunk_to_dataset_list, dtype=torch.long, device=self.device)
            self.dataset_names = list(dataset_name_to_id.keys())
            self.num_datasets = len(self.dataset_names)

            # Per-dataset sampling weight multiplier
            ds_weights_path = self.cfg["env"].get("datasetWeightsPath", "")
            if ds_weights_path and os.path.isfile(ds_weights_path):
                with open(ds_weights_path, "r") as f:
                    ds_weights_cfg = yaml.safe_load(f)
                multiplier = torch.ones(self.num_chunks, device=self.device)
                for ci in range(self.num_chunks):
                    ds_name = self.dataset_names[chunk_to_dataset_list[ci]]
                    # Try exact match first, then prefix match (longest prefix wins)
                    w = ds_weights_cfg.get(ds_name, None)
                    if w is None:
                        best_prefix = ""
                        for key in ds_weights_cfg:
                            if ds_name.startswith(key) and len(key) > len(best_prefix):
                                best_prefix = key
                        w = ds_weights_cfg[best_prefix] if best_prefix else 1.0
                    multiplier[ci] = float(w)
                # Normalize so sum(multiplier) = num_chunks
                multiplier *= self.num_chunks / multiplier.sum()
                self._dataset_weight_multiplier = multiplier
                print("[DexHandImitator] Loaded dataset weights from %s" % ds_weights_path)
                for ds_name in self.dataset_names:
                    ds_mask = self.chunk_to_dataset == dataset_name_to_id[ds_name]
                    mean_w = multiplier[ds_mask].mean().item() if ds_mask.any() else 0.0
                    print("  %s: mean_multiplier=%.3f" % (ds_name, mean_w))
            else:
                self._dataset_weight_multiplier = torch.ones(self.num_chunks, device=self.device)

            # Per-dataset frame-level error histograms
            self._err_histograms_by_dataset = {
                g: torch.zeros(self.num_datasets, self._err_hist_nbins, device=self.device)
                for g in self._err_joint_groups
            }
            self._err_hist_count_by_dataset = torch.zeros(self.num_datasets, dtype=torch.long, device=self.device)

            # Per-dataset chunk-max error histograms
            self._err_chunk_max_histograms_by_dataset = {
                g: torch.zeros(self.num_datasets, self._err_hist_nbins, device=self.device)
                for g in self._err_joint_groups
            }
            self._err_chunk_max_count_by_dataset = torch.zeros(self.num_datasets, dtype=torch.long, device=self.device)

            # Per-dataset adaptive thresholds (only active when adaptive_alpha > 0)
            if self._adaptive_alpha > 0:
                self._adaptive_thresholds = self._default_thresholds.unsqueeze(0).expand(
                    self.num_datasets, -1
                ).clone()  # shape (num_datasets, 7)

            # === Two-level sampling state ===

            # Bin-level state (BeyondMimic + SONIC)
            # Each chunk IS a 1-second bin, so bin_failed_count is per-chunk
            self.bin_failed_count = torch.zeros(self.num_chunks, device=self.device)
            # EMA completion rate per chunk (for mastery-based sampling)
            self.bin_completion_ema = torch.zeros(self.num_chunks, device=self.device)
            # Precompute: number of chunks per sequence (for uniform floor)
            self.chunks_per_seq = torch.zeros(self.num_sequences, device=self.device)
            self.chunks_per_seq.scatter_add_(
                0, self.chunk_to_seq,
                torch.ones(self.num_chunks, device=self.device)
            )

            # Cross-chunk continuation: link each chunk to its successor in the trajectory.
            # Chunks within the same sequence are consecutive in the flat array
            # (chunk_index.py generates them in chunk_id order per trajectory).
            self.next_chunk = torch.full((self.num_chunks,), -1, dtype=torch.long, device=self.device)
            if self.num_chunks > 1:
                same_seq = self.chunk_to_seq[:-1] == self.chunk_to_seq[1:]
                link_indices = torch.arange(self.num_chunks - 1, device=self.device)[same_seq]
                self.next_chunk[link_indices] = link_indices + 1


            # Initial combined weights (starts as uniform)
            self._recompute_sampling_weights()
        else:
            # Legacy mode: no split info, all chunks are "train"
            self._train_chunk_indices = torch.arange(self.num_chunks, dtype=torch.long, device=self.device)
            self._test_chunk_indices = torch.tensor([], dtype=torch.long, device=self.device)

        # Create environments
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            max_agg_bodies = (
                num_dexhand_bodies + 1 + (5 + (0 + self.dexhand.n_bodies if not self.headless else 0))
            )  # 1 for table
            max_agg_shapes = (
                num_dexhand_shapes
                + 1
                + (5 + (0 + self.dexhand.n_bodies if not self.headless else 0))
                + (1 if self._record else 0)
            )
            # Create actors and define aggregate group appropriately depending on setting
            # NOTE: dexhand should ALWAYS be loaded first in sim!
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # camera handler for view rendering
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

            # Create dexhand
            dexhand_actor = self.gym.create_actor(
                env_ptr,
                dexhand_asset,
                self.dexhand_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),  # ! some hand need to allow self-collision
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_actor, dexhand_dof_props)

            # Create table and obstacles
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i + self.num_envs, 0b11
            )  # ignore collision
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1  # ? only one table shape in each env
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            # set table's color to be dark gray
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # Store the created env pointers
            self.envs.append(env_ptr)
            self.dexhands.append(dexhand_actor)

        # Setup data
        self.init_data()

    def init_data(self):
        # Setup sim handles
        env_ptr = self.envs[0]
        dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand.body_names
        }
        # Get total DOFs
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # Setup tensor buffers
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        # ? >>> for visualization
        if not self.headless:

            self.mano_joint_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
                for i in range(self.dexhand.n_bodies)
            ]
        # ? <<<

        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self.dexhand_root_state = self._root_state[:, dexhand_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        # Initialize actions
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # Initialize indices
        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

    def pack_data(self, data):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        for k in data[0].keys():
            if "alt" in k:
                continue
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    mano_joints.append(
                        torch.concat(
                            [
                                d[k][self.dexhand.to_hand(j_name)[0]]
                                for j_name in self.dexhand.body_names
                                if self.dexhand.to_hand(j_name)[0] != "wrist"
                            ],
                            dim=-1,
                        )
                    )
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]

        def to_cuda(x):
            if type(x) == torch.Tensor:
                return x.to(self.device)
            elif type(x) == list:
                return [to_cuda(xx) for xx in x]
            elif type(x) == dict:
                return {k: to_cuda(v) for k, v in x.items()}
            else:
                return x

        packed_data = to_cuda(packed_data)

        return packed_data

    def pack_data_flat(self, data):
        """Pack data as flat concatenated tensors (ProtoMotions-style).

        Instead of padding all chunks to max_len and stacking into [num_chunks, max_len, ...],
        this concatenates all chunks into flat [total_frames, ...] tensors and uses
        length_starts offsets for indexing.
        """
        flat_data = {}

        # Compute per-chunk frame counts
        frame_counts = [len(d["obj_trajectory"]) for d in data]
        motion_num_frames = torch.tensor(frame_counts, dtype=torch.long, device=self.device)

        # Compute length_starts (exclusive cumsum)
        lengths_shifted = motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        self.motion_num_frames = motion_num_frames
        self.length_starts = length_starts

        # Temporal fields: torch.cat across chunks → [total_frames, ...]
        temporal_keys = [
            "wrist_pos", "wrist_rot", "wrist_velocity", "wrist_angular_velocity",
            "obj_trajectory", "obj_velocity", "obj_angular_velocity", "tips_distance",
        ]

        for k in temporal_keys:
            if k in data[0]:
                flat_data[k] = torch.cat([d[k] for d in data], dim=0).to(self.device)

        # mano_joints and mano_joints_velocity: apply joint name concatenation, then cat
        for k in ["mano_joints", "mano_joints_velocity"]:
            if k in data[0]:
                chunks = []
                for d in data:
                    chunks.append(
                        torch.concat(
                            [
                                d[k][self.dexhand.to_hand(j_name)[0]]
                                for j_name in self.dexhand.body_names
                                if self.dexhand.to_hand(j_name)[0] != "wrist"
                            ],
                            dim=-1,
                        )
                    )
                flat_data[k] = torch.cat(chunks, dim=0).to(self.device)

        # obj_verts: non-temporal, stack as [num_chunks, 1000, 3]
        if "obj_verts" in data[0]:
            flat_data["obj_verts"] = torch.stack([d["obj_verts"] for d in data]).squeeze().to(self.device)

        return flat_data

    def allocate_buffers(self):
        # will also allocate extra buffers for data dumping, used for distillation
        super().allocate_buffers()

        # basic prop fields
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _update_states(self):
        self.states.update(
            {
                "q": self._q[:, :],
                "cos_q": torch.cos(self._q[:, :]),
                "sin_q": torch.sin(self._q[:, :]),
                "dq": self._qd[:, :],
                "base_state": self._base_state[:, :],
            }
        )

        self.states["joints_state"] = torch.stack(
            [self._rigid_body_state[:, self.dexhand_handles[k], :][:, :10] for k in self.dexhand.body_names],
            dim=1,
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # Refresh states
        self._update_states()

    def compute_reward(self, actions):
        target_state = {}
        max_length = torch.clip(self.motion_num_frames[self.motion_ids], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_idx = torch.clamp(cur_idx, max=self.motion_num_frames[self.motion_ids] - 1)
        flat_idx = cur_idx + self.length_starts[self.motion_ids]
        cur_wrist_pos = self.demo_data["wrist_pos"][flat_idx]
        target_state["wrist_pos"] = cur_wrist_pos
        cur_wrist_rot = self.demo_data["wrist_rot"][flat_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = self.demo_data["wrist_velocity"][flat_idx]
        target_state["wrist_ang_vel"] = self.demo_data["wrist_angular_velocity"][flat_idx]

        cur_joints_pos = self.demo_data["mano_joints"][flat_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = self.demo_data["mano_joints_velocity"][
            flat_idx
        ].reshape(self.num_envs, -1, 3)

        power = torch.abs(torch.multiply(self.dof_force, self.states["dq"])).sum(dim=-1)
        target_state["power"] = power

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
                * self.states["base_state"][:, 7:10],
                dim=-1,
            )
        )  # ? linear force * linear velocity
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
                * self.states["base_state"][:, 10:],
                dim=-1,
            )
        )  # ? torque * angular velocity
        target_state["wrist_power"] = wrist_power

        # Build per-env termination thresholds tensor (num_envs, 7)
        if self._adaptive_alpha > 0 and self.training and hasattr(self, '_adaptive_thresholds'):
            # Adaptive path: per-dataset per-group thresholds
            env_ds = self.chunk_to_dataset[self.motion_ids]  # (num_envs,)
            term_thresholds = self._adaptive_thresholds[env_ds]  # (num_envs, 7)
        else:
            # Legacy path: compute scale_factor exactly as before
            if self.training:
                last_step = self.gym.get_frame_count(self.sim)
                if self.tighten_method == "None":
                    scale_factor = 1.0
                elif self.tighten_method == "const":
                    scale_factor = self.tighten_factor
                elif self.tighten_method == "linear_decay":
                    scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
                elif self.tighten_method == "exp_decay":
                    scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                        1 - self.tighten_factor
                    ) + self.tighten_factor
                elif self.tighten_method == "cos":
                    scale_factor = (self.tighten_factor) + np.abs(
                        -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                    ) * (2 ** (-1 * last_step / self.tighten_steps))
                else:
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
            # Build (num_envs, 7) from scalar scale_factor — numerically identical to old hardcoded values
            term_thresholds = (self._base_thresholds / 0.7 * scale_factor).unsqueeze(0).expand(self.num_envs, -1)

        assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        self.rew_buf[:], self.reset_buf[:], self.success_buf[:], self.failure_buf[:], self.reward_dict = (
            compute_imitation_reward(
                self.reset_buf,
                self.progress_buf,
                self.running_progress_buf,
                self.actions,
                self.states,
                target_state,
                max_length,
                term_thresholds,
                self.dexhand.weight_idx,
                self.wrist_power_weight,
                self.wrist_pos_weight,
                self.wrist_ang_vel_weight,
            )
        )
        self.total_rew_buf += self.rew_buf

        # Accumulate per-joint-group error histograms (exclude grace period)
        active = self.running_progress_buf >= 20
        if active.any():
            active_idx = active.nonzero(as_tuple=True)[0]
            for gi, g in enumerate(self._err_joint_groups):
                errs = self.reward_dict["err_" + g][active]
                hmax = self._err_hist_range[g]
                # Global frame-level histogram
                hist = torch.histc(errs, bins=self._err_hist_nbins, min=0, max=hmax)
                self._err_histograms[g] += hist
                # Update per-env chunk running max
                self._err_chunk_max[active_idx, gi] = torch.max(
                    self._err_chunk_max[active_idx, gi], errs
                )
            self._err_chunk_max_active[active_idx] = True
            self._err_hist_count += int(active.sum().item())

            # Per-dataset frame-level histograms
            if hasattr(self, '_err_histograms_by_dataset'):
                env_ds = self.chunk_to_dataset[self.motion_ids[active_idx]]
                for ds_id in range(self.num_datasets):
                    ds_mask = env_ds == ds_id
                    if not ds_mask.any():
                        continue
                    n_ds = int(ds_mask.sum().item())
                    self._err_hist_count_by_dataset[ds_id] += n_ds
                    for gi, g in enumerate(self._err_joint_groups):
                        errs_ds = self.reward_dict["err_" + g][active][ds_mask]
                        hmax = self._err_hist_range[g]
                        hist_ds = torch.histc(errs_ds, bins=self._err_hist_nbins, min=0, max=hmax)
                        self._err_histograms_by_dataset[g][ds_id] += hist_ds

    def compute_eval_metrics(self):
        """Compute raw tracking errors for evaluation. NOT for reward — raw L2 errors."""
        cur_idx = self.progress_buf
        # Clamp to valid range using flat-concat metadata
        cur_idx = torch.clamp(cur_idx, max=self.motion_num_frames[self.motion_ids] - 1)
        flat_idx = cur_idx + self.length_starts[self.motion_ids]

        # Reference positions (same fields as compute_reward, lines 680-691)
        ref_wrist_pos = self.demo_data["wrist_pos"][flat_idx]
        ref_wrist_rot = self.demo_data["wrist_rot"][flat_idx]
        ref_joints_pos = self.demo_data["mano_joints"][flat_idx].reshape(self.num_envs, -1, 3)

        # Current simulated state (same as compute_imitation_reward, lines 1317-1318, 1333)
        sim_wrist_pos = self.states["base_state"][:, :3]
        sim_wrist_quat = self.states["base_state"][:, 3:7]
        sim_joints_pos = self.states["joints_state"][:, 1:, :3]  # skip base joint

        # Per-joint L2 errors (meters)
        diff_joints_pos_dist = torch.norm(ref_joints_pos - sim_joints_pos, dim=-1)  # [num_envs, num_joints]

        # Per-joint-group errors (same index mapping as compute_imitation_reward, lines 1339-1345)
        thumb_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["thumb_tip"]]].mean(dim=-1)
        index_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["index_tip"]]].mean(dim=-1)
        middle_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["middle_tip"]]].mean(dim=-1)
        ring_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["ring_tip"]]].mean(dim=-1)
        pinky_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["pinky_tip"]]].mean(dim=-1)
        level_1_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["level_1_joints"]]].mean(dim=-1)
        level_2_err = diff_joints_pos_dist[:, [k - 1 for k in self.dexhand.weight_idx["level_2_joints"]]].mean(dim=-1)

        # Wrist errors
        E_wrist_pos = torch.norm(ref_wrist_pos - sim_wrist_pos, dim=-1)  # [num_envs], meters
        ref_wrist_quat = aa_to_quat(ref_wrist_rot)[:, [1, 2, 3, 0]]  # match ManipTrans convention (line 683)
        diff_rot = quat_mul(ref_wrist_quat, quat_conjugate(sim_wrist_quat))
        E_wrist_rot = quat_to_angle_axis(diff_rot)[0]  # radians

        # Aggregate: mean joint error, mean fingertip error
        E_j = diff_joints_pos_dist.mean(dim=-1)  # [num_envs]
        E_ft = torch.stack([thumb_err, index_err, middle_err, ring_err, pinky_err], dim=-1).mean(dim=-1)  # [num_envs]

        # Fixed eval thresholds: 60mm fingertips, 80mm joints
        failed = (
            (thumb_err > 0.06)
            | (index_err > 0.06)
            | (middle_err > 0.06)
            | (ring_err > 0.06)
            | (pinky_err > 0.06)
            | (level_1_err > 0.08)
            | (level_2_err > 0.08)
        )

        return {
            "E_j": E_j,                    # [num_envs], meters
            "E_ft": E_ft,                  # [num_envs], meters
            "E_wrist_pos": E_wrist_pos,    # [num_envs], meters
            "E_wrist_rot": E_wrist_rot,    # [num_envs], radians
            "thumb_err": thumb_err,
            "index_err": index_err,
            "middle_err": middle_err,
            "ring_err": ring_err,
            "pinky_err": pinky_err,
            "level_1_err": level_1_err,
            "level_2_err": level_2_err,
            "failed": failed,              # [num_envs], bool
        }

    def compute_observations(self):
        self._refresh()
        # obs_keys: q, cos_q, sin_q, base_state
        obs_values = []
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(self.states[ob][:, :3]), self.states[ob][:, 3:]], dim=-1)
                )  # ! ignore base position
            else:
                obs_values.append(self.states[ob])
        self.obs_dict["proprioception"][:] = torch.cat(obs_values, dim=-1)
        # privileged_obs_keys: dq, manip_obj_pos, manip_obj_quat, manip_obj_vel, manip_obj_ang_vel
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_values.append(self.states[ob] - self.states["base_state"][:, :3])
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(self.states["manip_obj_quat"][:, [1, 2, 3, 0]])
                        @ self.manip_obj_com.unsqueeze(-1)
                    ).squeeze(-1) + self.states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - self.states["base_state"][:, :3])
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((self.manip_obj_mass * -1 * prop.gravity.z).unsqueeze(-1))
                elif ob == "tip_force":
                    tip_force = torch.stack(
                        [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                else:
                    pri_obs_values.append(self.states[ob])
            self.obs_dict["privileged"][:] = torch.cat(pri_obs_values, dim=-1)

        next_target_state = {}

        cur_idx = self.progress_buf + 1
        seq_len = self.motion_num_frames[self.motion_ids]
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(seq_len), seq_len - 1)

        cur_idx = torch.stack(
            [torch.clamp(cur_idx + t, max=seq_len - 1) for t in range(self.obs_future_length)], dim=-1
        )  # [B, K], K = obs_future_length
        nE = self.num_envs
        nF = self.obs_future_length

        def indicing(flat_data, idx):
            # flat_data: [total_frames, ...], idx: [num_envs, K]
            offsets = self.length_starts[self.motion_ids].unsqueeze(1)  # [num_envs, 1]
            flat_idx = idx + offsets  # [num_envs, K]
            return flat_data[flat_idx]  # [num_envs, K, ...]

        target_wrist_pos = indicing(self.demo_data["wrist_pos"], cur_idx)  # [B, K, 3]
        cur_wrist_pos = self.states["base_state"][:, :3]  # [B, 3]
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = self.states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(self.demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot = self.states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = self.states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = self.states["joints_state"][:, 1:, :3]  # skip the base joint
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = self.states["joints_state"][:, 1:, 7:10]  # skip the base joint
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        self.obs_dict["target"][:] = torch.cat(
            [
                next_target_state[ob]
                for ob in [
                    "delta_wrist_pos",
                    "wrist_vel",
                    "delta_wrist_vel",
                    "wrist_quat",
                    "delta_wrist_quat",
                    "wrist_ang_vel",
                    "delta_wrist_ang_vel",
                    "delta_joints_pos",
                    "joints_vel",
                    "delta_joints_vel",
                ]
            ],
            dim=-1,
        )

        # update fields to dump
        # prop fields
        if not self.training:
            for prop_name in self._prop_dump_info.keys():
                self.dump_fileds[prop_name][:] = self.states[prop_name][:]
        return self.obs_dict

    def _reset_default(self, env_ids):
        # Sample new chunks for resetting envs (dynamic weighted sampling)
        if self.sampling_weights is not None:
            new_ids = torch.multinomial(
                self.sampling_weights, len(env_ids), replacement=True
            )
            self.motion_ids[env_ids] = new_ids

        chunk_seq_len = self.motion_num_frames[self.motion_ids[env_ids]]
        if self.random_state_init:
            seq_idx = torch.floor(
                chunk_seq_len * 0.99 * torch.rand_like(chunk_seq_len.float())
            ).long()
        else:
            seq_idx = torch.zeros_like(chunk_seq_len.long())

        if self.noisy_reset_init:
            noise_dof_pos = (
                torch.randn_like(self.dexhand_default_dof_pos[None].repeat(len(env_ids), 1))
                * ((self.dexhand_dof_upper_limits - self.dexhand_dof_lower_limits) / 8)[None]
            )
            dof_pos = torch.clamp(
                self.dexhand_default_dof_pos[None].repeat(len(env_ids), 1) + noise_dof_pos,
                self.dexhand_dof_lower_limits.unsqueeze(0),
                self.dexhand_dof_upper_limits.unsqueeze(0),
            )
            dof_vel = torch.randn([len(env_ids), self.dexhand.n_dofs], device=self.device) * 0.1
            dof_vel = torch.clamp(
                dof_vel,
                -1 * self._dexhand_dof_speed_limits.unsqueeze(0),
                self._dexhand_dof_speed_limits.unsqueeze(0),
            )
        else:
            dof_pos = self.dexhand_default_dof_pos[None].repeat(len(env_ids), 1)
            dof_vel = torch.zeros([len(env_ids), self.dexhand.n_dofs], device=self.device)

        reset_flat_idx = seq_idx + self.length_starts[self.motion_ids[env_ids]]
        opt_wrist_pos = self.demo_data["wrist_pos"][reset_flat_idx]
        opt_wrist_rot_aa = self.demo_data["wrist_rot"][reset_flat_idx]
        opt_wrist_vel = self.demo_data["wrist_velocity"][reset_flat_idx]
        opt_wrist_ang_vel = self.demo_data["wrist_angular_velocity"][reset_flat_idx]

        if self.noisy_reset_init:
            opt_wrist_pos = opt_wrist_pos + torch.randn_like(opt_wrist_pos) * 0.01
            opt_wrist_rot = aa_to_rotmat(opt_wrist_rot_aa)
            noise_rot = torch.rand(opt_wrist_rot.shape[0], 3, device=self.device)
            noise_rot = aa_to_rotmat(
                noise_rot
                / torch.norm(noise_rot, dim=-1, keepdim=True)
                * torch.randn(opt_wrist_rot.shape[0], 1, device=self.device)
                * (np.pi / 18)
            )
            opt_wrist_rot = noise_rot @ opt_wrist_rot
            opt_wrist_rot = rotmat_to_quat(opt_wrist_rot)
            opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]
            opt_wrist_vel = opt_wrist_vel + torch.randn_like(opt_wrist_vel) * 0.01
            opt_wrist_ang_vel = opt_wrist_ang_vel + torch.randn_like(opt_wrist_ang_vel) * 0.01
        else:
            opt_wrist_rot = rotmat_to_quat(aa_to_rotmat(opt_wrist_rot_aa))
            opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        self._base_state[env_ids, :] = opt_hand_pose_vel

        self._q[env_ids, :] = dof_pos
        self._qd[env_ids, :] = dof_vel
        self._pos_control[env_ids, :] = dof_pos

        # Deploy updates for dexhand
        dexhand_multi_env_ids_int32 = self._global_dexhand_indices[env_ids].flatten()

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0
        if self.use_pid_control:
            self.prev_pos_error[env_ids] = 0
            self.prev_rot_error[env_ids] = 0
            self.pos_error_integral[env_ids] = 0
            self.rot_error_integral[env_ids] = 0
        # Reset chunk-max error tracking for new episodes
        self._err_chunk_max[env_ids] = 0
        self._err_chunk_max_active[env_ids] = False

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        self._reset_default(env_ids)

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

            # asymmetric actor-critic
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        return obs, rew, done, info

    def pre_physics_step(self, actions):

        # ? >>> for visualization
        if not self.headless:

            cur_idx = self.progress_buf
            cur_idx = torch.clamp(cur_idx, max=self.motion_num_frames[self.motion_ids] - 1)

            viz_flat_idx = cur_idx + self.length_starts[self.motion_ids]
            cur_wrist_pos = self.demo_data["wrist_pos"][viz_flat_idx]

            cur_mano_joint_pos = self.demo_data["mano_joints"][viz_flat_idx].reshape(
                self.num_envs, -1, 3
            )
            cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)

            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = cur_mano_joint_pos[:, k]

            self.gym.clear_lines(self.viewer)
            for env_id, env_ptr in enumerate(self.envs):
                for k in self.dexhand.body_names:
                    self.set_force_vis(
                        env_ptr, k, torch.norm(self.net_cf[env_id, self.dexhand_handles[k]], dim=-1) != 0
                    )

                def add_lines(viewer, env_ptr, hand_joints):
                    assert hand_joints.shape[0] == self.dexhand.n_bodies and hand_joints.shape[1] == 3
                    hand_joints = hand_joints.cpu().numpy()
                    red = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
                    green = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                    thumb_idx = set(range(21, 28))
                    index_idx = set(range(1, 6))
                    for b in self.dexhand.bone_links:
                        line = np.array([[hand_joints[b[0]], hand_joints[b[1]]]])
                        color = red if (b[0] in thumb_idx or b[1] in thumb_idx or b[0] in index_idx or b[1] in index_idx) else green
                        self.gym.add_lines(viewer, env_ptr, 1, line, color)

                add_lines(self.viewer, env_ptr, cur_mano_joint_pos[env_id].cpu())

        # ? <<< for visualization
        curr_act_moving_average = self.act_moving_average

        root_control_dim = 9 if self.use_pid_control else 6

        dof_pos = actions[:, root_control_dim : root_control_dim + self.num_dexhand_dofs]
        dof_pos = torch.clamp(dof_pos, -1, 1)
        self.curr_targets = torch_jit_utils.scale(
            dof_pos,  # ! actions must in [-1, 1]
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )
        self.curr_targets = (
            curr_act_moving_average * self.curr_targets + (1.0 - curr_act_moving_average) * self.prev_targets
        )
        self.curr_targets = torch_jit_utils.tensor_clamp(
            self.curr_targets,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )

        if self.use_pid_control:
            position_error = actions[:, :3]
            self.pos_error_integral += position_error * self.dt
            self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
            pos_derivative = (position_error - self.prev_pos_error) / self.dt
            force = self.Kp_pos * position_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * pos_derivative
            self.prev_pos_error = position_error
            self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )

            rotation_error = actions[:, 3:9]
            rotation_error = rot6d_to_aa(rotation_error)
            self.rot_error_integral += rotation_error * self.dt
            self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
            rot_derivative = (rotation_error - self.prev_rot_error) / self.dt
            torque = self.Kp_rot * rotation_error + self.Ki_rot * self.rot_error_integral + self.Kd_rot * rot_derivative
            self.prev_rot_error = rotation_error
            self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )

        else:
            self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * (actions[:, 0:3] * self.dt * self.translation_scale * 500)
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * (actions[:, 3:6] * self.dt * self.orientation_scale * 200)
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self.prev_targets[:] = self.curr_targets[:]
        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))

    def post_physics_step(self):

        self.compute_observations()
        self.compute_reward(self.actions)

        # Update chunk metrics for terminated episodes (before motion_ids change)
        if self.sampling_weights is not None:
            done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.update_chunk_metrics(done_env_ids)
            self.update_bin_failures(done_env_ids)
            self._record_chunk_max_errors(done_env_ids)

            # Cross-chunk continuation: advance non-last chunks instead of resetting.
            # Must be AFTER metrics (so chunk completion is recorded) and
            # BEFORE progress_buf increment (so -1 becomes 0).
            self._handle_chunk_continuation()

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1

    def update_chunk_metrics(self, done_env_ids):
        """Update per-chunk metrics for terminated episodes."""
        if len(done_env_ids) == 0:
            return
        chunk_ids = self.motion_ids[done_env_ids]

        # Scatter-add success/fail/count
        ones = torch.ones_like(chunk_ids, dtype=torch.float)
        successes = self.success_buf[done_env_ids].float()
        self.chunk_eval_count.scatter_add_(0, chunk_ids, ones)
        self.chunk_success_count.scatter_add_(0, chunk_ids, successes)
        self.chunk_fail_count.scatter_add_(0, chunk_ids, self.failure_buf[done_env_ids].float())

        # Completion fraction
        seq_lens = self.motion_num_frames[chunk_ids].float()
        completion = self.progress_buf[done_env_ids].float() / seq_lens.clamp(min=1)
        self.chunk_completion_sum.scatter_add_(0, chunk_ids, completion)

        # Global counters (never reset)
        if hasattr(self, 'chunk_eval_count_global'):
            self.chunk_eval_count_global.scatter_add_(0, chunk_ids, ones)
            self.chunk_success_count_global.scatter_add_(0, chunk_ids, successes)
            self.chunk_completion_sum_global.scatter_add_(0, chunk_ids, completion)

    def get_chunk_metrics(self):
        """Return per-chunk metrics dict."""
        eval_count = self.chunk_eval_count.clamp(min=1)
        return {
            "chunk_success_rate": self.chunk_success_count / eval_count,
            "chunk_fail_rate": self.chunk_fail_count / eval_count,
            "chunk_completion_frac": self.chunk_completion_sum / eval_count,
            "chunk_eval_count": self.chunk_eval_count,
        }

    def get_track1_summary(self):
        """Track 1: Duration-weighted chunk-level summary."""
        metrics = self.get_chunk_metrics()
        evaluated = self.chunk_eval_count > 0
        if not evaluated.any():
            return {}

        durations = self.chunk_duration_frames[evaluated].float()
        weights = durations / durations.sum()

        return {
            "track1/weighted_success_rate": (weights * metrics["chunk_success_rate"][evaluated]).sum().item(),
            "track1/weighted_completion": (weights * metrics["chunk_completion_frac"][evaluated]).sum().item(),
            "track1/chunks_evaluated": evaluated.sum().item(),
            "track1/chunks_total": self.num_chunks,
        }

    def get_track2_summary(self):
        """Track 2: Per-sequence success rate (unbiased across sequences)."""
        metrics = self.get_chunk_metrics()

        # Aggregate chunks to sequences via scatter
        seq_weighted_success = torch.zeros(self.num_sequences, device=self.device)
        seq_total_duration = torch.zeros(self.num_sequences, device=self.device)

        weighted_success = self.chunk_duration_frames.float() * metrics["chunk_success_rate"]
        seq_weighted_success.scatter_add_(0, self.chunk_to_seq, weighted_success)
        seq_total_duration.scatter_add_(0, self.chunk_to_seq, self.chunk_duration_frames.float())

        seq_success_rate = seq_weighted_success / seq_total_duration.clamp(min=1)

        # Only count sequences with at least one evaluated chunk
        seq_eval_count = torch.zeros(self.num_sequences, device=self.device)
        seq_eval_count.scatter_add_(0, self.chunk_to_seq, (self.chunk_eval_count > 0).float())
        evaluated_seqs = seq_eval_count > 0

        if not evaluated_seqs.any():
            return {}

        return {
            "track2/seq_mean_success_rate": seq_success_rate[evaluated_seqs].mean().item(),
            "track2/seq_worst_success_rate": seq_success_rate[evaluated_seqs].min().item(),
            "track2/seq_best_success_rate": seq_success_rate[evaluated_seqs].max().item(),
            "track2/seqs_evaluated": evaluated_seqs.sum().item(),
            "track2/seqs_total": self.num_sequences,
        }

    def get_global_summary(self):
        """Global cumulative stats across all of training (never reset)."""
        if not hasattr(self, 'chunk_eval_count_global'):
            return {}
        evaluated = self.chunk_eval_count_global > 0
        if not evaluated.any():
            return {}

        eval_count = self.chunk_eval_count_global.clamp(min=1)
        chunk_sr = self.chunk_success_count_global / eval_count
        chunk_cr = self.chunk_completion_sum_global / eval_count

        # Duration-weighted SR/CR over all ever-evaluated chunks
        durations = self.chunk_duration_frames[evaluated].float()
        weights = durations / durations.sum()

        return {
            "global/chunks_ever_evaluated": evaluated.sum().item(),
            "global/chunks_total": self.num_chunks,
            "global/coverage_pct": 100.0 * evaluated.sum().item() / self.num_chunks,
            "global/weighted_SR": (weights * chunk_sr[evaluated]).sum().item(),
            "global/weighted_CR": (weights * chunk_cr[evaluated]).sum().item(),
            "global/mean_evals_per_chunk": self.chunk_eval_count_global[evaluated].mean().item(),
            "global/max_evals_per_chunk": self.chunk_eval_count_global.max().item(),
        }

    def get_per_dataset_summary(self):
        """Per-dataset metrics using global counters (train and test)."""
        if not hasattr(self, 'chunk_to_dataset') or not hasattr(self, 'chunk_eval_count_global'):
            return {}
        evaluated = self.chunk_eval_count_global > 0
        if not evaluated.any():
            return {}

        eval_count = self.chunk_eval_count_global.clamp(min=1)
        chunk_sr = self.chunk_success_count_global / eval_count
        chunk_cr = self.chunk_completion_sum_global / eval_count

        result = {}
        for ds_id, ds_name in enumerate(self.dataset_names):
            ds_mask = self.chunk_to_dataset == ds_id
            ds_eval = ds_mask & evaluated

            # Train split
            train_mask = torch.zeros(self.num_chunks, dtype=torch.bool, device=self.device)
            train_mask[self._train_chunk_indices] = True
            ds_train = ds_mask & train_mask & evaluated
            if ds_train.any():
                dur = self.chunk_duration_frames[ds_train].float()
                w = dur / dur.sum()
                result[f"dataset_train/{ds_name}/SR"] = (w * chunk_sr[ds_train]).sum().item()
                result[f"dataset_train/{ds_name}/CR"] = (w * chunk_cr[ds_train]).sum().item()
                result[f"dataset_train/{ds_name}/coverage"] = ds_train.sum().item() / (ds_mask & train_mask).sum().item()

            # Test split
            test_mask = torch.zeros(self.num_chunks, dtype=torch.bool, device=self.device)
            if len(self._test_chunk_indices) > 0:
                test_mask[self._test_chunk_indices] = True
            ds_test = ds_mask & test_mask & evaluated
            if ds_test.any():
                dur = self.chunk_duration_frames[ds_test].float()
                w = dur / dur.sum()
                result[f"dataset_test/{ds_name}/SR"] = (w * chunk_sr[ds_test]).sum().item()
                result[f"dataset_test/{ds_name}/CR"] = (w * chunk_cr[ds_test]).sum().item()

            # Overall (train+test)
            if ds_eval.any():
                result[f"dataset/{ds_name}/n_chunks"] = ds_mask.sum().item()
                result[f"dataset/{ds_name}/n_evaluated"] = ds_eval.sum().item()

        return result

    def get_per_seqlen_summary(self):
        """Per-sequence-length bucket metrics using global counters."""
        if not hasattr(self, 'chunks_per_seq') or not hasattr(self, 'chunk_eval_count_global'):
            return {}
        evaluated = self.chunk_eval_count_global > 0
        if not evaluated.any():
            return {}

        eval_count = self.chunk_eval_count_global.clamp(min=1)
        chunk_sr = self.chunk_success_count_global / eval_count
        chunk_cr = self.chunk_completion_sum_global / eval_count

        # Map each chunk to its trajectory length (number of chunks in traj)
        traj_len_per_chunk = self.chunks_per_seq[self.chunk_to_seq]

        # Buckets: 1-3, 4-8, 9-16, 17-32, 33+
        buckets = [(1, 3), (4, 8), (9, 16), (17, 32), (33, 9999)]
        train_mask = torch.zeros(self.num_chunks, dtype=torch.bool, device=self.device)
        train_mask[self._train_chunk_indices] = True

        result = {}
        for lo, hi in buckets:
            label = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
            bucket_mask = (traj_len_per_chunk >= lo) & (traj_len_per_chunk <= hi)

            # Train
            bm_train = bucket_mask & train_mask & evaluated
            if bm_train.any():
                dur = self.chunk_duration_frames[bm_train].float()
                w = dur / dur.sum()
                result[f"seqlen_train/{label}/SR"] = (w * chunk_sr[bm_train]).sum().item()
                result[f"seqlen_train/{label}/CR"] = (w * chunk_cr[bm_train]).sum().item()
                result[f"seqlen_train/{label}/n_chunks"] = bm_train.sum().item()

            # All (including test)
            bm_all = bucket_mask & evaluated
            if bm_all.any():
                dur = self.chunk_duration_frames[bm_all].float()
                w = dur / dur.sum()
                result[f"seqlen/{label}/SR"] = (w * chunk_sr[bm_all]).sum().item()
                result[f"seqlen/{label}/CR"] = (w * chunk_cr[bm_all]).sum().item()
                result[f"seqlen/{label}/n_chunks_total"] = (bucket_mask).sum().item()

        return result

    def update_sampling_weights(self):
        """Update sampling weights: upweight failed chunks, downweight mastered ones."""
        if self.sampling_weights is None:
            return

        metrics = self.get_chunk_metrics()
        eval_count = self.chunk_eval_count
        evaluated = eval_count > 0

        if not evaluated.any():
            return

        # Failure-based weighting: chunks with lower success get higher weight
        # For unevaluated chunks, keep current weight (don't zero them out)
        success_rate = metrics["chunk_success_rate"]  # [num_chunks]

        # New weight = 1 - success_rate (more failures = higher weight)
        # Clamp to [min_weight, max_weight] to prevent starvation or domination
        min_weight = 0.1
        max_weight = 2.0
        new_weights = (1.0 - success_rate).clamp(min=min_weight, max=max_weight)

        # Only update weights for chunks that were actually evaluated
        self.sampling_weights[evaluated] = new_weights[evaluated]
        # Unevaluated chunks keep their current weight

    # === Two-level hierarchical sampling methods ===

    def update_bin_failures(self, done_env_ids):
        """EMA update of per-chunk failure rate and completion rate.
        Only updates for chunks actually observed this step.
        """
        if len(done_env_ids) == 0 or self.sampling_weights is None:
            return

        chunk_ids = self.motion_ids[done_env_ids]

        # Count failures and total terminations per chunk this step
        failures = self.failure_buf[done_env_ids].float()
        chunk_fail_this_step = torch.zeros(self.num_chunks, device=self.device)
        chunk_total_this_step = torch.zeros(self.num_chunks, device=self.device)
        chunk_fail_this_step.scatter_add_(0, chunk_ids, failures)
        chunk_total_this_step.scatter_add_(0, chunk_ids, torch.ones_like(failures))

        # Completion fraction per env
        seq_lens = self.motion_num_frames[chunk_ids].float()
        completion = self.progress_buf[done_env_ids].float() / seq_lens.clamp(min=1)
        chunk_completion_this_step = torch.zeros(self.num_chunks, device=self.device)
        chunk_completion_this_step.scatter_add_(0, chunk_ids, completion)

        # Compute per-chunk rates for observed chunks
        observed = chunk_total_this_step > 0
        fail_rate = chunk_fail_this_step / chunk_total_this_step.clamp(min=1)
        comp_rate = chunk_completion_this_step / chunk_total_this_step.clamp(min=1)

        # EMA update only for observed chunks
        alpha = self.bin_ema_alpha
        self.bin_failed_count[observed] = (
            alpha * fail_rate[observed] + (1 - alpha) * self.bin_failed_count[observed]
        )
        self.bin_completion_ema[observed] = (
            alpha * comp_rate[observed] + (1 - alpha) * self.bin_completion_ema[observed]
        )

    def _handle_chunk_continuation(self):
        """Cross-chunk continuation: advance to next chunk instead of resetting.

        Called in post_physics_step AFTER update_chunk_metrics and
        update_bin_failures (so chunk completion is recorded) but BEFORE
        progress_buf is incremented.

        For envs that completed their chunk (success_buf=1) and have a
        successor chunk in the same trajectory:
        - Override reset_buf=0 (prevent reset_done from firing)
        - Advance motion_ids to the next chunk
        - Set progress_buf=-1 (becomes 0 after the +=1 in post_physics_step)
        - Do NOT reset physics state (hand continues from where it is)
        """
        if not hasattr(self, 'next_chunk'):
            return

        continue_mask = (
            (self.success_buf == 1)
            & (self.next_chunk[self.motion_ids] >= 0)
        )

        if not continue_mask.any():
            return

        continue_ids = continue_mask.nonzero(as_tuple=False).flatten()

        # Override: don't reset these envs
        self.reset_buf[continue_ids] = 0
        self.success_buf[continue_ids] = 0

        # Advance to next chunk (NO physics/PID/controller reset)
        self.motion_ids[continue_ids] = self.next_chunk[self.motion_ids[continue_ids]]
        self.progress_buf[continue_ids] = -1  # after +=1 → 0 = frame 0 of new chunk

        # Keep running_progress_buf counting: no grace period at chunk boundaries.
        # The policy must maintain tracking quality across transitions.
        # (running_progress_buf is only reset on full episode reset in _reset_default)

        # Reset chunk-max tracking for the new chunk (old chunk max already recorded)
        self._err_chunk_max[continue_ids] = 0
        self._err_chunk_max_active[continue_ids] = False

    def _convolve_within_seq(self, values):
        """Spread failure blame to preceding chunks within same trajectory.
        BeyondMimic: chunk i's failure is partially attributed to chunks i-1, i-2.
        Kernel [lam^2, lam, 1] applied backward (preceding chunks get blame)."""
        lam = self.bin_conv_lambda
        out = values.clone()

        # Shift by 1: chunk i-1 receives lam * values[i]
        if self.num_chunks > 1:
            shifted1 = torch.zeros_like(values)
            shifted1[:-1] = values[1:] * lam
            # Zero out where sequence boundary (different trajectory)
            same_seq_1 = (self.chunk_to_seq[:-1] == self.chunk_to_seq[1:])
            shifted1[:-1] *= same_seq_1.float()
            out += shifted1

        # Shift by 2: chunk i-2 receives lam^2 * values[i]
        if self.num_chunks > 2:
            shifted2 = torch.zeros_like(values)
            shifted2[:-2] = values[2:] * (lam * lam)
            same_seq_2 = (self.chunk_to_seq[:-2] == self.chunk_to_seq[2:])
            shifted2[:-2] *= same_seq_2.float()
            out += shifted2

        return out

    def _recompute_sampling_weights(self):
        """Recompute sampling weights."""
        if self.sampling_weights is None:
            return
        if self.sampling_method == "twolevel":
            self._recompute_weights_twolevel()
        elif self.sampling_method == "mastery":
            self._recompute_weights_mastery()
        else:
            self._recompute_weights_sonic()

    def _recompute_weights_sonic(self):
        """SONIC / BeyondMimic style: α * failure_adaptive + (1-α) * uniform.

        All chunks in a single flat pool.  Longer trajectories naturally get
        proportionally more samples because they contribute more bins.
        """
        N = len(self._train_chunk_indices)

        # 1. Adaptive: EMA failure counts + BeyondMimic uniform floor (0.1 / S_traj)
        uniform_floor = 0.1 / self.chunks_per_seq[self.chunk_to_seq].clamp(min=1)
        adaptive = self.bin_failed_count + uniform_floor

        # 2. Convolve within trajectory (BeyondMimic backward blame spread)
        adaptive = self._convolve_within_seq(adaptive)

        # 3. SONIC cap: cap per-chunk at beta * global mean
        global_mean = adaptive[self._train_chunk_indices].mean()
        cap = self.sonic_cap_beta * global_mean
        adaptive = torch.clamp(adaptive, max=cap)

        # 4. Zero out test chunks before normalization
        if len(self._test_chunk_indices) > 0:
            adaptive[self._test_chunk_indices] = 0.0

        # 5. Normalize globally across all training chunks
        adaptive_sum = adaptive.sum()
        if adaptive_sum > 0:
            adaptive_norm = adaptive / adaptive_sum
        else:
            adaptive_norm = torch.zeros_like(adaptive)
            adaptive_norm[self._train_chunk_indices] = 1.0 / N

        # 6. Blend: α * adaptive + (1-α) * uniform
        uniform = torch.zeros_like(adaptive)
        uniform[self._train_chunk_indices] = 1.0 / N

        self.sampling_weights[:] = (
            self.sonic_blend_alpha * adaptive_norm
            + (1 - self.sonic_blend_alpha) * uniform
        )
        self.sampling_weights *= self._dataset_weight_multiplier

        # Numerical safety
        self.sampling_weights.clamp_(min=1e-8)

        # Ensure test chunks stay at 0
        if len(self._test_chunk_indices) > 0:
            self.sampling_weights[self._test_chunk_indices] = 0.0

    def _recompute_weights_mastery(self):
        """Mastery-based: α * mastery_adaptive + (1-α) * uniform.

        Adaptive component: w_i ∝ max(1 - completion_ema_i, floor).
        Blended with uniform to ensure full dataset coverage.
        Uses sonic_blend_alpha for the blend ratio (default 0.1 = 10% adaptive).
        """
        N = len(self._train_chunk_indices)

        # 1. Raw mastery weight: 1 - completion (unmastered fraction)
        raw = 1.0 - self.bin_completion_ema

        # 2. Convolve within trajectory (backward blame spread)
        raw = self._convolve_within_seq(raw)

        # 3. Floor: prevent mastered chunks from zero adaptive weight
        raw.clamp_(min=self.mastery_floor)

        # 4. Zero out test chunks
        if len(self._test_chunk_indices) > 0:
            raw[self._test_chunk_indices] = 0.0

        # 5. Normalize adaptive component
        total = raw.sum()
        if total > 0:
            adaptive_norm = raw / total
        else:
            adaptive_norm = torch.zeros_like(raw)
            adaptive_norm[self._train_chunk_indices] = 1.0 / N

        # 6. Blend: α * adaptive + (1-α) * uniform
        uniform = torch.zeros_like(raw)
        uniform[self._train_chunk_indices] = 1.0 / N

        alpha = self.sonic_blend_alpha
        self.sampling_weights[:] = alpha * adaptive_norm + (1 - alpha) * uniform
        self.sampling_weights *= self._dataset_weight_multiplier

        # Numerical safety
        self.sampling_weights.clamp_(min=1e-8)

        # Ensure test chunks stay at 0
        if len(self._test_chunk_indices) > 0:
            self.sampling_weights[self._test_chunk_indices] = 0.0

    def _recompute_weights_twolevel(self):
        """Two-level hierarchical: trajectory-level then chunk-level.

        Level 1 (trajectory): α1 * adaptive_traj + (1-α1) * uniform_traj
            adaptive_traj = mean(1 - completion_ema) per trajectory
        Level 2 (chunk within traj): α2 * adaptive_chunk + (1-α2) * uniform_chunk
            adaptive_chunk = (1 - completion_ema) normalized within trajectory

        Final per-chunk weight = P(traj) * P(chunk | traj)
        Uses sonic_blend_alpha for α1 (level-1), twolevel_chunk_alpha for α2 (level-2).
        """
        alpha1 = self.sonic_blend_alpha       # level-1: 0.1 = 10% adaptive
        alpha2 = self.twolevel_chunk_alpha     # level-2: 0.5 = 50% adaptive

        # --- Per-chunk mastery (same as mastery method) ---
        chunk_mastery = 1.0 - self.bin_completion_ema  # higher = harder/unseen
        chunk_mastery = self._convolve_within_seq(chunk_mastery)
        chunk_mastery.clamp_(min=self.mastery_floor)

        # Zero test chunks
        if len(self._test_chunk_indices) > 0:
            chunk_mastery[self._test_chunk_indices] = 0.0

        # --- Level 1: trajectory weights ---
        N_trajs = self.num_sequences
        # Adaptive: mean mastery per trajectory
        traj_mastery_sum = torch.zeros(N_trajs, device=self.device)
        traj_mastery_sum.scatter_add_(0, self.chunk_to_seq, chunk_mastery)
        traj_adaptive = traj_mastery_sum / self.chunks_per_seq.clamp(min=1)

        # Identify training trajectories (have at least one training chunk)
        train_chunk_mask = torch.zeros(self.num_chunks, dtype=torch.bool, device=self.device)
        train_chunk_mask[self._train_chunk_indices] = True
        train_traj_count = torch.zeros(N_trajs, device=self.device)
        train_traj_count.scatter_add_(0, self.chunk_to_seq, train_chunk_mask.float())
        train_traj_mask = train_traj_count > 0
        N_train_trajs = train_traj_mask.sum().item()

        # Normalize adaptive over training trajectories
        traj_adaptive[~train_traj_mask] = 0.0
        traj_adapt_sum = traj_adaptive.sum()
        if traj_adapt_sum > 0:
            traj_adaptive_norm = traj_adaptive / traj_adapt_sum
        else:
            traj_adaptive_norm = torch.zeros_like(traj_adaptive)
            traj_adaptive_norm[train_traj_mask] = 1.0 / max(N_train_trajs, 1)

        # Uniform over training trajectories
        traj_uniform = torch.zeros(N_trajs, device=self.device)
        traj_uniform[train_traj_mask] = 1.0 / max(N_train_trajs, 1)

        # Blend level 1
        traj_weight = alpha1 * traj_adaptive_norm + (1 - alpha1) * traj_uniform

        # --- Level 2: chunk weights within trajectory ---
        # Per-chunk adaptive normalized within its trajectory
        # chunk_adaptive_within_traj = chunk_mastery / sum(chunk_mastery in same traj)
        traj_mastery_sum_expanded = traj_mastery_sum[self.chunk_to_seq]
        chunk_adaptive_within = chunk_mastery / traj_mastery_sum_expanded.clamp(min=1e-8)

        # Uniform within trajectory
        chunk_uniform_within = 1.0 / self.chunks_per_seq[self.chunk_to_seq].clamp(min=1)
        # Zero test chunks in uniform too
        if len(self._test_chunk_indices) > 0:
            chunk_uniform_within[self._test_chunk_indices] = 0.0

        # Blend level 2
        chunk_within_weight = alpha2 * chunk_adaptive_within + (1 - alpha2) * chunk_uniform_within

        # --- Final: P(traj) * P(chunk | traj) ---
        traj_weight_per_chunk = traj_weight[self.chunk_to_seq]
        self.sampling_weights[:] = traj_weight_per_chunk * chunk_within_weight
        self.sampling_weights *= self._dataset_weight_multiplier

        # Numerical safety
        self.sampling_weights.clamp_(min=1e-8)

        # Ensure test chunks stay at 0
        if len(self._test_chunk_indices) > 0:
            self.sampling_weights[self._test_chunk_indices] = 0.0

    def reset_chunk_metrics(self):
        """Reset all chunk-level metrics (call periodically for windowed stats)."""
        self.chunk_success_count.zero_()
        self.chunk_fail_count.zero_()
        self.chunk_eval_count.zero_()
        self.chunk_completion_sum.zero_()
        self.chunk_reward_sum.zero_()

    def _record_chunk_max_errors(self, done_env_ids):
        """Record per-chunk max errors into histograms when episodes end."""
        if len(done_env_ids) == 0:
            return
        # Only record envs that had at least one active frame (past grace period)
        valid = self._err_chunk_max_active[done_env_ids]
        if not valid.any():
            return
        valid_ids = done_env_ids[valid]

        # Global chunk-max histograms
        for gi, g in enumerate(self._err_joint_groups):
            maxvals = self._err_chunk_max[valid_ids, gi]
            hmax = self._err_hist_range[g]
            hist = torch.histc(maxvals, bins=self._err_hist_nbins, min=0, max=hmax)
            self._err_chunk_max_histograms[g] += hist
        self._err_chunk_max_count += len(valid_ids)

        # Per-dataset chunk-max histograms
        if hasattr(self, '_err_chunk_max_histograms_by_dataset'):
            env_ds = self.chunk_to_dataset[self.motion_ids[valid_ids]]
            for ds_id in range(self.num_datasets):
                ds_mask = env_ds == ds_id
                if not ds_mask.any():
                    continue
                ds_ids = valid_ids[ds_mask]
                self._err_chunk_max_count_by_dataset[ds_id] += len(ds_ids)
                for gi, g in enumerate(self._err_joint_groups):
                    maxvals = self._err_chunk_max[ds_ids, gi]
                    hmax = self._err_hist_range[g]
                    hist = torch.histc(maxvals, bins=self._err_hist_nbins, min=0, max=hmax)
                    self._err_chunk_max_histograms_by_dataset[g][ds_id] += hist

    def _percentiles_from_hist(self, hist, hmax, to_mm=False):
        """Compute percentiles from a single histogram. Returns dict {pct: value}."""
        total = hist.sum()
        if total < 1:
            return {}
        bin_width = hmax / self._err_hist_nbins
        bin_centers = torch.arange(self._err_hist_nbins, device=self.device) * bin_width + bin_width / 2
        cdf = hist.cumsum(0) / total
        result = {}
        for pct in [50, 75, 90, 95, 99]:
            idx = (cdf >= pct / 100.0).nonzero(as_tuple=True)[0]
            val = bin_centers[idx[0]].item() if len(idx) > 0 else hmax
            if to_mm:
                val = val * 1000
            result[pct] = val
        return result

    def get_error_percentiles(self):
        """Compute per-joint-group error percentiles from all histogram types."""
        result = {}
        pos_groups = {"thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
                      "level_1", "level_2", "wrist_pos"}

        # === Global frame-level percentiles ===
        if self._err_hist_count > 0:
            for g in self._err_joint_groups:
                hmax = self._err_hist_range[g]
                pcts = self._percentiles_from_hist(
                    self._err_histograms[g], hmax, to_mm=(g in pos_groups))
                for pct, val in pcts.items():
                    result["err_pct/%s/p%d" % (g, pct)] = val
            result["err_pct/sample_count"] = self._err_hist_count

        # === Global chunk-max percentiles ===
        if self._err_chunk_max_count > 0:
            for g in self._err_joint_groups:
                hmax = self._err_hist_range[g]
                pcts = self._percentiles_from_hist(
                    self._err_chunk_max_histograms[g], hmax, to_mm=(g in pos_groups))
                for pct, val in pcts.items():
                    result["err_chunk_max/%s/p%d" % (g, pct)] = val
            result["err_chunk_max/chunk_count"] = self._err_chunk_max_count

        # === Per-dataset frame-level percentiles ===
        if hasattr(self, '_err_histograms_by_dataset'):
            for ds_id, ds_name in enumerate(self.dataset_names):
                cnt = self._err_hist_count_by_dataset[ds_id].item()
                if cnt == 0:
                    continue
                for g in self._err_joint_groups:
                    hmax = self._err_hist_range[g]
                    pcts = self._percentiles_from_hist(
                        self._err_histograms_by_dataset[g][ds_id], hmax, to_mm=(g in pos_groups))
                    for pct, val in pcts.items():
                        result["err_pct_ds/%s/%s/p%d" % (ds_name, g, pct)] = val

        # === Per-dataset chunk-max percentiles ===
        if hasattr(self, '_err_chunk_max_histograms_by_dataset'):
            for ds_id, ds_name in enumerate(self.dataset_names):
                cnt = self._err_chunk_max_count_by_dataset[ds_id].item()
                if cnt == 0:
                    continue
                for g in self._err_joint_groups:
                    hmax = self._err_hist_range[g]
                    pcts = self._percentiles_from_hist(
                        self._err_chunk_max_histograms_by_dataset[g][ds_id], hmax, to_mm=(g in pos_groups))
                    for pct, val in pcts.items():
                        result["err_chunk_max_ds/%s/%s/p%d" % (ds_name, g, pct)] = val

        # === Per-dataset effective sampling fractions ===
        if hasattr(self, 'sampling_weights') and self.sampling_weights is not None:
            w = self.sampling_weights
            w_sum = w.sum()
            if w_sum > 0:
                for ds_id, ds_name in enumerate(self.dataset_names):
                    ds_mask = self.chunk_to_dataset == ds_id
                    frac = w[ds_mask].sum().item() / w_sum.item()
                    result["sampling_frac/%s" % ds_name] = frac

        return result

    def _percentile_from_hist_raw(self, hist, hmax, pct):
        """Compute a single percentile from a histogram. Returns value in native units (meters)."""
        total = hist.sum()
        if total < 1:
            return None
        bin_width = hmax / self._err_hist_nbins
        bin_centers = torch.arange(self._err_hist_nbins, device=self.device) * bin_width + bin_width / 2
        cdf = hist.cumsum(0) / total
        idx = (cdf >= pct / 100.0).nonzero(as_tuple=True)[0]
        return bin_centers[idx[0]].item() if len(idx) > 0 else hmax

    def update_adaptive_thresholds(self):
        """Update per-dataset per-group adaptive thresholds from chunk-max histograms.
        Only does work when adaptive_alpha > 0. Returns dict for TB logging."""
        result = {}
        if self._adaptive_alpha <= 0 or not hasattr(self, '_adaptive_thresholds'):
            return result

        pct = self._adaptive_alpha * 100  # e.g. 0.9 -> p90
        min_chunks = 50

        for gi, g in enumerate(self._termination_groups):
            hmax = self._err_hist_range[g]
            # Global fallback: p(alpha) from global chunk-max histogram
            global_val = None
            if self._err_chunk_max_count >= min_chunks:
                global_val = self._percentile_from_hist_raw(
                    self._err_chunk_max_histograms[g], hmax, pct)

            for ds_id in range(self.num_datasets):
                ds_name = self.dataset_names[ds_id]
                cnt = self._err_chunk_max_count_by_dataset[ds_id].item()
                if cnt >= min_chunks:
                    val = self._percentile_from_hist_raw(
                        self._err_chunk_max_histograms_by_dataset[g][ds_id], hmax, pct)
                elif global_val is not None:
                    val = global_val
                else:
                    val = None  # keep default

                if val is not None:
                    val = max(val, self._base_thresholds[gi].item())
                    self._adaptive_thresholds[ds_id, gi] = val
                    result["adaptive_th/%s/%s" % (ds_name, g)] = val * 1000  # log in mm

        return result

    def reset_error_histograms(self):
        """Reset all error histograms (call after logging)."""
        for g in self._err_joint_groups:
            self._err_histograms[g].zero_()
            self._err_chunk_max_histograms[g].zero_()
        self._err_hist_count = 0
        self._err_chunk_max_count = 0
        if hasattr(self, '_err_histograms_by_dataset'):
            for g in self._err_joint_groups:
                self._err_histograms_by_dataset[g].zero_()
                self._err_chunk_max_histograms_by_dataset[g].zero_()
            self._err_hist_count_by_dataset.zero_()
            self._err_chunk_max_count_by_dataset.zero_()

    def get_split_indices(self, split_name):
        """Return chunk indices for the given split."""
        if split_name == "test":
            return self._test_chunk_indices
        return self._train_chunk_indices

    def _build_test_trajectory_map(self):
        """Build ordered list of chunk indices per test trajectory.

        Populates:
            self._test_trajectory_chunks: list of 1-D long tensors, each containing
                the ordered chunk indices for one test trajectory.
            self._test_trajectory_total_frames: [num_test_trajectories] long tensor
                with the total number of frames per trajectory.
        """
        if hasattr(self, '_test_trajectory_chunks'):
            return  # already built

        test_chunks = self._test_chunk_indices  # [num_test_chunks]
        test_seq_ids = self.chunk_to_seq[test_chunks]  # sequence id per test chunk

        unique_seqs = test_seq_ids.unique(sorted=True)
        traj_chunks_list = []
        total_frames_list = []

        for seq_id in unique_seqs:
            mask = test_seq_ids == seq_id
            chunks_in_traj = test_chunks[mask]
            # Sort by chunk index (should already be consecutive, but be safe)
            chunks_in_traj = chunks_in_traj.sort()[0]
            traj_chunks_list.append(chunks_in_traj)
            total_frames_list.append(self.motion_num_frames[chunks_in_traj].sum())

        self._test_trajectory_chunks = traj_chunks_list
        self._test_trajectory_total_frames = torch.stack(total_frames_list)

    def cache_training_state(self):
        """Save training state before evaluation."""
        self._cached_state = {
            "motion_ids": self.motion_ids.clone(),
            "progress_buf": self.progress_buf.clone(),
            "running_progress_buf": self.running_progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "success_buf": self.success_buf.clone(),
            "failure_buf": self.failure_buf.clone(),
            "error_buf": self.error_buf.clone(),
            "total_rew_buf": self.total_rew_buf.clone(),
            "random_state_init": self.random_state_init,
            # Physics state
            "_dof_state": self._dof_state.clone(),
            "_root_state": self._root_state.clone(),
            "_pos_control": self._pos_control.clone(),
            "curr_targets": self.curr_targets.clone(),
            "prev_targets": self.prev_targets.clone(),
            "apply_forces": self.apply_forces.clone(),
            "apply_torque": self.apply_torque.clone(),
        }
        # Chunk metrics (prevent eval from polluting training statistics)
        if self.sampling_weights is not None:
            self._cached_state["chunk_eval_count"] = self.chunk_eval_count.clone()
            self._cached_state["chunk_success_count"] = self.chunk_success_count.clone()
            self._cached_state["chunk_fail_count"] = self.chunk_fail_count.clone()
            self._cached_state["chunk_completion_sum"] = self.chunk_completion_sum.clone()
            self._cached_state["chunk_reward_sum"] = self.chunk_reward_sum.clone()
            self._cached_state["bin_failed_count"] = self.bin_failed_count.clone()
            self._cached_state["bin_completion_ema"] = self.bin_completion_ema.clone()
            if hasattr(self, 'chunk_eval_count_global'):
                self._cached_state["chunk_eval_count_global"] = self.chunk_eval_count_global.clone()
                self._cached_state["chunk_success_count_global"] = self.chunk_success_count_global.clone()
                self._cached_state["chunk_completion_sum_global"] = self.chunk_completion_sum_global.clone()
        # Error histograms (prevent eval from polluting adaptive thresholds)
        self._cached_state["_err_hist_count"] = self._err_hist_count
        self._cached_state["_err_chunk_max_count"] = self._err_chunk_max_count
        self._cached_state["_err_histograms"] = {g: h.clone() for g, h in self._err_histograms.items()}
        self._cached_state["_err_chunk_max_histograms"] = {g: h.clone() for g, h in self._err_chunk_max_histograms.items()}
        self._cached_state["_err_chunk_max"] = self._err_chunk_max.clone()
        self._cached_state["_err_chunk_max_active"] = self._err_chunk_max_active.clone()
        if hasattr(self, '_err_histograms_by_dataset'):
            self._cached_state["_err_histograms_by_dataset"] = {g: h.clone() for g, h in self._err_histograms_by_dataset.items()}
            self._cached_state["_err_chunk_max_histograms_by_dataset"] = {g: h.clone() for g, h in self._err_chunk_max_histograms_by_dataset.items()}
            self._cached_state["_err_hist_count_by_dataset"] = self._err_hist_count_by_dataset.clone()
            self._cached_state["_err_chunk_max_count_by_dataset"] = self._err_chunk_max_count_by_dataset.clone()
        # PID state if applicable
        if self.use_pid_control:
            self._cached_state["prev_pos_error"] = self.prev_pos_error.clone()
            self._cached_state["prev_rot_error"] = self.prev_rot_error.clone()
            self._cached_state["pos_error_integral"] = self.pos_error_integral.clone()
            self._cached_state["rot_error_integral"] = self.rot_error_integral.clone()

    def restore_training_state(self):
        """Restore training state after evaluation."""
        c = self._cached_state
        self.motion_ids[:] = c["motion_ids"]
        self.progress_buf[:] = c["progress_buf"]
        self.running_progress_buf[:] = c["running_progress_buf"]
        self.reset_buf[:] = c["reset_buf"]
        self.success_buf[:] = c["success_buf"]
        self.failure_buf[:] = c["failure_buf"]
        self.error_buf[:] = c["error_buf"]
        self.total_rew_buf[:] = c["total_rew_buf"]
        self.random_state_init = c["random_state_init"]
        # Physics state
        self._dof_state[:] = c["_dof_state"]
        self._root_state[:] = c["_root_state"]
        self._pos_control[:] = c["_pos_control"]
        self.curr_targets[:] = c["curr_targets"]
        self.prev_targets[:] = c["prev_targets"]
        self.apply_forces[:] = c["apply_forces"]
        self.apply_torque[:] = c["apply_torque"]
        if self.use_pid_control:
            self.prev_pos_error[:] = c["prev_pos_error"]
            self.prev_rot_error[:] = c["prev_rot_error"]
            self.pos_error_integral[:] = c["pos_error_integral"]
            self.rot_error_integral[:] = c["rot_error_integral"]
        # Chunk metrics
        if "chunk_eval_count" in c:
            self.chunk_eval_count[:] = c["chunk_eval_count"]
            self.chunk_success_count[:] = c["chunk_success_count"]
            self.chunk_fail_count[:] = c["chunk_fail_count"]
            self.chunk_completion_sum[:] = c["chunk_completion_sum"]
            self.chunk_reward_sum[:] = c["chunk_reward_sum"]
            self.bin_failed_count[:] = c["bin_failed_count"]
            if "bin_completion_ema" in c:
                self.bin_completion_ema[:] = c["bin_completion_ema"]
            if "chunk_eval_count_global" in c:
                self.chunk_eval_count_global[:] = c["chunk_eval_count_global"]
                self.chunk_success_count_global[:] = c["chunk_success_count_global"]
                self.chunk_completion_sum_global[:] = c["chunk_completion_sum_global"]
        # Error histograms
        if "_err_hist_count" in c:
            self._err_hist_count = c["_err_hist_count"]
            self._err_chunk_max_count = c["_err_chunk_max_count"]
            for g in self._err_histograms:
                self._err_histograms[g][:] = c["_err_histograms"][g]
                self._err_chunk_max_histograms[g][:] = c["_err_chunk_max_histograms"][g]
            self._err_chunk_max[:] = c["_err_chunk_max"]
            self._err_chunk_max_active[:] = c["_err_chunk_max_active"]
            if "_err_histograms_by_dataset" in c:
                for g in self._err_histograms_by_dataset:
                    self._err_histograms_by_dataset[g][:] = c["_err_histograms_by_dataset"][g]
                    self._err_chunk_max_histograms_by_dataset[g][:] = c["_err_chunk_max_histograms_by_dataset"][g]
                self._err_hist_count_by_dataset[:] = c["_err_hist_count_by_dataset"]
                self._err_chunk_max_count_by_dataset[:] = c["_err_chunk_max_count_by_dataset"]
        # Re-apply physics state to simulator using INDEXED versions (take effect immediately).
        # Non-indexed set_*_tensor is deferred until simulate(), but refresh_*_tensor in
        # compute_observations would overwrite the tensor with stale (eval) state first.
        all_indices = self._global_dexhand_indices.flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(all_indices),
            len(all_indices),
        )
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        del self._cached_state

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    0.6,
                    0.6,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )


@torch.jit.script
def quat_to_angle_axis(q):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    # computes axis-angle representation from quaternion q
    # q must be normalized
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    term_thresholds: Tensor,
    dexhand_weight_idx: Dict[str, List[int]],
    wrist_power_weight: float,
    wrist_pos_weight: float,
    wrist_ang_vel_weight: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    # type: (Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor], Tensor, Tensor, Dict[str, List[int]], float, float, float) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]

    # end effector pose reward
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

    # ? assign different weights to different joints
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_pos = states["q"]
    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            (diff_thumb_tip_pos_dist > term_thresholds[:, 0])
            | (diff_index_tip_pos_dist > term_thresholds[:, 1])
            | (diff_middle_tip_pos_dist > term_thresholds[:, 2])
            | (diff_pinky_tip_pos_dist > term_thresholds[:, 3])
            | (diff_ring_tip_pos_dist > term_thresholds[:, 4])
            | (diff_level_1_pos_dist > term_thresholds[:, 5])
            | (diff_level_2_pos_dist > term_thresholds[:, 6])
        )
        & (running_progress_buf >= 20)
    ) | error_buf
    reward_execute = (
        wrist_pos_weight * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.75 * reward_middle_tip_pos
        + 0.6 * reward_pinky_tip_pos
        + 0.6 * reward_ring_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.3 * reward_level_2_pos
        + 0.1 * reward_eef_vel
        + wrist_ang_vel_weight * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
        + 0.5 * reward_power
        + wrist_power_weight * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 >= max_length
    ) & ~failed_execute  # reached the end of the chunk (obs lookahead is safe due to clamp at line 912)
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        # Raw per-joint-group L2 errors (meters) for adaptive threshold statistics
        "err_thumb_tip": diff_thumb_tip_pos_dist,
        "err_index_tip": diff_index_tip_pos_dist,
        "err_middle_tip": diff_middle_tip_pos_dist,
        "err_ring_tip": diff_ring_tip_pos_dist,
        "err_pinky_tip": diff_pinky_tip_pos_dist,
        "err_level_1": diff_level_1_pos_dist,
        "err_level_2": diff_level_2_pos_dist,
        # Wrist errors (no termination threshold, but useful for statistics)
        "err_wrist_pos": diff_eef_pos_dist,
        "err_wrist_rot": diff_eef_rot_angle.abs(),
        "err_wrist_vel": diff_eef_vel.abs().mean(dim=-1),
        "err_wrist_ang_vel": diff_eef_ang_vel.abs().mean(dim=-1),
        "err_joints_vel": diff_joints_vel.abs().mean(dim=-1).mean(-1),
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict


class DexHandImitatorLHEnv(DexHandImitatorRHEnv):
    side = "left"

    def __init__(
        self,
        cfg,
        *,
        rl_device=0,
        sim_device=0,
        graphics_device_id=0,
        display=False,
        record=False,
        headless=True,
    ):
        self.dexhand = DexHandFactory.create_hand(cfg["env"]["dexhand"], "left")
        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
