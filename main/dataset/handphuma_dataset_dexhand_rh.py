import csv
import os
from functools import lru_cache

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp
from termcolor import cprint

from main.dataset.transform import aa_to_rotmat, rotmat_to_aa
from .base import ManipData
from .decorators import register_manipdata

# Involution: undoes the YUP_TO_ZUP applied during preprocessing (common.py).
# Preprocessing converts MANO Y-up -> Z-up; this offset reverses it so that
# mujoco2gym_transf @ _YUP_TO_ZUP @ (YUP_TO_ZUP @ raw) = mujoco2gym_transf @ raw,
# matching what the OakInk2 ManipTrans loader produces (identity offset).
_YUP_TO_ZUP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)


@register_manipdata("handphuma_rh")
class HandPhumaDatasetDexHandRH(ManipData):
    def __init__(
        self,
        *,
        data_dir: str,
        device="cuda:0",
        mujoco2gym_transf=None,
        max_seq_len=int(1e10),
        dexhand=None,
        target_fps: int = 60,
        index_path: str = "",
        **kwargs,
    ):
        super().__init__(
            data_dir=data_dir,
            device=device,
            mujoco2gym_transf=mujoco2gym_transf,
            max_seq_len=max_seq_len,
            dexhand=dexhand,
            **kwargs,
        )

        self.target_fps = target_fps

        # Load CSV index
        index_file = os.path.join(data_dir, index_path)
        self.sequences = []
        with open(index_file, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.sequences.append(row)
        self.data_pathes = []
        for seq in self.sequences:
            npy_path = os.path.join(data_dir, seq["dataset"], seq["filename"], "rhand.npy")
            self.data_pathes.append(npy_path)

        # Offset: undo the YUP_TO_ZUP that preprocessing applied
        transf_offset = np.eye(4)
        transf_offset[:3, :3] = _YUP_TO_ZUP
        self.transf_offset = torch.tensor(transf_offset, dtype=torch.float32, device=mujoco2gym_transf.device)
        self.mujoco2gym_transf = mujoco2gym_transf @ self.transf_offset

    @staticmethod
    def _resample(wrist_pos, wrist_rot, mano_joints, source_fps, target_fps):
        """Resample temporal data from source_fps to target_fps.

        Handles both downsampling (e.g. 120->60) and upsampling (e.g. 30->60).
        Uses linear interpolation for positions, SLERP for rotations.
        All inputs/outputs are numpy arrays.
        """
        if source_fps == target_fps:
            return wrist_pos, wrist_rot, mano_joints

        T_src = wrist_pos.shape[0]
        T_tgt = max(1, round(T_src * target_fps / source_fps))

        # Integer downsampling: simple slicing (most common case: 120->60)
        if source_fps > target_fps and source_fps % target_fps == 0:
            step = source_fps // target_fps
            indices = np.arange(0, T_src, step)[:T_tgt]
            return (
                wrist_pos[indices],
                wrist_rot[indices],
                {k: v[indices] for k, v in mano_joints.items()},
            )

        # General case: interpolation
        t_src = np.linspace(0, 1, T_src)
        t_tgt = np.linspace(0, 1, T_tgt)

        wrist_pos = interp1d(t_src, wrist_pos, axis=0)(t_tgt).astype(np.float32)
        mano_joints = {
            k: interp1d(t_src, v, axis=0)(t_tgt).astype(np.float32)
            for k, v in mano_joints.items()
        }

        rots = Rotation.from_matrix(wrist_rot)
        slerp = Slerp(t_src, rots)
        wrist_rot = slerp(t_tgt).as_matrix().astype(np.float32)

        return wrist_pos, wrist_rot, mano_joints

    @lru_cache(maxsize=None)
    def __getitem__(self, index):
        if isinstance(index, str) and index.startswith("hp"):
            idx = int(index[2:])
        else:
            idx = int(index)

        assert 0 <= idx < len(self.data_pathes), f"index {idx} out of range [0, {len(self.data_pathes)})"

        seq_info = self.sequences[idx]
        npy_path = self.data_pathes[idx]

        source_fps = int(seq_info["fps"])

        raw = np.load(npy_path, allow_pickle=True).item()

        wrist_pos = raw["wrist_pos"]    # (T_raw, 3)  numpy
        wrist_rot = raw["wrist_rot"]    # (T_raw, 3, 3) numpy
        mano_joints = raw["mano_joints"]  # dict of numpy

        # Check for pre-computed velocities
        has_precomputed_vel = "wrist_velocity" in raw

        # Chunk slicing (backward compatible: no-op if start_frame/end_frame absent)
        if "start_frame" in seq_info and "end_frame" in seq_info:
            sf, ef = int(seq_info["start_frame"]), int(seq_info["end_frame"])
            wrist_pos, wrist_rot = wrist_pos[sf:ef], wrist_rot[sf:ef]
            mano_joints = {k: v[sf:ef] for k, v in mano_joints.items()}
            if has_precomputed_vel:
                wrist_velocity = raw["wrist_velocity"][sf:ef]
                wrist_angular_velocity = raw["wrist_angular_velocity"][sf:ef]
                mano_joints_velocity = {k: v[sf:ef] for k, v in raw["mano_joints_velocity"].items()}
        elif has_precomputed_vel:
            wrist_velocity = raw["wrist_velocity"]
            wrist_angular_velocity = raw["wrist_angular_velocity"]
            mano_joints_velocity = raw["mano_joints_velocity"]

        # Resample to target_fps
        wrist_pos, wrist_rot, mano_joints = self._resample(
            wrist_pos, wrist_rot, mano_joints, source_fps, self.target_fps
        )

        # Convert to tensors on device
        wrist_pos = torch.tensor(wrist_pos, device=self.device)
        wrist_rot = torch.tensor(wrist_rot, device=self.device)
        mano_joints = {k: torch.tensor(v, device=self.device) for k, v in mano_joints.items()}

        length = wrist_pos.shape[0]

        # Apply dexhand relative offset
        wrist_pos = wrist_pos + torch.tensor(
            self.dexhand.relative_translation, device=self.device, dtype=torch.float32
        )
        wrist_rot = wrist_rot @ torch.tensor(
            np.repeat(self.dexhand.relative_rotation[None], length, axis=0).astype(np.float32),
            device=self.device,
        )

        # Dummy object fields (hand-only, stage 1)
        dummy_obj_traj = torch.eye(4, device=self.device).unsqueeze(0).expand(length, -1, -1).contiguous()

        data = {
            "data_path": npy_path,
            "obj_id": "none",
            "obj_mesh_path": "",
            "obj_verts": torch.zeros((1000, 3), device=self.device),
            "obj_urdf_path": "",
            "obj_trajectory": dummy_obj_traj,
            "scene_objs": [],
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "mano_joints": mano_joints,
        }

        if has_precomputed_vel:
            data["_precomputed_wrist_velocity"] = torch.tensor(wrist_velocity, device=self.device)
            data["_precomputed_wrist_angular_velocity"] = torch.tensor(wrist_angular_velocity, device=self.device)
            data["_precomputed_mano_joints_velocity"] = {
                k: torch.tensor(v, device=self.device) for k, v in mano_joints_velocity.items()
            }

        self.process_data(data, idx, data["obj_verts"])

        # Retargeted data (falls back to zeros if no file exists)
        opt_dir = f"data/retargeting/handphuma/mano2{str(self.dexhand)}"
        opt_path = os.path.join(opt_dir, f"{seq_info['dataset']}_{seq_info['filename'].replace('/', '_')}.pkl")
        self.load_retargeted_data(data, opt_path)

        return data

    def process_data(self, data, idx, rs_verts_obj):
        """Override: skip chamfer/object, use target_fps for time_delta."""
        time_delta = 1.0 / self.target_fps

        # Coordinate transforms
        data["obj_trajectory"] = self.mujoco2gym_transf @ data["obj_trajectory"]
        data["wrist_pos"] = (self.mujoco2gym_transf[:3, :3] @ data["wrist_pos"].T).T + self.mujoco2gym_transf[:3, 3]
        data["wrist_rot"] = rotmat_to_aa(self.mujoco2gym_transf[:3, :3] @ data["wrist_rot"])
        for k in data["mano_joints"].keys():
            data["mano_joints"][k] = (
                self.mujoco2gym_transf[:3, :3] @ data["mano_joints"][k].T
            ).T + self.mujoco2gym_transf[:3, 3]

        # Tips distance: large values (no object contact)
        data["tips_distance"] = torch.ones(
            (data["wrist_pos"].shape[0], 5), device=self.device, dtype=torch.float32
        )

        # Velocities (hand only)
        data["obj_velocity"] = torch.zeros_like(data["obj_trajectory"][:, :3, 3])
        data["obj_angular_velocity"] = torch.zeros_like(data["obj_trajectory"][:, :3, 3])

        if "_precomputed_wrist_velocity" in data:
            # Use pre-computed velocities: just apply coordinate rotation
            R = self.mujoco2gym_transf[:3, :3]  # (3, 3) tensor
            data["wrist_velocity"] = (R @ data["_precomputed_wrist_velocity"].T).T
            data["wrist_angular_velocity"] = (R @ data["_precomputed_wrist_angular_velocity"].T).T
            data["mano_joints_velocity"] = {}
            for k in data["mano_joints"].keys():
                data["mano_joints_velocity"][k] = (R @ data["_precomputed_mano_joints_velocity"][k].T).T
            # Clean up temporary keys
            del data["_precomputed_wrist_velocity"]
            del data["_precomputed_wrist_angular_velocity"]
            del data["_precomputed_mano_joints_velocity"]
        else:
            # Fallback: compute per-chunk (old behavior, for .npy files without velocities)
            data["wrist_velocity"] = self.compute_velocity(
                data["wrist_pos"][:, None], time_delta, guassian_filter=True
            ).squeeze(1)
            data["wrist_angular_velocity"] = self.compute_angular_velocity(
                aa_to_rotmat(data["wrist_rot"][:, None]), time_delta, guassian_filter=True
            ).squeeze(1)
            data["mano_joints_velocity"] = {}
            for k in data["mano_joints"].keys():
                data["mano_joints_velocity"][k] = self.compute_velocity(
                    data["mano_joints"][k], time_delta, guassian_filter=True
                )

        # Truncate
        if len(data["obj_trajectory"]) > self.max_seq_len:
            cprint(
                f"WARN: {self.data_pathes[idx]} is too long: {len(data['obj_trajectory'])}, cut to {self.max_seq_len}",
                "yellow",
            )
            data["obj_trajectory"] = data["obj_trajectory"][: self.max_seq_len]
            data["obj_velocity"] = data["obj_velocity"][: self.max_seq_len]
            data["obj_angular_velocity"] = data["obj_angular_velocity"][: self.max_seq_len]
            data["wrist_pos"] = data["wrist_pos"][: self.max_seq_len]
            data["wrist_rot"] = data["wrist_rot"][: self.max_seq_len]
            for k in data["mano_joints"].keys():
                data["mano_joints"][k] = data["mano_joints"][k][: self.max_seq_len]
            data["wrist_velocity"] = data["wrist_velocity"][: self.max_seq_len]
            data["wrist_angular_velocity"] = data["wrist_angular_velocity"][: self.max_seq_len]
            for k in data["mano_joints_velocity"].keys():
                data["mano_joints_velocity"][k] = data["mano_joints_velocity"][k][: self.max_seq_len]
            data["tips_distance"] = data["tips_distance"][: self.max_seq_len]

    def load_retargeted_data(self, data, retargeted_data_path):
        """Override to use target_fps for velocity computation."""
        time_delta = 1.0 / self.target_fps

        if not os.path.exists(retargeted_data_path):
            if self.verbose and not getattr(self, '_retarget_warned', False):
                cprint(f"\nWARNING: Retargeted data not found (e.g. {retargeted_data_path}).", "red")
                cprint(f"WARNING: This may lead to a slower transfer process or even failure to converge.", "red")
                cprint(f"WARNING: It is recommended to first execute the retargeting code to obtain initial values.\n", "red")
                self._retarget_warned = True
            data.update(
                {
                    "opt_wrist_pos": data["wrist_pos"],
                    "opt_wrist_rot": data["wrist_rot"],
                    "opt_dof_pos": torch.zeros(
                        [data["wrist_pos"].shape[0], self.dexhand.n_dofs], device=self.device
                    ),
                }
            )
        else:
            import pickle

            opt_params = pickle.load(open(retargeted_data_path, "rb"))
            data.update(
                {
                    "opt_wrist_pos": torch.tensor(opt_params["opt_wrist_pos"], device=self.device),
                    "opt_wrist_rot": torch.tensor(opt_params["opt_wrist_rot"], device=self.device),
                    "opt_dof_pos": torch.tensor(opt_params["opt_dof_pos"], device=self.device),
                }
            )

        if not os.path.exists(retargeted_data_path):
            # No retargeting (artimano): opt velocities = hand velocities
            data["opt_wrist_velocity"] = data["wrist_velocity"].clone()
            data["opt_wrist_angular_velocity"] = data["wrist_angular_velocity"].clone()
            data["opt_dof_velocity"] = self.compute_dof_velocity(
                data["opt_dof_pos"], time_delta, guassian_filter=True
            )
        else:
            # Retargeted data from .pkl: compute velocities from retargeted positions
            data["opt_wrist_velocity"] = self.compute_velocity(
                data["opt_wrist_pos"][:, None], time_delta, guassian_filter=True
            ).squeeze(1)
            data["opt_wrist_angular_velocity"] = self.compute_angular_velocity(
                aa_to_rotmat(data["opt_wrist_rot"][:, None]), time_delta, guassian_filter=True
            ).squeeze(1)
            data["opt_dof_velocity"] = self.compute_dof_velocity(
                data["opt_dof_pos"], time_delta, guassian_filter=True
            )

        if len(data["opt_wrist_pos"]) > self.max_seq_len:
            data["opt_wrist_pos"] = data["opt_wrist_pos"][: self.max_seq_len]
            data["opt_wrist_rot"] = data["opt_wrist_rot"][: self.max_seq_len]
            data["opt_wrist_velocity"] = data["opt_wrist_velocity"][: self.max_seq_len]
            data["opt_wrist_angular_velocity"] = data["opt_wrist_angular_velocity"][: self.max_seq_len]
            data["opt_dof_pos"] = data["opt_dof_pos"][: self.max_seq_len]
            data["opt_dof_velocity"] = data["opt_dof_velocity"][: self.max_seq_len]

        assert len(data["opt_wrist_pos"]) == len(data["obj_trajectory"])
