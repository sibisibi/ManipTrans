"""Interactive 3D wandb media during training, following robotic_grounding's pose viewer."""

from __future__ import annotations

import base64
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import trimesh
import wandb

from maniptrans_lab.tasks.bih.mdp import dexhand_artimano as HAND
from maniptrans_lab.tasks.bih.mdp import jit_math as JM
from maniptrans_lab.utils.interactive_viewer.viewer_api import create_html, make_embedded_robot

GREEN = (0.20, 0.72, 0.31)


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _xyzw(quat_wxyz):
    return _np(quat_wxyz)[..., [1, 2, 3, 0]]


def _mesh_to_stl_data_uri(mesh_path: Path) -> str:
    """Any mesh to a binary STL data URI because the template renders only STL from data URIs."""
    m = trimesh.load(mesh_path, force="mesh")
    verts = np.asarray(m.vertices, dtype="<f4")
    faces = np.asarray(m.faces, dtype=int)
    assert len(verts) and len(faces), f"no geometry parsed from {mesh_path}"
    blob = bytearray(struct.pack("<80xI", len(faces)))
    zero_normal = struct.pack("<3f", 0.0, 0.0, 0.0)
    for a, b, c in faces:
        blob += zero_normal
        blob += verts[a].tobytes() + verts[b].tobytes() + verts[c].tobytes()
        blob += b"\x00\x00"
    return "data:model/stl;base64," + base64.b64encode(bytes(blob)).decode("ascii") + "#ext=.stl"


def inline_urdf_meshes(urdf_path: Path, scale: float | None = None, green: bool = False) -> str:
    """Embed a URDF's visual meshes so the page renders with no filesystem behind it."""
    root = ET.fromstring(urdf_path.read_text(encoding="utf-8").lstrip())
    for link in root.iter("link"):
        for col in list(link.findall("collision")):
            link.remove(col)
    for mesh in root.iter("mesh"):
        f = (urdf_path.parent / mesh.get("filename")).resolve()
        ext = f.suffix.lower()
        if ext == ".stl":
            b64 = base64.b64encode(f.read_bytes()).decode()
            mesh.set("filename", f"data:model/stl;base64,{b64}#ext=.stl")
        else:
            mesh.set("filename", _mesh_to_stl_data_uri(f))
        if scale is not None:
            mesh.set("scale", f"{scale} {scale} {scale}")
    if green:
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "material":
                    parent.remove(child)
        for visual in root.iter("visual"):
            mat = ET.SubElement(visual, "material")
            mat.set("name", "goal_green")
            ET.SubElement(mat, "color").set("rgba", "0.20 0.72 0.31 1")
    return ET.tostring(root, encoding="unicode")


def _table_urdf(size, pos) -> str:
    return (
        '<?xml version="1.0"?>\n<robot name="table">\n  <link name="l">\n'
        f'    <visual><origin xyz="{pos[0]} {pos[1]} {pos[2]}"/>'
        f'<geometry><box size="{size[0]} {size[1]} {size[2]}"/></geometry>'
        '<material name="m"><color rgba="0.35 0.33 0.30 1"/></material></visual>\n'
        "  </link>\n</robot>\n"
    )


class BiHPoseViewer:
    def __init__(self, env, *, log_dir, hand_urdf_dir, obj_scale, table_size, table_pos,
                 capture_len: int = 300, interval: int = 20000, key: str = "interactive_viewer"):
        self.env = env
        self.capture_len = capture_len
        self.interval = interval
        self.key = key
        self.out_dir = Path(log_dir) / "interactive_viewer"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self._open = False
        self._f: dict[str, list] = {}
        self.env_id = 0

        self.cmd = env.command_manager.get_term("tracking")
        self.jnames = list(HAND.DOF_NAMES)
        self.dt = float(env.step_dt)

        hand_urdf_dir = Path(hand_urdf_dir)
        self.hand_urdf = {s: inline_urdf_meshes(hand_urdf_dir / f"{s}_mano.urdf") for s in ("rh", "lh")}
        self.goal_hand_urdf = {
            s: inline_urdf_meshes(hand_urdf_dir / f"{s}_mano.urdf", green=True) for s in ("rh", "lh")
        }
        self.obj_urdf = {}
        self.goal_obj_urdf = {}
        self.goal_q = {}
        self.goal_base = {}
        self.goal_obj = {}
        for s in ("rh", "lh"):
            demo = self.cmd.demo[s]
            p = Path(demo["obj_urdf_path"]).resolve()
            self.obj_urdf[s] = inline_urdf_meshes(p, scale=obj_scale[s])
            self.goal_obj_urdf[s] = inline_urdf_meshes(p, scale=obj_scale[s], green=True)
            self.goal_q[s] = _np(demo["opt_dof_pos"])
            self.goal_base[s] = np.concatenate(
                [_np(demo["opt_wrist_pos"]), _xyzw(JM.aa_to_quat(demo["opt_wrist_rot"]))], axis=-1
            )
            tr = demo["obj_trajectory"]
            self.goal_obj[s] = np.concatenate(
                [_np(tr[:, :3, 3]), _xyzw(JM.rotmat_to_quat(tr[:, :3, :3]))], axis=-1
            )
        self.table_urdf = _table_urdf(table_size, table_pos)

    def on_step(self, step: int) -> None:
        if not self._open and step % self.interval == 0:
            self._open, self._f = True, {}
            self.env_id = int(np.random.randint(self.env.num_envs))
        if self._open:
            self._sample()
            if len(self._f["rh_q"]) >= self.capture_len:
                self._finalize(step)

    def _sample(self) -> None:
        e = self.env_id
        origin = _np(self.env.scene.env_origins[e])
        f = int(self.cmd.frame_buf[e].item())
        for s in ("rh", "lh"):
            robot = self.cmd.robots[s]
            q = _np(robot.data.joint_pos[e][self.cmd.gym_from_lab[s]])
            self._f.setdefault(f"{s}_q", []).append(q.copy())
            root = robot.data.root_state_w[e]
            self._f.setdefault(f"{s}_base", []).append(
                np.concatenate([_np(root[:3]) - origin, _xyzw(root[3:7])])
            )
            self._f.setdefault(f"{s}_goal_q", []).append(self.goal_q[s][f].copy())
            self._f.setdefault(f"{s}_goal_base", []).append(self.goal_base[s][f].copy())
            oroot = self.cmd.objects[s].data.root_state_w[e]
            self._f.setdefault(f"{s}_obj", []).append(
                np.concatenate([_np(oroot[:3]) - origin, _xyzw(oroot[3:7])])
            )
            self._f.setdefault(f"{s}_goal_obj", []).append(self.goal_obj[s][f].copy())

    def _finalize(self, step: int) -> None:
        self._open = False
        arr = {k: np.asarray(v) for k, v in self._f.items()}

        robots = [
            make_embedded_robot(name="rh_hand", urdf_text=self.hand_urdf["rh"], animated=True),
            make_embedded_robot(name="lh_hand", urdf_text=self.hand_urdf["lh"], animated=True),
            make_embedded_robot(name="rh_goal_hand", urdf_text=self.goal_hand_urdf["rh"],
                                animated=True, color_override=GREEN),
            make_embedded_robot(name="lh_goal_hand", urdf_text=self.goal_hand_urdf["lh"],
                                animated=True, color_override=GREEN),
            make_embedded_robot(name="table", urdf_text=self.table_urdf),
        ]
        n = len(arr["rh_q"])
        object_poses = {"table": np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], (n, 1))}
        for s in ("rh", "lh"):
            robots.append(make_embedded_robot(name=f"{s}_obj", urdf_text=self.obj_urdf[s]))
            robots.append(make_embedded_robot(name=f"{s}_goal_obj", urdf_text=self.goal_obj_urdf[s],
                                              color_override=GREEN))
            object_poses[f"{s}_obj"] = arr[f"{s}_obj"]
            object_poses[f"{s}_goal_obj"] = arr[f"{s}_goal_obj"]

        extra = {
            "lh_hand": {"joint_names": self.jnames,
                        "positions": arr["lh_q"], "base_poses": arr["lh_base"]},
            "rh_goal_hand": {"joint_names": self.jnames,
                             "positions": arr["rh_goal_q"], "base_poses": arr["rh_goal_base"]},
            "lh_goal_hand": {"joint_names": self.jnames,
                             "positions": arr["lh_goal_q"], "base_poses": arr["lh_goal_base"]},
        }

        html = create_html(
            joint_names=self.jnames,
            robot_joint_positions=arr["rh_q"],
            robots=robots,
            object_poses=object_poses,
            robot_base_poses=arr["rh_base"],
            extra_animated=extra,
            dt=self.dt,
            robot_name="rh_hand",
        )
        path = self.out_dir / f"capture_{self.n:04d}_step{step}.html"
        path.write_text(html, encoding="utf-8")
        print(f"[pose_viewer] {path.name}  {len(html) / 1e6:.2f} MB  env {self.env_id}", flush=True)

        if wandb.run is not None:
            # committing here would claim a step the trainer has not reached
            wandb.log({self.key: wandb.Html(html), f"{self.key}/env_id": self.env_id}, commit=False)
            wandb.run.summary["interactive_viewer_latest"] = self.n

        self.n += 1
        self._f = {}
