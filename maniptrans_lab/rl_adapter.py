"""VecTask-surface adapter over ManagerBasedRLEnv for ManipTrans's rl_games stack."""

from __future__ import annotations

import os
import subprocess

import gym.spaces as spaces
import torch


class GymSurface:
    def __init__(self, env, clip_obs: float = 5.0, pose_viewer=None):
        self.env = env
        self.clip_obs = clip_obs
        n_act = 2 * (6 + 22)  # residual action count from gym's numActions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n_act,))
        dims = {"proprioception": 158, "privileged": 118, "target": 900}
        self.observation_space = spaces.Dict(
            {k: spaces.Box(low=-clip_obs, high=clip_obs, shape=(d,)) for k, d in dims.items()}
        )
        self.num_states = 0
        dev = env.device
        n = env.num_envs
        self._zeros_timeout = torch.zeros(n, dtype=torch.bool, device=dev)
        self._ones_error_mask = torch.ones(n, device=dev)
        self._last_obs = None
        self._last_done_idx = torch.zeros(0, dtype=torch.long, device=dev)
        self.success_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.failure_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.error_buf = torch.zeros(n, dtype=torch.bool, device=dev)

        # first-episode outcomes, reported when every environment has finished
        self._eval_mode = not env.cfg.commands.tracking.training
        self._eval_outcome = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._eval_reported = False

        # capture environment 0 through the optional scene camera into an mp4
        self._video_dir = os.environ.get("MT_VIDEO_DIR", "")
        self._video_frames = []
        self._video_done = False
        self._cam_posed = False

        # periodic interactive 3D wandb media counted in policy steps
        self._pose_viewer = pose_viewer
        self._policy_steps = 0

    def get_number_of_agents(self):
        return 1

    def _clip(self, obs):
        return {k: torch.clamp(v, -self.clip_obs, self.clip_obs) for k, v in obs.items()}

    def reset(self):
        obs, _ = self.env.reset()
        self._last_obs = self._clip(obs)
        return self._last_obs

    def reset_done(self):
        return self._last_obs, self._last_done_idx

    def step(self, actions):
        obs, rew, term, trunc, extras = self.env.step(actions)
        done = (term | trunc).long()
        self._last_obs = self._clip(obs)
        self._last_done_idx = done.nonzero(as_tuple=False).flatten()
        self.success_buf = extras["success_buf"]
        self.failure_buf = extras["failure_buf"]
        self.error_buf = extras["error_buf"]
        if self._eval_mode and not self._eval_reported:
            ended = (term | trunc) & (self._eval_outcome < 0)
            self._eval_outcome[ended & self.success_buf] = 1
            self._eval_outcome[ended & ~self.success_buf] = 0
            if (self._eval_outcome >= 0).all():
                n = self._eval_outcome.numel()
                sr = (self._eval_outcome == 1).float().mean().item() * 100
                print(f"[eval] first-episode success from frame 0: {sr:.2f}% over {n} envs", flush=True)
                self._eval_reported = True

        if self._video_dir and not self._video_done:
            self._capture_frame(term | trunc)

        if self._pose_viewer is not None:
            self._pose_viewer.on_step(self._policy_steps)
        self._policy_steps += 1

        infos = {
            "time_outs": self._zeros_timeout,  # gym's episode timeout never fires on this clip because the demo ends first
            "error_masks": self._ones_error_mask,  # gym sends all-ones
            "reward_dict": extras.get("reward_dict", {}),
        }
        return self._last_obs, rew, done, infos

    def _capture_frame(self, done):
        cam = self.env.scene.sensors["camera"]
        if not self._cam_posed:
            origin = self.env.scene.env_origins[0]
            eye = (origin + torch.tensor([0.7, 0.9, 1.0], device=origin.device)).unsqueeze(0)
            target = (origin + torch.tensor([0.0, 0.0, 0.5], device=origin.device)).unsqueeze(0)
            cam.set_world_poses_from_view(eyes=eye, targets=target, env_ids=torch.tensor([0], device=origin.device))
            self._cam_posed = True
            return  # first frame renders with the old pose
        rgb = cam.data.output["rgb"][0]
        self._video_frames.append(rgb.detach().cpu().numpy())
        if bool(done[0]) and len(self._video_frames) > 30:
            self._write_video()
            self._video_done = True

    def _write_video(self):
        os.makedirs(self._video_dir, exist_ok=True)
        out = os.path.join(self._video_dir, "lab-policy-rollout.mp4")
        self._encode(self._video_frames, out)
        print(f"[video] {len(self._video_frames)} frames -> {out}", flush=True)

    def _encode(self, frames, out):
        h, w = frames[0].shape[:2]
        pix = "rgba" if frames[0].shape[-1] == 4 else "rgb24"
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", pix, "-s", f"{w}x{h}", "-r", "60",
             "-i", "-", "-pix_fmt", "yuv420p", out],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
