"""Reward and termination terms reading the command term's kernel output."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _cmd(env):
    return env.command_manager.get_term("tracking")


def imitation_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    rew, _, _, _, rdict, _ = _cmd(env).run_kernel()
    # expose the per-term dict and success like the gym env's info
    env.extras["reward_dict"] = rdict
    # divided by the step time because the reward manager multiplies by it
    return rew / env.step_dt


def tracking_done(env: "ManagerBasedRLEnv") -> torch.Tensor:
    cmd = _cmd(env)
    rew, reset, success, failure, rdict, error = cmd.run_kernel()
    env.extras["success_buf"] = success
    env.extras["failure_buf"] = failure
    env.extras["error_buf"] = error
    return reset
