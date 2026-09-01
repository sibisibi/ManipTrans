"""Gym's deterministic warmup taking object friction from 6 to 2 and gravity from 0 to full over the first 1920 environment steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import torch

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

_SCHEDULE_STEPS = 1920
_OG_FRICTION = 2.0
_BUCKET_LO, _BUCKET_HI, _NUM_BUCKETS = 1.0, 6.0, 250
_GRAVITY_Z = -9.8

_done = False


def _bucketed(val: float) -> float:
    width = (_BUCKET_HI - _BUCKET_LO) / _NUM_BUCKETS
    idx = int((val - _BUCKET_LO) / width)
    idx = max(0, min(idx, _NUM_BUCKETS - 1))
    return _BUCKET_LO + width * idx


def friction_gravity_warmup(env: "ManagerBasedEnv", env_ids: torch.Tensor | None):
    global _done
    step = env.common_step_counter
    s = max(0.0, 1.0 - step / _SCHEDULE_STEPS)
    if s == 0.0:
        if _done:
            return
        _done = True

    friction = _bucketed(_OG_FRICTION * (3.0 * s + (1.0 - s)))
    for name in ("manip_obj_rh", "manip_obj_lh"):
        obj = env.scene[name]
        mats = obj.root_physx_view.get_material_properties()
        mats[..., 0] = friction
        mats[..., 1] = friction
        obj.root_physx_view.set_material_properties(mats, torch.arange(env.num_envs))

    gravity = carb.Float3(0.0, 0.0, _GRAVITY_Z * (1.0 - s))
    sim_utils.SimulationContext.instance().physics_sim_view.set_gravity(gravity)
