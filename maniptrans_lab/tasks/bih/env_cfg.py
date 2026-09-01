"""Manager-based env cfg for the bimanual artiMANO ManipTrans port, mirroring dexhandmanip_bih.py."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from .mdp import actions as bih_actions
from .mdp import commands as bih_commands
from .mdp import events as bih_events
from .mdp import observations as bih_obs
from .mdp import rewards as bih_rewards

ROBOT_HEIGHT = 0.00214874  # maniptrans_envs/lib/envs/core/config.py

# table numbers in gym's own arithmetic from dexhandmanip_bih._create_envs
TABLE_WIDTH_OFFSET = 0.2
TABLE_SIZE = (0.8 + TABLE_WIDTH_OFFSET, 1.6, 0.03)
TABLE_POS = (-TABLE_WIDTH_OFFSET / 2, 0.0, 0.4)
TABLE_SURFACE_Z = TABLE_POS[2] + TABLE_SIZE[2] / 2
TABLE_HALF_WIDTH = 0.4

_HALF_SQRT2 = math.sqrt(0.5)
_HAND_ROT_WXYZ = (_HALF_SQRT2, 0.0, -_HALF_SQRT2, 0.0)  # gym Quat.from_euler_zyx(0, -pi/2, 0)


def _hand_cfg(side: str) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_" + ("r" if side == "rh" else "l"),
        spawn=sim_utils.UsdFileCfg(
            usd_path=MISSING,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=20.0,
                angular_damping=20.0,
                max_linear_velocity=50.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-TABLE_HALF_WIDTH, 0.0, TABLE_SURFACE_Z + ROBOT_HEIGHT),
            rot=_HAND_ROT_WXYZ,
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,  # inherit USD (500)
                damping=None,    # inherit USD (30)
            )
        },
    )


def _obj_cfg(side: str) -> RigidObjectCfg:
    # the spawn pose does not matter because reset writes demo poses before any stepping
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/manip_obj_" + side,
        spawn=sim_utils.UsdFileCfg(
            usd_path=MISSING,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005, rest_offset=0.0
            ),
        ),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dexhand_r: ArticulationCfg = _hand_cfg("rh")
    dexhand_l: ArticulationCfg = _hand_cfg("lh")
    manip_obj_rh: RigidObjectCfg = _obj_cfg("rh")
    manip_obj_lh: RigidObjectCfg = _obj_cfg("lh")

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.1, dynamic_friction=0.1, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS),
    )

    # one sensor per tip: IsaacLab filtered contact reporting is one-to-many
    contact_rh_thumb3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_r/thumb3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_rh"],
    )
    contact_rh_index3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_r/index3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_rh"],
    )
    contact_rh_middle3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_r/middle3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_rh"],
    )
    contact_rh_ring3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_r/ring3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_rh"],
    )
    contact_rh_pinky3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_r/pinky3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_rh"],
    )
    contact_lh_thumb3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_l/thumb3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_lh"],
    )
    contact_lh_index3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_l/index3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_lh"],
    )
    contact_lh_middle3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_l/middle3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_lh"],
    )
    contact_lh_ring3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_l/ring3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_lh"],
    )
    contact_lh_pinky3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/dexhand_l/pinky3",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/manip_obj_lh"],
    )


@configclass
class CommandsCfg:
    tracking = bih_commands.BiHTrackingCommandCfg(
        random_state_init=True,
        training=True,
    )


@configclass
class ActionsCfg:
    bih = bih_actions.BiHManipActionCfg(
        translation_scale=1.0,
        orientation_scale=0.1,
        act_moving_average=0.4,
    )


@configclass
class ObservationsCfg:
    @configclass
    class Proprio(ObservationGroupCfg):
        v = ObservationTermCfg(func=bih_obs.proprioception)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Privileged(ObservationGroupCfg):
        v = ObservationTermCfg(func=bih_obs.privileged)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class Target(ObservationGroupCfg):
        v = ObservationTermCfg(func=bih_obs.target)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    proprioception: Proprio = Proprio()
    privileged: Privileged = Privileged()
    target: Target = Target()


@configclass
class RewardsCfg:
    imitation = RewardTermCfg(func=bih_rewards.imitation_reward, weight=1.0)


@configclass
class TerminationsCfg:
    tracking = TerminationTermCfg(func=bih_rewards.tracking_done, time_out=False)


@configclass
class EventCfg:
    warmup_startup = EventTermCfg(func=bih_events.friction_gravity_warmup, mode="startup")
    warmup = EventTermCfg(
        func=bih_events.friction_gravity_warmup,
        mode="interval",
        interval_range_s=(32.0 / 60.0, 32.0 / 60.0),  # gym frequency 32 env steps
    )


@configclass
class BiHManipEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1200 * (1.0 / 60.0)  # gym episodeLength 1200 control steps
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -9.8)
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 8
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**21
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
