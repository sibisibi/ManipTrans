"""IsaacLab training entry for the bimanual ManipTrans port, reusing ManipTrans's rl_games stack unchanged."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--experiment", type=str, default="lab_bih_retgt")
parser.add_argument("--test", action="store_true")
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--early_stop_epochs", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--data_idx", type=str, required=True)
parser.add_argument("--demo_path", type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import os
import sys
from datetime import datetime

import numpy
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

# stand-in modules so lib imports resolve without isaacgym, bps_torch, pytorch3d, cv2 in this environment
import types


class _HandInfo:
    n_dofs = 22
    n_bodies = 28


class _HandInfoFactory:
    @classmethod
    def create_hand(cls, name, side, *args, **kwargs):
        assert name == "artimano", name
        return _HandInfo()


_shims = {}
for _name in (
    "maniptrans_envs",
    "maniptrans_envs.lib",
    "maniptrans_envs.lib.envs",
    "maniptrans_envs.lib.envs.dexhands",
    "maniptrans_envs.lib.envs.dexhands.factory",
):
    _shims[_name] = types.ModuleType(_name)
    sys.modules[_name] = _shims[_name]
_shims["maniptrans_envs.lib"].TASK_MAP = {}
_shims["maniptrans_envs.lib.envs.dexhands.factory"].DexHandFactory = _HandInfoFactory

# rl_games checkpoints carry these numpy globals, which torch.load blocks by default
torch.serialization.add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype, numpy.dtypes.Float32DType])

import lib  # noqa: F401  registers OmegaConf resolvers
from lib.utils.reformat import omegaconf_to_dict
from lib.utils.rlgames_utils import ComplexObsRLGPUEnv, RLGPUAlgoObserver, MultiObserver
from lib.rl.runner import Runner
from lib.rl.network_builder import DictObsBuilder
from lib.rl.models import ModelA2CContinuousLogStd
from lib.rl.network_builder_residual_bih import ResBiHDictObsBuilder
from lib.rl.res_models import ModelA2CContinuousLogStdResBiH
from rl_games.algos_torch.model_builder import register_network, register_model
from rl_games.common import env_configurations, vecenv

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.envs.manager_based_env as _mbe
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg
from maniptrans_lab.tasks.bih.collision_filter import author_hand_table_filter
from maniptrans_lab.rl_adapter import GymSurface
from maniptrans_lab.tasks.bih.env_cfg import BiHManipEnvCfg

class _SingleCamera(Camera):
    def reset(self, env_ids=None):
        super().reset(None)  # one camera serves every env so per-env reset indexing does not apply


def preprocess_train_config(cfg, config_dict):
    """Add common configuration parameters to the rl_games train config."""

    train_cfg = config_dict["params"]["config"]

    train_cfg["device"] = cfg.rl_device

    train_cfg["population_based_training"] = False
    train_cfg["pbt_idx"] = None

    train_cfg["full_experiment_name"] = cfg.get("full_experiment_name")

    return config_dict


OVERRIDES = [
    "task=ResDexHand",
    "dexhand=artimano",
    "side=BiH",
    f"num_envs={args.num_envs}",
    f"test={str(args.test).lower()}",
    "randomStateInit=" + ("false" if args.test else "true"),
    "rh_base_model_checkpoint=null",
    "lh_base_model_checkpoint=null",
    f"dataIndices=[{args.data_idx}]",
    f"early_stop_epochs={args.early_stop_epochs}",
    "actionsMovingAverage=0.4",
    "learning_rate=2e-4",
    "usePIDControl=True",
    "useRetargetedBase=True",
    f"experiment={args.experiment}",
    f"seed={args.seed}",
    f"checkpoint={args.checkpoint}",
]


def main():
    with initialize_config_dir(version_base="1.1", config_dir=os.path.join(REPO, "main", "cfg")):
        cfg = compose(config_name="config", overrides=OVERRIDES)

    register_model("my_continuous_a2c_logstd", ModelA2CContinuousLogStd)
    register_network("dict_obs_actor_critic", DictObsBuilder)
    register_network("res_bih_dict_obs_actor_critic", ResBiHDictObsBuilder)
    register_model("res_bih_my_continuous_a2c_logstd", ModelA2CContinuousLogStdResBiH)

    torch.manual_seed(cfg.seed)
    env_cfg = BiHManipEnvCfg()
    usd = os.path.join(REPO, "maniptrans_lab", "assets", "usd")
    env_cfg.scene.dexhand_r.spawn.usd_path = os.path.join(usd, "rh_mano.usd")
    env_cfg.scene.dexhand_l.spawn.usd_path = os.path.join(usd, "lh_mano.usd")
    env_cfg.scene.manip_obj_rh.spawn.usd_path = os.path.join(usd, "obj_rh", "obj_rh.usd")
    env_cfg.scene.manip_obj_lh.spawn.usd_path = os.path.join(usd, "obj_lh", "obj_lh.usd")
    env_cfg.commands.tracking.demo_path = args.demo_path
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    env_cfg.seed = cfg.seed
    env_cfg.commands.tracking.training = not args.test
    env_cfg.commands.tracking.random_state_init = not args.test
    if os.environ.get("MT_VIDEO_DIR"):
        # the offscreen renderer has no default lighting and the camera needs --enable_cameras on the launcher
        env_cfg.scene.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=2000.0),
        )
        env_cfg.scene.camera = CameraCfg(
            class_type=_SingleCamera,
            prim_path="/World/envs/env_0/eval_cam",
            update_period=0.0,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(),
        )

    # collision filter must be authored before PhysX parses the stage
    class _FilteredScene(_mbe.InteractiveScene):
        def __init__(self, scene_cfg):
            super().__init__(scene_cfg)
            author_hand_table_filter(self.stage, scene_cfg.num_envs)

    _mbe.InteractiveScene = _FilteredScene

    if args.test:
        experiment_dir = os.path.join("dumps", "test_" + args.experiment)
    else:
        experiment_dir = os.path.join(
            "runs", args.experiment + "__" + "{date:%m-%d-%H-%M-%S}".format(date=datetime.now())
        )
    os.makedirs(experiment_dir, exist_ok=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    adapter = GymSurface(env, clip_obs=cfg.task.env.clipObservations)

    env_configurations.register("rlgpu", {"vecenv_type": "RLGPU", "env_creator": lambda **kw: adapter})
    vecenv.register("RLGPU", lambda config_name, num_actors: ComplexObsRLGPUEnv(config_name))

    rlg_config_dict = omegaconf_to_dict(cfg.rl_train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)
    # the lab entry always trains without imitator checkpoints and leaves gym's network layout untouched
    rlg_config_dict["params"]["network"]["base_model"]["drop_when_none"] = True

    runner = Runner(MultiObserver([RLGPUAlgoObserver()]))
    runner.load(rlg_config_dict)
    runner.reset()

    if not args.test:
        runner.params["config"]["full_experiment_name"] = experiment_dir.replace("runs/", "")
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    runner.run(
        {
            "train": not args.test,
            "play": args.test,
            "checkpoint": args.checkpoint if args.checkpoint else None,
            "from_ckpt_epoch": False,
            "sigma": None,
            "save_rollouts": {"save_rollouts": False, "rollout_saving_fpath": None},
        }
    )


if __name__ == "__main__":
    main()
    app.close()
