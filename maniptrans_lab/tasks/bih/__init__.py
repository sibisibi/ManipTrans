import gymnasium as gym

from .env_cfg import BiHManipEnvCfg

gym.register(
    id="ManipTrans-BiH-Artimano-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": BiHManipEnvCfg},
)
