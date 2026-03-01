"""Behavioral Cloning agent for ManipTrans.

Freezes an RL teacher and trains a student via MSE on action means.
Student optionally sees noisy observations (target trajectory only)
while teacher always sees clean observations.

Usage: set rl_train.params.algo.name=bc in hydra config, plus
  +rl_train.params.config.teacher_checkpoint=<path>
  +rl_train.params.config.bc_lr=1e-4
  +rl_train.params.config.obs_noise_sigma=0.0
"""

import torch
from torch import optim

from lib.rl.agent import PPOAgent, load_checkpoint


class BCAgent(PPOAgent):
    def __init__(self, base_name, params):
        PPOAgent.__init__(self, base_name, params)

        # --- Load frozen teacher ---
        teacher_path = self.config["teacher_checkpoint"]
        self._load_teacher(teacher_path, params)

        # --- BC config ---
        self.bc_lr = float(self.config.get("bc_lr", 1e-4))
        self.obs_noise_sigma = float(self.config.get("obs_noise_sigma", 0.0))

        # --- Replace optimizer: actor params only (skip critic/value) ---
        skip_keywords = ["critic_mlp", "value"]
        actor_params = [
            p for n, p in self.model.named_parameters()
            if not any(kw in n for kw in skip_keywords)
        ]
        self.optimizer = optim.Adam(actor_params, lr=self.bc_lr, eps=1e-08)

        print(f"[BC] teacher={teacher_path}")
        print(f"[BC] bc_lr={self.bc_lr}, obs_noise_sigma={self.obs_noise_sigma}")
        n_actor = sum(p.numel() for p in actor_params)
        n_total = sum(p.numel() for p in self.model.parameters())
        print(f"[BC] Optimizing {n_actor}/{n_total} params (actor only)")

    # ------------------------------------------------------------------
    # Teacher loading
    # ------------------------------------------------------------------
    def _load_teacher(self, path, params):
        build_config = {
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "num_seqs": self.num_actors * self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
            "normalize_input_excluded_keys": self.normalize_input_excluded_keys,
            "use_pid_control": self.config["use_pid_control"],
            **params,
        }
        self.teacher_model = self.network.build(build_config)
        self.teacher_model.to(self.ppo_device)

        checkpoint = load_checkpoint(path)
        self.teacher_model.load_state_dict(checkpoint["model"])

        for p in self.teacher_model.parameters():
            p.requires_grad = False
        self.teacher_model.eval()
        print(f"[BC] Teacher loaded and frozen ({sum(p.numel() for p in self.teacher_model.parameters())} params)")

    # ------------------------------------------------------------------
    # Checkpoint restore: model weights only (skip optimizer state)
    # ------------------------------------------------------------------
    def restore(self, fn, set_epoch=True):
        checkpoint = load_checkpoint(fn)
        self.model.load_state_dict(checkpoint["model"])
        self.set_stats_weights(checkpoint)
        print(f"[BC] Loaded student weights from {fn} (optimizer not restored)")

    # ------------------------------------------------------------------
    # Rollout: run teacher on stored obs after standard play_steps
    # ------------------------------------------------------------------
    def play_steps(self):
        batch_dict = super().play_steps()

        # Run teacher in chunks to avoid OOM (full batch is num_envs * horizon)
        obses = batch_dict["obses"]
        if isinstance(obses, dict):
            total = next(iter(obses.values())).shape[0]
        else:
            total = obses.shape[0]

        chunk_size = self.num_actors  # num_envs — one horizon step at a time
        all_mus = []

        with torch.no_grad():
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                if isinstance(obses, dict):
                    chunk_obs = {k: v[start:end] for k, v in obses.items()}
                else:
                    chunk_obs = obses[start:end]

                teacher_input = {
                    "is_train": False,
                    "prev_actions": None,
                    "obs": chunk_obs,
                    "rnn_states": None,
                }
                teacher_res = self.teacher_model(teacher_input)
                all_mus.append(teacher_res["mus"])

            batch_dict["teacher_mus"] = torch.cat(all_mus, dim=0)

        return batch_dict

    # ------------------------------------------------------------------
    # Dataset: pass teacher_mus through to minibatches
    # ------------------------------------------------------------------
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["teacher_mus"] = batch_dict["teacher_mus"]

    # ------------------------------------------------------------------
    # Observation noise injection (extensible)
    # ------------------------------------------------------------------
    def _apply_obs_noise(self, obs_dict):
        """Apply noise to target observations for the student.

        Currently: i.i.d. Gaussian on the 'target' key.
        Extend this method to add:
          - AR(1) temporally correlated noise (P1 from plan.md)
          - Per-joint non-uniform variance (P2)
          - Global wrist translation noise (P3)
          - Stale observation / freeze (P4)
          - Slow wrist drift (P5)
          - Scale perturbation (P6)
        """
        if self.obs_noise_sigma <= 0:
            return obs_dict

        noisy = {k: v for k, v in obs_dict.items()}
        noisy["target"] = obs_dict["target"] + torch.randn_like(obs_dict["target"]) * self.obs_noise_sigma
        return noisy

    # ------------------------------------------------------------------
    # BC loss: MSE(student_mu, teacher_mu)
    # ------------------------------------------------------------------
    def calc_gradients(self, input_dict):
        obs_batch = input_dict["obs"]
        teacher_mus = input_dict["teacher_mus"]
        actions_batch = input_dict["actions"]

        student_obs = self._apply_obs_noise(obs_batch)

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": student_obs,
        }

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            student_mu = res_dict["mus"]
            bc_loss = torch.mean((student_mu - teacher_mus) ** 2)

        for param in self.model.parameters():
            param.grad = None

        self.scaler.scale(bc_loss).backward()
        self.trancate_gradients_and_step()

        with torch.no_grad():
            kl_dist = torch.zeros(1, device=self.ppo_device)

        self.train_result = (
            bc_loss,                                          # a_loss slot (logged as losses/a_loss)
            torch.zeros(1, device=self.ppo_device),           # c_loss
            torch.zeros(1, device=self.ppo_device),           # entropy
            kl_dist,                                          # kl
            self.last_lr,                                     # lr
            1.0,                                              # lr_mul
            student_mu.detach(),                              # cmu
            res_dict["sigmas"].detach(),                      # csigma
            torch.zeros(1, device=self.ppo_device),           # b_loss
        )

    def train_actor_critic(self, input_dict):
        self.calc_gradients(input_dict)
        return self.train_result
