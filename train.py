import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from racing_env import RacingEnv

# ══════════════════════════════════════════════════════════════════
# GateCallback
#   • Tracks gate counts every rollout
#   • Saves the BEST model whenever mean-gate-per-rollout improves
#     (same idea as EvalCallback saving by best mean reward in HW4)
#   • Records data for a 3-panel convergence plot
# ══════════════════════════════════════════════════════════════════
class GateCallback(BaseCallback):
    def __init__(self, save_path="./models/racing_ppo_best", verbose=0):
        super().__init__(verbose)
        self.save_path       = save_path
        self.best_gates      = 0        # highest single gate ever seen
        self.best_mean_gate  = 0.0      # best rollout-mean gate → triggers save

        self.gate_counts = [0] * 12

        # Convergence logs (one point per rollout)
        self.timestep_log       = []
        self.best_gate_log      = []
        self.mean_gate_log      = []
        self.episode_reward_log = []
        self.kl_log             = []    # approx KL — useful for checking target_kl

    # ── per step: catch new individual-gate records ───────────────
    def _on_step(self):
        for info in self.locals.get("infos", []):
            gates = info.get("gate", 0)
            if gates > 0:
                self.gate_counts[min(gates, 11)] += 1
            if gates > self.best_gates:
                self.best_gates = gates
                print(f"\n  ★ NEW BEST: Gate {gates}/11 at step {self.num_timesteps:,}!")
        return True

    # ── per rollout: log + conditionally save best model ─────────
    def _on_rollout_end(self):
        total     = sum(self.gate_counts)
        mean_gate = (
            sum(i * self.gate_counts[i] for i in range(12)) / total
            if total > 0 else 0.0
        )

        # Save whenever this rollout's mean gate beats the running best
        if mean_gate > self.best_mean_gate:
            self.best_mean_gate = mean_gate
            self.model.save(self.save_path)
            print(f"  ✔ Best model saved  (mean gate {mean_gate:.3f})")

        # Mean episode reward from VecMonitor buffer
        ep_rewards = [ep["r"] for ep in self.model.ep_info_buffer if "r" in ep]
        mean_rew   = float(np.mean(ep_rewards)) if ep_rewards else 0.0

        # Approx KL logged internally by SB3's PPO
        kl = self.model.logger.name_to_value.get("train/approx_kl", float("nan"))

        # Print rollout summary
        print(f"\n{'─'*54}")
        print(f"  Step {self.num_timesteps:>13,}  │  best gate : {self.best_gates}/11")
        print(f"  Mean gate : {mean_gate:.3f}        │  best mean : {self.best_mean_gate:.3f}")
        print(f"  Ep reward : {mean_rew:8.1f}        │  approx KL : {kl:.5f}")
        for i in range(1, 12):
            if self.gate_counts[i] > 0:
                print(f"    Gate {i:2d}/11 : {self.gate_counts[i]:6d} episodes")
        print(f"{'─'*54}\n")

        # Store for convergence plot
        self.timestep_log.append(self.num_timesteps)
        self.best_gate_log.append(self.best_gates)
        self.mean_gate_log.append(mean_gate)
        self.episode_reward_log.append(mean_rew)
        self.kl_log.append(kl if not np.isnan(kl) else 0.0)

        self.gate_counts = [0] * 12

    # ── 3-panel convergence figure ────────────────────────────────
    def save_convergence_plot(self, path="convergence.png"):
        if not self.timestep_log:
            print("No data to plot.")
            return

        steps = np.array(self.timestep_log) / 1_000_000  # → millions

        fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
        fig.suptitle(
            "PPO Training Convergence — Drone Gate Racing\n"
            f"Best gate: {self.best_gates}/11   |   "
            f"Best mean gate: {self.best_mean_gate:.2f}",
            fontsize=13, fontweight="bold",
        )

        # ── Panel 1: gate progress ──────────────────────────────────
        ax = axes[0]
        ax.plot(steps, self.best_gate_log, color="#1565C0", lw=2,
                label="Best gate (ever)", zorder=3)
        ax.plot(steps, self.mean_gate_log, color="#FF6F00", lw=1.5,
                ls="--", label="Mean gate per rollout", zorder=2)
        ax.fill_between(steps, self.mean_gate_log, alpha=0.12, color="#FF6F00")
        ax.axhline(11, color="#2E7D32", ls=":", lw=1.2, label="All 11 gates ✓")
        ax.set_ylabel("Gate Reached", fontsize=10)
        ax.set_ylim(-0.3, 12.5)
        ax.set_yticks(range(0, 12))
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.25)

        # Annotate each new best gate with an arrow
        prev = 0
        for s, bg in zip(steps, self.best_gate_log):
            if bg > prev:
                ax.annotate(
                    f"G{bg}", xy=(s, bg), xytext=(s, bg + 0.6),
                    fontsize=7.5, color="#1565C0", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=0.8),
                )
                prev = bg

        # ── Panel 2: episode reward ─────────────────────────────────
        ax = axes[1]
        rew    = np.array(self.episode_reward_log)
        win    = max(1, len(rew) // 20)
        smooth = np.convolve(rew, np.ones(win) / win, mode="valid")
        ax.plot(steps, rew, color="#BDBDBD", lw=0.7, alpha=0.5, label="Raw")
        ax.plot(steps[win - 1:], smooth, color="#C62828", lw=2,
                label=f"Smoothed (window={win})")
        ax.set_ylabel("Mean Episode Reward", fontsize=10)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.25)

        # ── Panel 3: approx KL divergence ──────────────────────────
        ax = axes[2]
        kl = np.array(self.kl_log)
        ax.plot(steps, kl, color="#6A1B9A", lw=1.2, label="Approx KL")
        ax.axhline(0.03, color="#E65100", ls="--", lw=1.2,
                   label="target_kl = 0.03")
        ax.set_ylabel("Approx KL Divergence", fontsize=10)
        ax.set_xlabel("Timesteps (millions)", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Convergence plot saved → {path}")


# ══════════════════════════════════════════════════════════════════
# Training setup
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import torch
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    N_ENVS    = 300
    TIMESTEPS = 120_000_000
    BEST_PATH = "./models/racing_ppo_best"   # saved when mean gate improves ← use this for eval
    LAST_PATH = "./models/racing_ppo_last"   # always saved at end

    env = make_vec_env(
        lambda: RacingEnv(num_drones=1, gui=False, random_spawn=False),
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,           # true multiprocess on cluster
    )
    env = VecMonitor(env)

    # ── PPO hyperparameters ────────────────────────────────────────────
    #
    #  The key additions from HW4 that make training more stable:
    #
    #  gae_lambda = 0.95   → smoother advantage estimates; the HW4 default
    #                         and SB3's recommended value
    #
    #  target_kl  = 0.03   → PPO stops the update early if policy changes
    #                         too fast; prevents catastrophic forgetting
    #                         when the drone is just learning to chain gates
    #
    #  ent_coef   = 0.005  → slightly less entropy than before (0.01); once
    #                         the drone can reach G2-G3 we want exploitation
    #
    #  net_arch   = [256,256,128]  → wider front layers suit the 507-dim obs;
    #                                narrowing tail focuses the output head
    #
    #  batch_size = 512    → right-sized for 300 envs on L40S; large batches
    #                         give stable gradient estimates at cluster scale
    # ──────────────────────────────────────────────────────────────────
    model = PPO(
        policy          = "MlpPolicy",
        env             = env,
        verbose         = 1,
        device          = "cuda",
        tensorboard_log = "./logs/",
        # ── core ──────────────────────
        learning_rate   = 3e-4,
        n_steps         = 2048,
        batch_size      = 512,         # large batch for 300 envs on L40S
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,        # ← HW4 addition
        clip_range      = 0.2,
        ent_coef        = 0.005,       # ← tuned down from 0.01
        target_kl       = 0.03,        # ← HW4 addition (most important)
        # ── network ───────────────────
        policy_kwargs   = dict(
            net_arch      = [256, 256, 128],   # wider network for cluster scale
            activation_fn = torch.nn.ReLU,
        ),
    )

    callback = GateCallback(save_path=BEST_PATH)
    model.learn(total_timesteps=TIMESTEPS, callback=callback)

    # Always save the last checkpoint too
    model.save(LAST_PATH)
    env.close()

    # Convergence plot
    callback.save_convergence_plot("convergence.png")

    print(f"\n{'═'*54}")
    print(f"  Training complete")
    print(f"  Best model (by mean gate) → {BEST_PATH}.zip")
    print(f"  Last model                → {LAST_PATH}.zip")
    print(f"  Best gate ever reached    → {callback.best_gates}/11")
    print(f"  Best mean gate            → {callback.best_mean_gate:.3f}")
    print(f"{'═'*54}")
