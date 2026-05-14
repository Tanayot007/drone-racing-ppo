import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from stable_baselines3 import PPO
from racing_env import RacingEnv
from gates import GATES
import glob
import os

MODEL_PATH  = "./models/racing_ppo_best"
LOG_DIR     = "./logs/"
N_EPISODES  = 1000

# ══════════════════════════════════════════════════════════════════
# Part 1 — Convergence plot from TensorBoard logs
# ══════════════════════════════════════════════════════════════════
def plot_convergence(log_dir, save_path="convergence.png"):
    try:
        from tensorboard.backend.event_processing import event_accumulator

        # Find event file
        files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents*"), recursive=True)
        if not files:
            print(f"No TensorBoard event files found in {log_dir}")
            return

        print(f"Loading TensorBoard logs from {files[0]}...")
        ea = event_accumulator.EventAccumulator(files[0])
        ea.Reload()

        available = ea.Tags().get('scalars', [])
        print(f"Available tags: {available}")

        fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
        fig.suptitle("PPO Training Convergence", fontsize=13, fontweight='bold')

        # ── Panel 1: Episode Reward ───────────────────────────────
        ax = axes[0]
        if 'rollout/ep_rew_mean' in available:
            events = ea.Scalars('rollout/ep_rew_mean')
            steps  = np.array([e.step  for e in events]) / 1_000_000
            values = np.array([e.value for e in events])
            win    = max(1, len(values) // 20)
            smooth = np.convolve(values, np.ones(win)/win, mode='valid')
            ax.plot(steps, values, color='#BDBDBD', lw=0.7, alpha=0.5, label='Raw')
            ax.plot(steps[win-1:], smooth, color='#C62828', lw=2, label=f'Smoothed (w={win})')
            ax.set_ylabel("Mean Episode Reward", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

        # ── Panel 2: Episode Length ───────────────────────────────
        ax = axes[1]
        if 'rollout/ep_len_mean' in available:
            events = ea.Scalars('rollout/ep_len_mean')
            steps  = np.array([e.step  for e in events]) / 1_000_000
            values = np.array([e.value for e in events])
            win    = max(1, len(values) // 20)
            smooth = np.convolve(values, np.ones(win)/win, mode='valid')
            ax.plot(steps, values, color='#BDBDBD', lw=0.7, alpha=0.5, label='Raw')
            ax.plot(steps[win-1:], smooth, color='#1565C0', lw=2, label=f'Smoothed (w={win})')
            ax.set_ylabel("Mean Episode Length (steps)", fontsize=10)
            ax.set_xlabel("Timesteps (millions)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Convergence plot saved → {save_path} ✔")

    except ImportError:
        print("tensorboard not installed. Run: pip install tensorboard")
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")


# ══════════════════════════════════════════════════════════════════
# Part 2 — Evaluation: 1000 episodes from Gate 1
# ══════════════════════════════════════════════════════════════════
print("=" * 52)
print("  CONVERGENCE PLOT")
print("=" * 52)
plot_convergence(LOG_DIR, save_path="convergence.png")

print("\n" + "=" * 52)
print("  LOADING MODEL")
print("=" * 52)
print("Loading model...")
model = PPO.load(MODEL_PATH, device="cpu")
print("Model loaded ✔")

env = RacingEnv(num_drones=1, gui=False, random_spawn=False)

all_gates   = []
all_rewards = []
all_steps   = []
best_gates     = -1
best_reward    = -np.inf
best_positions = None

print(f"\nRunning {N_EPISODES} episodes from Gate 1...")
print("-" * 52)

for ep in range(N_EPISODES):
    env.INIT_XYZS = np.array([[0.0, 0.0, 1.0]])
    env.gate_index = np.array([0])
    obs, info = env.reset()

    positions    = []
    total_reward = 0
    step         = 0

    while True:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        positions.append(env.pos[0].copy())
        total_reward += reward
        step += 1
        if terminated or truncated or step > 5000:
            break

    gates_passed = int(env.gate_index[0])
    all_gates.append(gates_passed)
    all_rewards.append(total_reward)
    all_steps.append(step)

    if (gates_passed > best_gates) or (gates_passed == best_gates and total_reward > best_reward):
        best_gates     = gates_passed
        best_reward    = total_reward
        best_positions = np.array(positions)

    if (ep + 1) % 100 == 0:
        print(f"  Episode {ep+1:4d}/{N_EPISODES}  |  gates: {gates_passed}/11  |  "
              f"reward: {total_reward:8.1f}  |  steps: {step}", flush=True)

env.close()

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 52)
print(f"  EVALUATION SUMMARY ({N_EPISODES} episodes)")
print("=" * 52)
print(f"  Best gates reached  : {best_gates}/11")
print(f"  Mean gates reached  : {np.mean(all_gates):.2f}")
print(f"  Std gates           : {np.std(all_gates):.2f}")
print(f"  Best reward         : {best_reward:.2f}")
print(f"  Mean reward         : {np.mean(all_rewards):.2f}")
print(f"  Mean steps          : {np.mean(all_steps):.1f}")
print("-" * 52)
print("  Gate distribution:")
for g in range(12):
    count = all_gates.count(g)
    if count > 0:
        bar = "█" * int(count / 20)
        print(f"    Gate {g:2d}/11 : {count:4d} episodes  {bar}")
print("=" * 52)

# ── Compute speed coloring ────────────────────────────────────────
speeds      = np.linalg.norm(np.diff(best_positions, axis=0), axis=1) * 240
speed_colors = plt.cm.RdYlGn(np.clip(speeds / 5.0, 0, 1))

# ══════════════════════════════════════════════════════════════════
# Plot 1 — 2D top-down trajectory
# ══════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(14, 5))
scatter = ax.scatter(
    best_positions[:-1, 0], best_positions[:-1, 1],
    c=speeds, cmap='RdYlGn', s=3, vmin=0, vmax=5
)
plt.colorbar(scatter, label='Speed (m/s)')
for i, gate in enumerate(GATES):
    passed = i < best_gates
    color  = '#1976D2' if passed else '#9E9E9E'
    rect   = patches.Rectangle((gate[0]-0.75, gate[1]-0.75), 1.5, 1.5,
                                fill=False, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(gate[0], gate[1], str(i+1), ha='center', va='center',
            fontsize=9, color=color, fontweight='bold')
ax.plot(best_positions[0,0], best_positions[0,1], 'g^', markersize=10, label='Start', zorder=5)
ax.set_xlim(-5, 100)
ax.set_ylim(-35, 5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Best Episode (2D) — Gates: {best_gates}/11 | Reward: {best_reward:.1f} | Mean gates ({N_EPISODES} eps): {np.mean(all_gates):.2f}')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('trajectory_best_2d.png', dpi=150, bbox_inches='tight')
print("Saved → trajectory_best_2d.png ✔")

# ══════════════════════════════════════════════════════════════════
# Plot 3 — Gate distribution bar chart
# ══════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(10, 4))
gate_range = range(0, max(all_gates) + 1)
counts     = [all_gates.count(g) for g in gate_range]
bars = ax3.bar(gate_range, counts, color='#1976D2', edgecolor='white', linewidth=0.5)
ax3.set_xlabel('Gates Reached')
ax3.set_ylabel('Number of Episodes')
ax3.set_title(f'Gate Distribution over {N_EPISODES} Episodes — Best: {best_gates}/11  |  Mean: {np.mean(all_gates):.2f}')
ax3.set_xticks(list(gate_range))
ax3.set_xticklabels([f'G{g}' for g in gate_range])
for bar, count in zip(bars, counts):
    if count > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontsize=9)
ax3.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('gate_distribution.png', dpi=150, bbox_inches='tight')
print("Saved → gate_distribution.png ✔")

print("\nAll plots saved! ✔")
