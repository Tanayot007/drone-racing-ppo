import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stable_baselines3 import PPO
from racing_env import RacingEnv
from gates import GATES

MODEL_PATH  = "./models/racing_ppo_best"
N_EPISODES  = 100

print("Loading model...")
model = PPO.load(MODEL_PATH, device="cpu")
print("Model loaded ✔\n")

env = RacingEnv(num_drones=1, gui=False, random_spawn=False)

# ── Track results across all episodes ────────────────────────────
all_gates   = []
all_rewards = []
all_steps   = []

best_gates     = -1
best_reward    = -np.inf
best_positions = None

print(f"Running {N_EPISODES} episodes...")
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

    print(f"  Episode {ep+1:3d}/100  |  gates: {gates_passed}/11  |  "
          f"reward: {total_reward:8.1f}  |  steps: {step}", flush=True)

env.close()

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("  EVALUATION SUMMARY (100 episodes)")
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
        bar = "█" * int(count / 2)
        print(f"    Gate {g:2d}/11 : {count:3d} episodes  {bar}")
print("=" * 52)

# ── Compute speed for coloring ────────────────────────────────────
speeds = np.linalg.norm(np.diff(best_positions, axis=0), axis=1) * 240
speed_colors = plt.cm.RdYlGn(np.clip(speeds / 5.0, 0, 1))

# ══════════════════════════════════════════════════════════════════
# Plot 1 — 2D top-down trajectory (XY)
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
    rect   = patches.Rectangle(
        (gate[0] - 0.75, gate[1] - 0.75), 1.5, 1.5,
        fill=False, edgecolor=color, linewidth=2
    )
    ax.add_patch(rect)
    ax.text(gate[0], gate[1], str(i + 1),
            ha='center', va='center', fontsize=9,
            color=color, fontweight='bold')

ax.plot(best_positions[0, 0], best_positions[0, 1], 'g^', markersize=10, label='Start', zorder=5)
ax.set_xlim(-5, 100)
ax.set_ylim(-35, 5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'Best Episode (2D) — Gates: {best_gates}/11 | Reward: {best_reward:.1f} | Mean gates (100 eps): {np.mean(all_gates):.2f}')
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('trajectory_best_2d.png', dpi=150, bbox_inches='tight')
print("Saved → trajectory_best_2d.png ✔")

# ══════════════════════════════════════════════════════════════════
# Plot 2 — 3D trajectory (XYZ)
# ══════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(16, 9))
ax3d = fig2.add_subplot(111, projection='3d')

# Draw trajectory with speed coloring
for i in range(len(best_positions) - 1):
    ax3d.plot(
        best_positions[i:i+2, 0],
        best_positions[i:i+2, 1],
        best_positions[i:i+2, 2],
        color=speed_colors[i], linewidth=1.5
    )

# Draw gates as squares in 3D
GATE_HALF = 0.75
for i, gate in enumerate(GATES):
    passed = i < best_gates
    color  = '#1976D2' if passed else '#9E9E9E'
    gx, gy, gz = gate

    # Gate square corners (in YZ plane — gates face X direction)
    corners = np.array([
        [gx, gy - GATE_HALF, gz - GATE_HALF],
        [gx, gy + GATE_HALF, gz - GATE_HALF],
        [gx, gy + GATE_HALF, gz + GATE_HALF],
        [gx, gy - GATE_HALF, gz + GATE_HALF],
        [gx, gy - GATE_HALF, gz - GATE_HALF],
    ])
    ax3d.plot(corners[:, 0], corners[:, 1], corners[:, 2],
              color=color, linewidth=2)
    ax3d.text(gx, gy, gz + GATE_HALF + 0.2, str(i + 1),
              ha='center', fontsize=8, color=color, fontweight='bold')

# Start marker
ax3d.scatter(*best_positions[0], color='green', s=100, marker='^',
             zorder=5, label='Start')

# Colorbar proxy
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0, 5))
sm.set_array([])
plt.colorbar(sm, ax=ax3d, label='Speed (m/s)', shrink=0.5, pad=0.1)

ax3d.set_xlabel('X (m)')
ax3d.set_ylabel('Y (m)')
ax3d.set_zlabel('Z / Height (m)')
ax3d.set_title(
    f'Best Episode (3D) — Gates: {best_gates}/11 | '
    f'Reward: {best_reward:.1f} | Mean gates: {np.mean(all_gates):.2f}'
)
ax3d.legend(fontsize=9)
ax3d.view_init(elev=25, azim=-60)   # nice viewing angle
plt.tight_layout()
plt.savefig('trajectory_best_3d.png', dpi=150, bbox_inches='tight')
print("Saved → trajectory_best_3d.png ✔")

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
