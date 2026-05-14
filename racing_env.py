import numpy as np
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gates import GATES, GATE_SIZE

BASE_OBS_PER_DRONE = 492
EXTRA_OBS = 6 + 9
TOTAL_OBS = BASE_OBS_PER_DRONE + EXTRA_OBS

class RacingEnv(BaseRLAviary):

    def __init__(self, num_drones=1, gui=False, random_spawn=False):
        self.GATES = GATES
        self.num_gates = len(GATES)
        self.gate_index = np.zeros(num_drones, dtype=int)
        self.prev_dist = np.zeros(num_drones)
        self._num_drones = num_drones
        self.random_spawn = random_spawn  # kept for API compatibility, but default is False
        self._start_gate = 0
        init_xyz = np.array([[0.0, 0.0, 1.0]])
        init_rpy = np.array([[0.0, 0.0, 0.0]])
        super().__init__(
            drone_model=DroneModel.RACE,
            num_drones=num_drones,
            physics=Physics.PYB,
            gui=gui,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            initial_xyzs=init_xyz,
            initial_rpys=init_rpy,
        )

    def _observationSpace(self):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TOTAL_OBS,),
            dtype=np.float32
        )

    def _computeObs(self):
        base_obs = super()._computeObs()
        drone_obs = base_obs[0]
        gate_obs = []
        for offset in range(2):
            idx = min(self.gate_index[0] + offset, self.num_gates - 1)
            rel = self.GATES[idx] - self.pos[0]
            gate_obs.extend(rel)
        opp_obs = [0.0] * 9
        full_obs = np.concatenate([drone_obs, gate_obs, opp_obs])
        return full_obs.astype(np.float32)

    def _check_gate_passed(self, drone_idx):
        gate_idx = self.gate_index[drone_idx]
        if gate_idx >= self.num_gates:
            return False
        gate_pos = self.GATES[gate_idx]
        drone_pos = self.pos[drone_idx]
        dx = abs(drone_pos[0] - gate_pos[0])
        dy = abs(drone_pos[1] - gate_pos[1])
        return dx < GATE_SIZE and dy < GATE_SIZE

    def _computeReward(self):
        if self.gate_index[0] >= self.num_gates:
            return 0.0

        gate_pos = self.GATES[self.gate_index[0]]
        drone_pos = self.pos[0]
        dist = np.linalg.norm(gate_pos - drone_pos)

        # Progress reward: reward for getting closer to the next gate
        reward = (self.prev_dist[0] - dist) * 10

        # Gate passing bonus
        if self._check_gate_passed(0):
            reward += 100
            self.gate_index[0] += 1

        # Crash penalty scaled by gates remaining
        if self.pos[0][2] < 0.1:
            gates_remaining = self.num_gates - self.gate_index[0]
            reward -= 50 * gates_remaining

        self.prev_dist[0] = dist
        return float(reward)

    def _computeTerminated(self):
        if self.step_counter > 50 and self.pos[0][2] < 0.1:
            return True
        if self.gate_index[0] >= self.num_gates:
            return True
        return False

    def _computeTruncated(self):
        if self.step_counter > 2000:
            return True
        return False

    def _computeInfo(self):
        return {"gate": int(self.gate_index[0])}

    def reset(self, seed=None, options=None):
        # Always spawn at Gate 1 (index 0) — professor's recommendation
        # This teaches the drone to chain gates from the very start
        self._start_gate = 0
        spawn_gate = self.GATES[self._start_gate]

        # Small noise around Gate 1 spawn to improve robustness
        noise = np.random.randn(3) * 0.3
        noise[2] = abs(noise[2])  # keep above ground

        self.INIT_XYZS = np.array([[
            spawn_gate[0] - 1.0 + noise[0],
            spawn_gate[1] + noise[1],
            max(0.5, spawn_gate[2] + noise[2])
        ]])

        self.gate_index = np.array([self._start_gate])
        obs, info = super().reset(seed=seed, options=options)
        self.prev_dist[0] = np.linalg.norm(
            self.GATES[self._start_gate] - self.pos[0]
        )
        return obs, info
