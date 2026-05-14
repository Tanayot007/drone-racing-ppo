import numpy as np

GATES = np.array([
    [ 2,  -1,  1.0],  # Gate 1  ← same height as spawn
    [ 5,  -3,  1.0],  # Gate 2
    [ 9,  -6,  1.0],  # Gate 3
    [14, -10,  1.0],  # Gate 4
    [20, -13,  1.0],  # Gate 5
    [27, -16,  1.0],  # Gate 6
    [35, -18,  1.0],  # Gate 7
    [44, -19,  1.0],  # Gate 8
    [54, -18,  1.0],  # Gate 9
    [64, -14,  1.0],  # Gate 10
    [75, -10,  1.0],  # Gate 11
])

GATE_SIZE = 1.5

if __name__ == "__main__":
    print(f"Track: {len(GATES)} gates defined")
    print(f"Gate size: {GATE_SIZE}m")
