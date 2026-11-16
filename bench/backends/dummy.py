"""Dummy backend: proves the harness works, and gives a Python-side floor.

Two purposes:
1. **Self-test** - the harness must produce sane numbers before any GPU work.
2. **Overhead floor** - it measures pure Python control-flow cost per step, which
   is exactly the overhead that shows up as the "knee" when env counts grow.

The numbers here mean nothing about any physics engine. They exist to validate
the measurement path.
"""

from __future__ import annotations

import math
import random


class Backend:
    name = 'dummy'

    def __init__(self, envs: int, seed: int = 0, dims: int = 12) -> None:
        self.envs = int(envs)
        self.dims = int(dims)
        rng = random.Random(seed)
        self.state = [[rng.uniform(-0.1, 0.1) for _ in range(self.dims)]
                      for _ in range(self.envs)]

    def step(self, n: int = 1) -> None:
        envs, dims = self.envs, self.dims
        state = self.state
        for _ in range(n):
            for i in range(envs):
                row = state[i]
                # a cheap, non-trivial update so the loop is not optimised away
                for j in range(dims):
                    row[j] = math.tanh(row[j] * 0.98 + 0.01)

    def close(self) -> None:
        self.state = []
