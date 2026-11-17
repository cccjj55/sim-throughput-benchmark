"""MuJoCo CPU backend: the baseline every GPU number gets compared against.

A laptop CPU run is the honest floor. If a GPU engine is only 3x faster than
this, the overhead is not worth it -- that is exactly the kind of claim the
harness exists to test.

The model is a cart-pole written inline so the backend has no asset
dependencies and runs identically on any machine.
"""

from __future__ import annotations

import numpy as np

CART_POLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.002" integrator="RK4"/>
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" damping="0.10"/>
      <geom type="box" size="0.10 0.05 0.05" mass="1.0"/>
      <body name="pole" pos="0 0 0.05">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.02"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


class Backend:
    name = 'mujoco_cpu'

    def __init__(self, envs: int = 64, seed: int = 0, timestep: float | None = None) -> None:
        import mujoco  # imported lazily so the harness works without mujoco installed

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(CART_POLE_XML)
        if timestep is not None:
            self.model.opt.timestep = float(timestep)
        self.envs = int(envs)
        rng = np.random.default_rng(seed)
        self.datas = []
        for i in range(self.envs):
            data = mujoco.MjData(self.model)
            # spread initial states so the batch is not trivially cache-friendly
            data.qpos[0] = rng.uniform(-0.2, 0.2)
            data.qpos[1] = rng.uniform(-0.1, 0.1)
            mujoco.mj_forward(self.model, data)
            self.datas.append(data)

    def step(self, n: int = 1) -> None:
        mj, model = self._mj, self.model
        for _ in range(n):
            for data in self.datas:
                data.ctrl[0] = 1.0
                mj.mj_step(model, data)

    def close(self) -> None:
        self.datas = []
