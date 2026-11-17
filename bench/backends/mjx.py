"""MuJoCo MJX backend: the same cart-pole, batched on GPU.

Why the same model as ``mujoco_cpu``: comparing two *different* engines on two
*different* models tells you nothing. Here the model, the integrator and the
timestep are identical; only the device and the batching change.

Notes that matter for honest numbers
------------------------------------
* JIT compilation is excluded from timing (``warmup()`` runs the compiled
  function once; the harness also has a warmup argument).
* JAX dispatches asynchronously, so ``step`` blocks on the result -- otherwise
  you would be timing the queueing, not the physics.
* Blackwell GPUs (RTX 50-series, sm_120) need a recent jaxlib/CUDA (12.8+).
  If ``jax.devices()`` shows no ``CudaDevice``, run with ``--device cpu``: the
  code path is identical, only slower.
"""

from __future__ import annotations

import dataclasses

from bench.backends.mujoco_cpu import CART_POLE_XML


class Backend:
    name = 'mjx'

    def __init__(self, envs: int = 64, seed: int = 0, device: str = 'cuda',
                 timestep: float | None = None) -> None:
        try:
            import jax
            import jax.numpy as jnp
            import mujoco
            from mujoco import mjx
        except ImportError as exc:  # pragma: no cover - env dependent
            raise SystemExit(
                'mjx backend needs:  pip install mujoco mujoco-mjx "jax[cuda12]" flax\n'
                f'import error: {exc}'
            ) from exc

        self._jax, self._jnp, self._mjx = jax, jnp, mjx
        self.envs = int(envs)

        devices = jax.devices(device)
        if device == 'cuda' and not devices:
            raise SystemExit('no CUDA device visible to JAX; try --device cpu')
        self.device = devices[0]

        with jax.default_device(self.device):
            mj_model = mujoco.MjModel.from_xml_string(CART_POLE_XML)
            if timestep is not None:
                mj_model.opt.timestep = float(timestep)
            self.model = mjx.put_model(mj_model)

            single = mjx.make_data(self.model)
            batched = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (self.envs,) + x.shape), single)
            # spread the initial state so the batch is not one cache-friendly case
            qpos = batched.qpos.at[:, 0].set(jnp.linspace(-0.2, 0.2, self.envs))
            self.data = dataclasses.replace(batched, qpos=qpos)

            ctrl = jnp.full((self.envs, single.ctrl.shape[0]), 1.0)

            def body(_i, d):
                d = dataclasses.replace(d, ctrl=ctrl)
                return jax.vmap(mjx.step, in_axes=(None, 0))(self.model, d)

            @jax.jit
            def run(data, n):
                return jax.lax.fori_loop(0, n, body, data)

            self._run = run
            self.warmup()

    def warmup(self, n: int = 5) -> None:
        self.data = self._run(self.data, n)
        self.data.qpos.block_until_ready()

    def step(self, n: int = 1) -> None:
        self.data = self._run(self.data, n)
        self.data.qpos.block_until_ready()   # block: measure physics, not queueing

    def close(self) -> None:
        self.data = None
