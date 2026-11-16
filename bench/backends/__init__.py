"""Backend registry.

A backend is a class with this contract::

    class Backend:
        def __init__(self, envs: int, seed: int = 0, **kwargs): ...
        def step(self, n: int = 1) -> None: ...   # advance n simulator steps
        def close(self) -> None: ...

That is deliberately minimal: the harness must not know whether the physics
runs on CPU threads, PhysX, or JAX. Anything else (GPU utilisation, VRAM)
belongs in the adapter, sampled by the adapter itself.
"""

from __future__ import annotations

import importlib
from typing import Type

AVAILABLE = ('dummy', 'gazebo', 'isaac_lab', 'mjx')


def load(name: str) -> Type:
    if name not in AVAILABLE:
        raise SystemExit(f'unknown backend {name!r}; available: {", ".join(AVAILABLE)}')
    try:
        module = importlib.import_module(f'bench.backends.{name}')
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f'backend {name!r} is declared but not implemented yet ({exc}). '
            f'Start with --backend dummy.'
        ) from exc
    if not hasattr(module, 'Backend'):
        raise SystemExit(f'{name}.py must define a `Backend` class')
    return module.Backend
