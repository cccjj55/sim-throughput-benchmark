"""Throughput-accuracy frontier: how much error buys how much speed.

The question this answers is the one every simulation engineer gets asked:

    "You say it's fast -- fast at what accuracy?"

Method
------
1. Integrate the same model with a very small timestep to get a *reference*
   trajectory (dt_ref = 1e-4). This is expensive and treated as ground truth.
2. Integrate with dt in {1e-3, 2e-3, 5e-3, 1e-2} over the same control input.
3. Compare the two on a common time grid, report RMSE for each state, plus the
   wall-clock cost of each run.
4. Emit a table: dt | steps/s | RMSE. The Pareto front is "for this error
   budget, this is the fastest timestep you can afford".

    python bench/accuracy.py --duration 2.0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.backends.mujoco_cpu import CART_POLE_XML  # noqa: E402

DT_REF = 1e-4
DT_SWEEP = (0.001, 0.002, 0.005, 0.01)


def rollout(mj, model, duration: float, sample_dt: float) -> tuple[np.ndarray, int]:
    """Integrate for `duration` seconds, sampling state every sample_dt."""
    data = mj.MjData(model)
    steps = int(round(duration / model.opt.timestep))
    out = []
    next_sample = 0.0
    for i in range(steps):
        t = i * model.opt.timestep
        data.ctrl[0] = 1.5 * np.sin(4.0 * t)          # deterministic, non-trivial input
        mj.mj_step(model, data)
        if t + model.opt.timestep >= next_sample:
            out.append(np.array([data.qpos[0], data.qpos[1], data.qvel[0], data.qvel[1]]))
            next_sample += sample_dt
    return np.asarray(out), steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--duration', type=float, default=2.0)
    ap.add_argument('--sample-dt', type=float, default=0.01)
    ap.add_argument('--out', default='results/accuracy.csv')
    args = ap.parse_args()

    import mujoco

    model_ref = mujoco.MjModel.from_xml_string(CART_POLE_XML)
    model_ref.opt.timestep = DT_REF

    t0 = time.perf_counter()
    ref, ref_steps = rollout(mujoco, model_ref, args.duration, args.sample_dt)
    ref_wall = time.perf_counter() - t0
    print(f'reference: dt={DT_REF:g}, {ref_steps} steps, {ref_wall:.2f}s')

    rows = []
    for dt in DT_SWEEP:
        model = mujoco.MjModel.from_xml_string(CART_POLE_XML)
        model.opt.timestep = dt
        t0 = time.perf_counter()
        traj, steps = rollout(mujoco, model, args.duration, args.sample_dt)
        wall = time.perf_counter() - t0

        n = min(len(ref), len(traj))
        diff = traj[:n] - ref[:n]
        rmse = np.sqrt((diff ** 2).mean(axis=0))    # [slider(m), hinge(rad), v0, v1]
        rows.append({
            'dt': dt,
            'steps': steps,
            'wall_s': round(wall, 4),
            'steps_per_s': round(steps / wall) if wall else 0,
            'rmse_slider_mm': round(float(rmse[0]) * 1e3, 3),
            'rmse_hinge_rad': round(float(rmse[1]), 4),
            'speedup_vs_ref': round(ref_wall / wall, 1) if wall else 0,
        })
        print(f'dt={dt:g}: {rows[-1]["steps_per_s"]:>8,} steps/s  '
              f'RMSE slider {rows[-1]["rmse_slider_mm"]:.3f} mm  '
              f'hinge {rows[-1]["rmse_hinge_rad"]:.4f} rad')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ['| dt | steps/s | wall (s) | RMSE slider (mm) | RMSE hinge (rad) | speedup vs ref |',
          '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        md.append(f"| {r['dt']:g} | {r['steps_per_s']:,} | {r['wall_s']:.3f} | "
                  f"{r['rmse_slider_mm']:.3f} | {r['rmse_hinge_rad']:.4f} | "
                  f"{r['speedup_vs_ref']}× |")
    out.with_suffix('.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(md))
    print(f'\nwrote {out} and {out.with_suffix(".md")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
