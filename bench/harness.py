"""Sweep env counts across simulator backends and report throughput.

    python bench/harness.py --backend dummy --envs 64,256,1024 --steps 200 --repeat 3

Each backend implements the same three-line contract (see backends/__init__.py),
so adding Isaac Lab or MuJoCo MJX later is a new file, not a new harness.

Metric definitions
------------------
env_steps_per_s = envs * steps / wall_s   -- the headline number
steps_per_s     = steps / wall_s          -- exposes per-step scheduling cost
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.backends import load  # noqa: E402


def parse_int_list(text: str) -> list[int]:
    values = [int(x) for x in text.replace(' ', '').split(',') if x]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f'bad env list: {text!r}')
    return values


def run_one(backend_name: str, envs: int, steps: int, repeat: int, warmup: int, seed: int) -> dict:
    cls = load(backend_name)
    walls = []
    for run in range(repeat):
        backend = cls(envs=envs, seed=seed + run)
        if warmup:
            backend.step(warmup)          # JIT / shader / allocator warm-up
        start = time.perf_counter()
        backend.step(steps)
        walls.append(time.perf_counter() - start)
        backend.close()

    best = min(walls)
    median = statistics.median(walls)
    return {
        'backend': backend_name,
        'envs': envs,
        'steps': steps,
        'repeat': repeat,
        'wall_s_best': round(best, 4),
        'wall_s_median': round(median, 4),
        'spread_pct': round((median - best) / best * 100.0, 1) if best else 0.0,
        'env_steps_per_s': round(envs * steps / best),
        'steps_per_s': round(steps / best, 1),
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


def to_markdown(rows: list[dict]) -> str:
    head = (f"| {'backend':<10} | {'envs':>6} | {'steps':>6} | {'wall_s':>8} | "
            f"{'env-steps/s':>12} | {'steps/s':>9} | {'spread':>7} |")
    sep = f"|{'-' * 12}|{'-' * 8}|{'-' * 8}|{'-' * 10}|{'-' * 14}|{'-' * 11}|{'-' * 9}|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['backend']:<10} | {r['envs']:>6} | {r['steps']:>6} | {r['wall_s_best']:>8.3f} | "
            f"{r['env_steps_per_s']:>12,} | {r['steps_per_s']:>9.1f} | {r['spread_pct']:>6.1f}% |"
        )
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend', default='dummy')
    ap.add_argument('--envs', type=parse_int_list, default=[64, 256, 1024])
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', default='results/latest.csv')
    args = ap.parse_args()

    rows = []
    for envs in args.envs:
        row = run_one(args.backend, envs, args.steps, args.repeat, args.warmup, args.seed)
        rows.append(row)
        print(f"{row['backend']} envs={envs}: {row['env_steps_per_s']:,} env-steps/s "
              f"(wall {row['wall_s_best']:.3f}s, spread {row['spread_pct']}%)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table = to_markdown(rows)
    Path(out).with_suffix('.md').write_text(table + '\n', encoding='utf-8')
    print('\n' + table)
    print(f'\nwrote {out} and {out.with_suffix(".md")}')
    return 0


if __name__ == '__main__':
    import time  # noqa: E402  (imported late so --help stays fast)
    sys.exit(main())
