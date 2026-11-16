"""Benchmark results schema.

Every result file is a CSV with these columns, one row per (backend, envs):

| column | type | meaning |
| --- | --- | --- |
| backend | str | dummy / gazebo / isaac_lab / mjx |
| envs | int | parallel environments |
| steps | int | simulated steps per measurement |
| repeat | int | number of measurement runs |
| wall_s_best | float | fastest run (throughput best case) |
| wall_s_median | float | median run |
| spread_pct | float | (median - best) / best |
| env_steps_per_s | int | envs * steps / wall_s_best |
| steps_per_s | float | steps / wall_s_best |
| timestamp | str | ISO-8601 UTC |

`results/latest.md` is generated from the CSV, and the README table is copied
from there. No number in the README may be typed by hand.

Report spread alongside throughput: a run with >20% spread is not a measurement,
it is a rumour (thermal throttling, other processes, cold caches).
"""

SCHEMA = (
    'backend', 'envs', 'steps', 'repeat',
    'wall_s_best', 'wall_s_median', 'spread_pct',
    'env_steps_per_s', 'steps_per_s', 'timestamp',
)
