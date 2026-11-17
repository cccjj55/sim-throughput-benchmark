# sim-throughput-benchmark

**同一任务，不同引擎：多少 env-steps/s，以及在给定精度下最快能跑多快。**
*How many env-steps per second does each simulator deliver — and at what accuracy?*

两个只回答两个问题的脚本：

| 脚本 | 回答的问题 | 输出 |
| --- | --- | --- |
| `bench/harness.py` | 每个引擎每秒能跑多少环境步？ | `env-steps/s` / `steps/s` / `spread` |
| `bench/accuracy.py` | 时间步放大到多少，误差还能接受？ | `dt × steps/s × RMSE`（精度-速度前沿） |

```
bench/
├── harness.py                 # 统一扫描（Backend.step(n) 一行契约）
├── accuracy.py                # dt 扫描 + 相对参考解的 RMSE + Pareto 表
└── backends/
    ├── dummy.py               # ✅ 自测后端（Python 控制流开销下限）
    ├── mujoco_cpu.py          # ✅ MuJoCo CPU 基线（内联 cart-pole，无外部资产）
    ├── mjx.py                 # ⬜ MuJoCo MJX（JAX，GPU 批量）
    ├── gazebo.py              # ⬜ Gazebo（ROS 2，CPU）
    └── isaac_lab.py           # ⬜ Isaac Lab（PhysX 5，GPU）
```

## 快速开始

```bash
pip install mujoco                       # 只有 CPU 基线需要它

python bench/harness.py --backend mujoco_cpu --envs 8,32,64 --steps 200 --repeat 3
python bench/accuracy.py --duration 1.0  # dt 1e-3..1e-2，参考解 dt=1e-4
```

输出（示例格式，数字取决于你的机器）：

```
| backend    |   envs |  steps |   wall_s |   env-steps/s |   steps/s |  spread |
| mujoco_cpu |      8 |    200 |    0.008 |       xxx,xxx |   xxx,x.x |    0.1% |
| mujoco_cpu |     32 |    200 |    0.033 |       xxx,xxx |   xxx,x.x |    0.4% |
```

```
| dt   | steps/s | RMSE slider (mm) | RMSE hinge (rad) | speedup vs ref |
| 0.001| xxx,xxx |            0.xxx |           0.xxx x |           x.x× |
| 0.01 |  xx,xxx |             x.xxx |           0.xxx x |          xx.x× |
```

## 指标口径（写论文/汇报都按这个）

| 指标 | 定义 | 为什么需要它 |
| --- | --- | --- |
| `env_steps_per_s` | `envs × steps / wall_s` | 唯一能跨模型比较的吞吐口径 |
| `steps_per_s` | `steps / wall_s` | 暴露单步调度开销 |
| `spread_pct` | `(median − best) / best` | **>20% 不算测量，算命**（热降频/其他进程） |
| `RMSE (mm)` | 相对 `dt=1e-4` 参考解的状态误差 | 精度必须对着参考解，不是对着自己 |
| `speedup_vs_ref` | 参考解耗时 / 本次耗时 | 精度-速度权衡的斜率 |

## 精度-速度前沿（要填的表）

题目不是"跑得快不快"，而是**"误差 1 mm 以内最快能跑多快"**。

| dt | steps/s | RMSE slider (mm) | 满足 ≤1 mm | 备注 |
| --- | --- | --- | --- | --- |
| 0.001 | _待填_ | _待填_ | ✅/❌ | |
| 0.002 | _待填_ | _待填_ | ✅/❌ | |
| 0.005 | _待填_ | _待填_ | ✅/❌ | |
| 0.01 | _待填_ | _待填_ | ✅/❌ | |

## 状态

- [x] 统一 Backend 契约 + dummy 后端（自测）
- [x] MuJoCo CPU 基线（内联 cart-pole，无外部依赖）
- [x] 精度扫描：dt → RMSE vs 参考解 → Pareto 表
- [x] CI：装 mujoco，跑 harness + accuracy（无 GPU 也能绿）
- [ ] MJX（JAX/CUDA）后端
- [ ] Isaac Lab（PhysX 5）后端
- [ ] Gazebo（CPU）后端
- [ ] 拐点分析：env 数 × device 的瓶颈归因

> **数据口径**：README 里的数字只从 `results/*.csv` 生成，不手填；
> 每次跑都记录硬件、驱动、版本（`--out` 会写时间戳）。

## License

MIT
