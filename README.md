# sim-throughput-benchmark

**同一任务，不同仿真器，多少 env-steps/s？**
*How many env-steps per second does each simulator actually deliver?*

这个仓库只做一件事：把「仿真跑得快不快」变成一个**可复现的数字**，而不是一句"很快"。

```
harness（统一接口）
   ├── backends/dummy.py     ← 纯 Python 假后端，用来验证 harness 自身
   ├── backends/gazebo.py    ← 待补：ros2 + Gazebo（CPU 基线）
   ├── backends/isaac_lab.py ← 待补：Isaac Lab（GPU / PhysX 5）
   └── backends/mjx.py       ← 待补：MuJoCo MJX（JAX，GPU 批量）
```

## 快速开始

```bash
python bench/harness.py --backend dummy --envs 64,256,1024,4096 --steps 200 --repeat 3
```

输出（示例，dummy 后端的数字不代表任何真实引擎）：

```
| backend | envs | steps | wall_s | env_steps_per_s | steps_per_s |
| dummy   |   64 |   200 |  1.204 |          10,631 |      166.1 |
```

## 指标口径（写论文/汇报都按这个）

| 指标 | 定义 | 为什么要它 |
| --- | --- | --- |
| `env_steps_per_s` | `envs × steps / wall_s` | 唯一的吞吐主指标 |
| `steps_per_s` | `steps / wall_s` | 看单步调度开销 |
| `gpu_util` | 采样窗口内 `nvidia-smi` 均值 | 判断是否吃满 GPU |
| `vram_gb` | 峰值显存 | 决定能上多少 env |
| `$ / 1k env-steps` | 云价格 ÷ 吞吐 | 老板最爱问的那个数 |

## 实验设计（要填的表）

| backend | envs | steps/s | env-steps/s | GPU util | VRAM | 拐点 |
| --- | --- | --- | --- | --- | --- | --- |
| gazebo (CPU) | 1 | _待填_ | _待填_ | n/a | n/a | — |
| isaac_lab | 64 / 256 / 1024 / 4096 | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| mjx | 64 / 256 / 1024 / 4096 | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |

**要能回答的三个问题**：
1. 拐点出现在多少 env？为什么（Python 调度 / CPU-GPU 同步 / 显存带宽）？
2. 同样吞吐下，哪个引擎的显存更省？
3. 从 CPU 换到 GPU，`$/1k env-steps` 降了多少？

## 状态

- [x] 统一 backend 接口 + dummy 后端 + 扫描 harness
- [ ] Gazebo（CPU）后端
- [ ] Isaac Lab 后端
- [ ] MuJoCo MJX 后端
- [ ] 4090 上的实测曲线与瓶颈分析

> 数据口径：README 的数字只从 `results/*.csv` 生成，不手填。

## License

MIT
