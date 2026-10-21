# RL-X / RCSSServerMJ / Apollo 运行环境镜像

这是一个**只包含运行环境**的 Docker 镜像：里面没有 RL-X、RCSSServerMJ、Apollo 的源码，
源码由你自己下载后挂载进容器。镜像里已经装好训练、仿真、模型转换需要的全部依赖。

| 用途 | 镜像内提供 |
| --- | --- |
| RL-X 训练 | Python 3.11.4、JAX/Flax 0.7.2、PyTorch 2.13.0（CUDA 12.9）、MuJoCo 3.15、MJX、Warp 1.18、TensorFlow 2.21、envpool |
| RCSSServerMJ | Python 3.13.16、`rcsssmj 0.2.1`、MuJoCo 3.5.0 |
| Apollo 编译/运行 | g++ 13.3、CMake 3.28、yaml-cpp 0.8、ONNX Runtime 1.22.0 |

镜像里没有 NVIDIA 驱动，也**没有 CUDA Toolkit**（无 `nvcc`、无 `/usr/local/cuda`）。
CUDA 用户态库（cudart、cuBLAS、cuDNN 9.20 等）来自 pip 的 `nvidia-*-cu12` wheel，
由镜像自己提供；主机只需要驱动。


## 文件清单

```
rlx_env_release/
├── README.md                     本文件
├── image/
│   ├── rlx_env.tar.gz.part01     镜像分卷（每卷 < 2 GiB，GitHub Release 单文件上限）
│   ├── rlx_env.tar.gz.part02
│   ├── rlx_env.tar.gz.part03
│   ├── rlx_env.tar.gz.part04
│   └── SHA256SUMS.parts          分卷校验和
└── model_conversion/
    ├── README.md                 转换链说明（输入输出契约、std_dev 判据）
    ├── convert.py                checkpoint -> PyTorch state_dict
    ├── export_onnx.py            state_dict -> Apollo 的 policy.onnx（含数值自检）
    ├── torch_policy.py           PyTorch 网络定义
    ├── pretrained_nn.pth         示例产物
    ├── pretrained_nn_meta.json
    └── policy.onnx               示例产物
```


## 主机要求

- **Ubuntu 原生 Linux**（本镜像在 Ubuntu 24.04 上构建和验证）
- NVIDIA GPU，驱动可用：`nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv`
- 驱动需要支持 CUDA 12.x（PyTorch/JAX）；Warp 1.18 自带 CUDA Toolkit 13.4，
  因此实际按 CUDA 13 的驱动算（R580 及以上）。本项目在 RTX 5060 Laptop + 驱动 595.99.02 上验证。
- Docker
- **NVIDIA Container Toolkit**（Linux 原生 Docker 必需；它不是驱动，也不是 CUDA Toolkit）

主机不需要安装 CUDA Toolkit：Dockerfile 里没有 CUDA 编译需求，Apollo 的编译在容器内完成。


### 安装 NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

确认 runtime 已注册：

```bash
docker info --format '{{json .Runtimes}}'      # 输出里应有 "nvidia"
```


## 导入镜像

分卷是同一个 `.tar.gz` 被切开的结果。先合并、校验，再导入：

```bash
cd image/
cat rlx_env.tar.gz.part* > rlx_env.tar.gz
sha256sum -c SHA256SUMS.parts        # 每个分卷都应 OK
sha256sum rlx_env.tar.gz
# 期望：98f0af292d0fbed4aa3d3f8a6ad1c5edb77b0b2fa06072275fe4ad40654ea8d2

docker load -i rlx_env.tar.gz
```

也可以不落盘直接导入（这时无法先校验）：

```bash
cat rlx_env.tar.gz.part* | docker load
```

导入后镜像名是 `rlx_env:latest`，约占 21.6 GB 磁盘。


## 获取源码

镜像不含源码，请自己克隆到主机（建议放在 `~/rlx` 下）：

```bash
git clone https://github.com/nico-bohlinger/RL-X ~/rlx/RL-X
git clone <ApolloCodebase 的仓库地址> ~/rlx/ApolloCodebase
```


## 启动容器

```bash
docker run -it --rm --gpus all --shm-size=8g \
  -v "$HOME/rlx/RL-X:/workspace/RL-X" \
  -v "$HOME/rlx/ApolloCodebase:/workspace/ApolloCodebase" \
  rlx_env
```

进入后 `rlx` 环境已激活，工作目录是 `/workspace/RL-X/experiments`，
`/workspace/RL-X` 已加入 `PYTHONPATH`，不需要 `pip install -e .`。

容器默认是 root，挂载目录里生成的 `runs/`、`build/` 会归 root。介意就加
`--user "$(id -u):$(id -g)" -e HOME=/tmp`。


## GPU 验证（分层，别跳步）

| 层 | 命令 | 通过标准 |
| --- | --- | --- |
| A 主机 | `nvidia-smi` | 型号 + 驱动 |
| B 容器 | `docker run --rm --gpus all rlx_env nvidia-smi` | 容器内输出 GPU 信息 |
| C PyTorch | `docker run --rm --gpus all rlx_env python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability(0))"` | `True` + 算力 |
| D PyTorch 计算 | `docker run --rm --gpus all rlx_env python -c "import torch; x=torch.randn(2048,2048,device='cuda'); print(float((x@x).sum()))"` | 输出数值 |
| E JAX | `docker run --rm --gpus all rlx_env python -c "import jax, jax.numpy as jnp; print(jax.devices()); print(float(jax.jit(lambda x:(x@x).sum())(jnp.ones((1024,1024)))))"` | 含 `CudaDevice` |
| F Warp | 见下 | 内核执行成功 |

Warp 的内核必须定义在 `.py` 文件里（Warp 会读取源码），不能用 `python -c`。
把下面内容存成 `warp_check.py`，再 `-v` 挂进容器运行：

```python
import numpy as np, warp as wp
wp.init()
@wp.kernel
def saxpy(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float), s: float):
    i = wp.tid(); c[i] = a[i] * s + b[i]
a = wp.array(np.ones(1024, dtype=np.float32), dtype=float, device="cuda:0")
c = wp.zeros(1024, dtype=float, device="cuda:0")
wp.launch(saxpy, dim=1024, inputs=[a, a, c, 3.0], device="cuda:0")
wp.synchronize(); print(c.numpy()[:3])      # 期望 [4. 4. 4.]
```

本次构建的实测结果：A–F 全部通过（RTX 5060 Laptop、算力 12.0、驱动 595.99.02、toolkit 1.20.1）。


## 训练

```bash
cd /workspace/RL-X/experiments
python experiment.py \
    --algorithm.name=ppo_gru.flax_full_jit \
    --algorithm.total_timesteps=2000000000 \
    --algorithm.evaluation_and_save_frequency=16777216 \
    --algorithm.gae_lambda=0.7 \
    --algorithm.entropy_coef=0.002 \
    --algorithm.max_grad_norm=3.0 \
    --environment.name=custom_mujoco.robocup_soccer.locomotion.mjx \
    --environment.seed=0 \
    --runner.track_console=True \
    --runner.track_wandb=False \
    --runner.save_model=True \
    --runner.project_name="robocup_soccer_locomotion" \
    --runner.exp_name="try"
```

**短跑冒烟时一定要同时改 `evaluation_and_save_frequency`。** 它默认 17301504，
而更新次数 = `total_timesteps // evaluation_and_save_frequency`。只把总步数改小（例如 65536），
这个除法等于 0：XLA 照常编译、进程退出码 0、不打印任何指标，但一次参数更新都没跑。

可用的短跑配置（已实测：262144 步、80 次更新、SPS ≈ 34k、约 10 分钟，其中编译占大头）：

```bash
python experiment.py \
    --algorithm.name=ppo_gru.flax_full_jit \
    --algorithm.total_timesteps=262144 \
    --algorithm.evaluation_and_save_frequency=65536 \
    --algorithm.minibatch_size=32768 \
    --environment.name=custom_mujoco.robocup_soccer.locomotion.mjx \
    --environment.nr_envs=256 \
    --runner.track_console=True --runner.track_wandb=False --runner.save_model=True
```

`evaluation_and_save_frequency` 必须是 `nr_envs * nr_steps` 的整数倍。
首次运行有较长的 XLA 编译期，期间 GPU 利用率接近 0，属正常。

结果写在 `experiments/runs/<project>/<exp>/<run>`（在挂载的 RL-X 里，容器删除后保留）。


## 无 GPU 时的 CPU 回退（已实测，但很慢）

- PyTorch：加 `--algorithm.device=cpu --algorithm.bf16_mixed_precision_training=False`
- JAX/MJX：设 `JAX_PLATFORMS=cpu`（不设也能回退，但会先打印一次 CUDA 初始化错误日志）
- 实测：16 envs、8192 步、8 次更新、SPS ≈ 400（GPU 约 34k），编译约 8 分钟


## 仿真

Apollo 是 C++ 项目，第一次使用前在容器里编译（产物留在挂载目录）：

```bash
rlx-apollo-build
```

必须在容器内编译：Apollo 会把源码绝对路径写进二进制，用于运行时定位 `assets`。

然后一条命令启动服务器、7 个 agent 和开球：

```bash
rlx-sim-demo
```

- 默认 `--no-render`；有图形环境时用 `rlx-sim-demo --render`
- 日志默认在容器内 `/tmp/rlx_sim/`；想保留就设 `RLX_RUN_DIR=<挂载目录>`
- `Ctrl-C` 会先停 agent、再停服务器，退出后不应有残留进程
- 机器人站在 `BeforeKickOff` 不动，通常是没有发 `(kickOff Left)`；`rlx-sim-demo` 已经自动发了

Apollo 的推理走 ONNX Runtime 的 CPU 后端，仿真本身不使用 GPU。

本次实测：7 条 `joined the game`，`play_mode` 从 1（KickOff_Left）进入 3（PlayOn），
agent 日志为空，退出后无残留进程。


## 模型转换

`model_conversion/` 里是从训练 checkpoint 到 Apollo `policy.onnx` 的完整链路，
包括脚本、示例产物和详细说明。基本用法：

```bash
cd /workspace/RL-X/rl_x/environments/custom_mujoco/robocup_soccer/rcssserver_deployment
python convert.py --model ../pretrained.model --output pretrained_nn.pth \
    --meta-output pretrained_nn_meta.json --obs-dim 92 --action-dim 23 \
    --policy-observation-indices "$(seq -s, 0 81)"
python export_onnx.py --weights pretrained_nn.pth --meta pretrained_nn_meta.json \
    --output policy.onnx
```

`export_onnx.py` 自带 onnxruntime 与 PyTorch 的数值对比，误差超限会直接报错。
把产物覆盖到 Apollo 的 `assets/networks/walk/policy.onnx` 后重启仿真。
输入输出契约和 `std_dev` 判据见 `model_conversion/README.md`。


## 数据持久化

容器用 `--rm` 启动，退出即删除。只有写在**挂载目录**里的文件会保留：

| 内容 | 位置 | 容器删除后 |
| --- | --- | --- |
| 训练结果 `runs/` | 挂载的 RL-X | 保留 |
| Apollo 编译产物 `build/` | 挂载的 ApolloCodebase | 保留 |
| 仿真日志（默认） | 容器内 `/tmp/rlx_sim` | 丢失 |
| 容器内 `pip/apt install` | 容器文件系统 | 丢失 |


## 版本与校验

| 项 | 值 |
| --- | --- |
| 镜像 ID | `sha256:79cd42144e6ea53a61b7352ec0eaf5ca55592544a9ad611d6cb8b7a0e23d61eb` |
| 完整压缩包 sha256 | `98f0af292d0fbed4aa3d3f8a6ad1c5edb77b0b2fa06072275fe4ad40654ea8d2` |
| 压缩包大小 | 6,709,152,584 字节（6.3 GiB） |
| 分卷 | 4 个，每个 1900 MiB（最后一卷较小） |
| 构建/验证日期 | 2026-10-20 |

镜像与构建时的 RL-X `pyproject.toml` 绑定。如果之后 RL-X 的依赖变了，镜像里的环境可能过时，
需要重新构建。


## 未验证的部分

- 多 GPU 调度
- 驱动低于 CUDA 13（R580 之前）时 Warp 的行为
- Windows / Docker Desktop（本镜像按 Ubuntu 原生 Linux 验证）
- CPU 完整训练（只做了小规模短跑）
