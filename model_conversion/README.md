# 从训练 checkpoint 到 Apollo 的 ONNX 部署链

本目录把一次 RL-X 训练产出的 `.model`，一路转成 Apollo（Booster T1）实际加载的
`policy.onnx`。整条链的两个步骤都可以在本仓库内重跑，产物是确定性的。

```
rl_x/environments/custom_mujoco/robocup_soccer/pretrained.model   训练 checkpoint (orbax, 6.0 MB)
        │
        │  convert.py          Flax 参数 -> PyTorch state_dict
        ▼
rcssserver_deployment/pretrained_nn.pth          1.31 MB, 31 个张量
rcssserver_deployment/pretrained_nn_meta.json    895 B, 结构 + std_dev
        │
        │  export_onnx.py      打包成 Apollo 的单输入/单输出契约
        ▼
rcssserver_deployment/policy.onnx                1.32 MB
        │
        │  拷贝
        ▼
ApolloCodebase/assets/networks/walk/policy.onnx  ← Apollo 运行时加载的文件
```

`convert.py` 是上游 RL-X 自带的脚本（作者 Nico），本仓库未改动；`export_onnx.py`
是把最后一步 ONNX 导出补进流水线的新脚本。


## 运行环境

两个脚本都要用 `rlx` 这个 conda 环境（jax / optax / orbax / torch / onnx /
onnxruntime 都在里面），并且需要 `PYTHONPATH` 指向 RL-X 仓库根目录才能 import 到
`rl_x.*`：

```bash
cd rl_x/environments/custom_mujoco/robocup_soccer/rcssserver_deployment
export PYTHONPATH=$(git rev-parse --show-toplevel)
PY=~/miniconda3/envs/rlx/bin/python
```

不要加 `-I`：隔离模式下脚本所在目录不会进 `sys.path`，`from torch_policy import ...`
会失败。


## 第一步：`.model` → `.pth`

```bash
$PY convert.py \
    --model ../pretrained.model \
    --output pretrained_nn.pth \
    --meta-output pretrained_nn_meta.json \
    --obs-dim 92 \
    --action-dim 23 \
    --policy-observation-indices "$(seq -s, 0 81)"
```

`convert.py` 做的事情：把 orbax checkpoint 解成 Flax 参数树，套进一个临时的
`DummyEnv`，再逐个模块把 `kernel`/`scale` 转成 PyTorch 的 `weight`，Flax 的
`GRUCell` 展开成 `ir`/`iz`/`in_proj`/`hr`/`hz`/`hn` 六个线性层。

参数说明（这三个坑都实际踩过）：

- **`--obs-dim 92`，不是上游 README 里写的 82。** `convert.py` 先用 `DummyEnv`
  随机构造一份参数树，再让 orbax 按这个形状去恢复 checkpoint。critic 吃的是完整的
  92 维 observation（82 维策略观测 + 2+2+2 脚部接触/离地时间 + 3 维 IMU 线速度 +
  1 维 critic 外感受）。填 82 会在恢复 critic 第一层时直接报错：

  ```
  ValueError: Requested shape: (82, 512) is not compatible with the stored shape: (92, 512).
  ```

- **`--policy-observation-indices` 必须显式给。** 策略的输入层是 82 维，不传这个参数
  时 `convert.py` 默认用 `range(obs_dim)`，也就是 92 个索引，长度对不上会抛
  `ValueError`。

- **这里传的是恒等索引 `[0..81]`，和训练环境真实的索引并不一样。** 训练环境给策略
  的是完整 92 维观测里的一部分（`[0:69] + [78:91]`，即前 69 维加上从 IMU 角速度
  开始的后 13 维）。部署时 Apollo 自己已经把
  82 维摘好了再喂给网络，所以网络这一层要的是恒等映射——`convert.py` 只用到索引的
  *长度* 来定输入层形状，索引值本身会原样写进 meta 和 ONNX 图里的 `index_select`。
  如果以后想把选择也交给网络做，就传真实索引并让 `--obs-dim 92`，但那样 Apollo
  侧的输入宽度也要一起改（`[1,156]`），目前不需要。

产物 `pretrained_nn_meta.json` 里最该看的字段：

| 字段 | 含义 |
| --- | --- |
| `std_dev` | 策略输出高斯的标准差，`exp(policy_logstd)`。**收敛与否的第一体检指标** |
| `expected_policy_obs_dim` | 策略输入维度，应为 82 |
| `gru_hidden_dim` | GRU 隐状态维度，64 |
| `action_dim` | 动作维度，23 |
| `obs_encoding_dim` | 观测编码宽度，128 |
| `gru_obs_combine_method` | `concat` 或 `film`，决定 GRU 隐状态怎么进 torso |
| `share_gru_obs_encoder` | 策略观测和 GRU 观测是否共用编码器 |
| `policy_observation_indices` | 上面的索引，部署时按恒等理解即可 |

参考值：本目录的 `pretrained_nn.pth` 出自 `pretrained.model`，`std_dev = 0.0839`，
在仿真里能走。作者训练出的模型如果 `std_dev` 是几十这个量级，说明策略根本没收敛，
转换链再正确也没用，得回去重训。


## 第二步：`.pth` → `.onnx`

```bash
$PY export_onnx.py \
    --weights pretrained_nn.pth \
    --meta pretrained_nn_meta.json \
    --output policy.onnx
```

脚本做四件事：

1. 用一个 `ApolloPolicyGRU` 把 `TorchPolicyGRU` 包一层，让整张图只有一个输入一个输出
   （见下一节的契约）。
2. 以 opset 18、静态 shape 导出。
3. `torch.onnx.export` 在权重较大时会把权重甩到 `.onnx.data` 旁挂文件里，而 Apollo
   只认一个 `.onnx` 路径，所以这里把外部权重内联回主文件、删掉旁挂文件。同时清掉
   torch 给每个节点打的调试元数据——它们会把包装类的名字和本脚本的**绝对路径**写进
   产物，清掉之后导出结果只由权重决定，重跑两次字节一致。
4. 自检：断言输入输出形状，并用随机输入对比 onnxruntime 和 PyTorch 的输出，相对误差
   超过 `1e-5` 直接报错。

`--obs-dim` 默认取 meta 里的 `expected_policy_obs_dim`，一般不用手填。脚本会检查 meta
里的索引是不是恒等映射，不是就拒绝导出，避免导出出一张和 Apollo 的喂法对不上的图。

输出示例：

```
obs_dim 82 + carry_dim 64 -> action_dim 23 + carry_dim 64
input  obs     [1, 146]
output actions [1, 87]
torch vs onnxruntime: max abs diff 3.815e-06 over 5 inputs, 4.142e-07 relative to output scale 9.210
policy std_dev 0.083903
```


## ONNX 契约

```
输入  obs     [1, 146]  =  [ 82 维观测 | 64 维 GRU 隐状态 carry ]
输出  actions [1, 87]   =  [ 23 维动作均值 | 64 维下一步 carry ]
```

策略是循环网络，但 Apollo 的 `OnnxSession` 只绑定第 0 个输入和第 0 个输出，`run()`
也不解释张量结构。所以做法是把 64 维 GRU 隐状态直接拼在观测后面和动作均值后面，
把「循环」拆成调用方每步手动回灌：**取出输出后 64 维当下一步的 carry 再拼进输入**。
Apollo 侧对应 `WalkRunner::step()` 里的 `std::copy`。

第一帧的 carry 用 64 个 0 初始化，和训练时 `initialize_carry` 一致。

### 82 维观测的构成

| 切片 | 内容 | 归一化 |
| --- | --- | --- |
| `0:23` | 关节位置 | `(q - nominal) / 3.14` |
| `23:46` | 关节速度 | `/ 100` |
| `46:69` | 上一步网络输出 | `/ 10` |
| `69:72` | IMU 角速度 | `clip(v / 50, -1, 1)` |
| `72:75` | 速度指令 `(vx, vy, wz)` | 原样 |
| `75:79` | 步态相位 `[sin(p0), sin(p1), cos(p0), cos(p1)]` | 原样 |
| `79:82` | 重力向量（机体系） | 原样 |

`nominal` 是训练环境的 home keyframe，不是 Apollo 里的 `kDefaultPosRad`，腿部关节
不一样。`ApolloCodebase/src/behavior/walk_runner.h` 里的 `kHomePoseRad` 和
`kObsDim`/`kCarryDim`/`kActionDim` 以及各段 offset 就是这份契约的 C++ 副本，两边改
必须同步改。

动作解码同样是 `target = nominal + action * 0.5`（`scaling_factor`），PD 增益用
训练环境那一套 `kp=25, kd=0.6`，不是 T1RobotModel 的每关节增益。


## 装到 Apollo

```bash
cp policy.onnx ~/桌面/ApolloCodebase/assets/networks/walk/policy.onnx
```

Apollo 侧只要这个文件，重启后由 `WalkRunner` 在构造时按上面的形状契约加载；形状对
不上会直接抛 `ONNX input shape mismatch`。


## 验证过的结果

以 `pretrained.model` 为输入，跑完整条链：

- 第一步产物 `pretrained_nn.pth` / `pretrained_nn_meta.json`
- 第二步产物 `policy.onnx`，sha256 `70be15ae0df219acb726082d7a626744e73027663e1a9c205d5450bc03a97529`
- 连跑两次 `export_onnx.py`，产物字节一致
- onnxruntime 对 PyTorch 参考实现：5 个随机输入最大绝对误差 `3.815e-06`，相对输出
  量级 `4.142e-07`
- 上面这份图和 Apollo 里现装的 `assets/networks/walk/policy.onnx`：50 个随机输入
  输出**完全相等**（最大差 `0.0`），两者只差 torch 的调试元数据字节；连续 30 步
  回灌 carry 的循环 rollout，对 PyTorch 参考实现误差 `5.7e-06`

也就是说仓库里重跑出来的 `policy.onnx` 和 Apollo 正在跑的那份是同一个函数。


## 排错

- **`std_dev` 很大（几十）**：策略没收敛，别浪费时间部署，先回去把训练跑通。
- **`ValueError: Requested shape: (82, 512) ... (92, 512)`**：`--obs-dim` 填成了 82，
  改回 92 并显式传 `--policy-observation-indices`。
- **`ValueError: policy_observation_indices length ... does not match`**：同上，少传了
  索引参数。
- **`ModuleNotFoundError: No module named 'rl_x'`**：忘了 `PYTHONPATH`。
- **`ModuleNotFoundError: No module named 'torch_policy'`**：用了 `-I`，去掉即可。
- **Apollo 报 `ONNX input shape mismatch`**：装错文件了（比如装回了旧的 MLP 策略），
  或者 Apollo 侧的 `kObsDim`/`kCarryDim` 和这份图对不上。
- **仿真里机器人站着不动**：多半不是策略的问题，是服务器还停在 `BeforeKickOff`，
  需要从 monitor 端口发一次 `(kickOff Left)`。
