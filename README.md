# KAN-RL Double Pendulum

基于 **Kolmogorov-Arnold Network (KAN)** 的强化学习智能体，用于解决双倒立摆 (Acrobot) 连续控制问题。

**核心目标**：利用 KAN 的稀疏化特性，将 Agent 的黑盒控制策略提取为人类可读的数学公式。

![算法对比](results/algorithm_comparison.png)

---

## 项目概述

本项目通过对比 **BC、PPO、SAC、DAgger** 四种算法，验证了 KAN 在强化学习中的可行性和可解释性。

### 实验结果总览

| 排名 | 算法 | 平均奖励 | 成功率 | 训练时间 | 评价 |
|:----:|------|----------|--------|----------|------|
| 🥇 | **SAC** | **-193.7** | **96%** | ~56分钟 | 🏆 冠军，超越专家 |
| 🥈 | 启发式专家 | -265.8 | - | - | 基线参考 |
| 🥉 | **BC** | -439.2 | **88%** | ~2分钟 | 稳定基线 |
| 4 | DAgger (30轮) | -443.2 | 82% | ~90分钟 | 接近 BC |
| 5 | PPO (50万步) | -466.5 | 50% | ~60分钟 | 不如 BC |

**关键发现**：
- SAC 是唯一**超越专家**的算法（快 27%，195 步 vs 267 步）
- KAN 网络在 4GB 显存下成功训练，峰值占用 **< 10MB**
- 网络稀疏化达 **43%**，可提取显式数学公式

---

## 网络架构设计

### KAN 极简结构 [6, 8, 1]

```
输入层 (6维)          隐藏层 (8节点)         输出层 (1维)
[c1, s1, c2, s2,     ┌──────────────┐
 d1, d2] ─────────>  │  KAN Layer   │ ──────>  action
    │                │  B-spline    │            │
    │                │  Grid=5      │          tanh
    │                └──────────────┘            │
    │                                            │
    └────────────────────────────────────────────┘
```

**设计约束**（针对 RTX 3050Ti 4GB 显存）：
- B-spline 网格大小：5（节省显存）
- 批次大小：64（显存安全）
- 参数总量：448（可解释性强）

---

## 快速开始

### 环境配置

```bash
# 创建 conda 环境
conda create -n kan_rl python=3.10 -y
conda activate kan_rl

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

### 评估最佳策略（SAC）

```bash
python scripts/evaluate.py --model checkpoints/sac_kan_model.pt --n_episodes 100
```

### 查看符号化公式

```bash
# BC 策略公式
cat results/symbolic_formula.py

# SAC 策略公式
cat results/sac_symbolic_formula.py
```

### 生成对比可视化

```bash
python scripts/plot_comparison.py
```

---

## 算法对比详情

### 冠军：SAC（Soft Actor-Critic）

![SAC 训练细节](results/sac_training_detail.png)

SAC 是本项目表现最佳的算法：
- **平均奖励：-193.7**（超越专家 27%）
- **成功率：96%**（最高）
- **特点**：Off-Policy + 自适应熵调节 + 双 Q 网络

### BC（Behavioral Cloning）

- **平均奖励：-439.2**
- **成功率：88%**
- **特点**：简单高效，2分钟训练即可使用
- **局限**：分布偏移问题，上限受限于专家

### PPO（Proximal Policy Optimization）

- **平均奖励：-466.5**
- **成功率：50%**
- **分析**：50万步训练未能超越 BC，需要更精细调参

### DAgger（Dataset Aggregation）

- **平均奖励：-443.2**
- **成功率：82%**
- **分析**：30轮迭代未能突破，说明启发式专家本身有局限

---

## 可解释性分析

### 稀疏化统计

| 层 | 总参数 | 零参数 | 稀疏化比例 |
|----|--------|--------|------------|
| Layer 1 | 384 | ~170 | 42-47% |
| Layer 2 | 64 | ~25 | 39-41% |
| **总计** | **448** | **~195** | **~43%** |

### 提取的符号化公式

**BC 策略核心公式**（简化）：
```python
# Layer 1: 6输入 -> 8隐藏
h0 = 0.327*c1 + 1.285*s1 + 0.396*d1 + 0.267*d2 + ...
h1 = 4.073*s2 + 0.269*c1 - 0.040*c2 + ...
...

# Layer 2: 8隐藏 -> 1输出
action = tanh(1.634*h3 + 1.625*h7 + 1.446*h5 + ...)
```

**特征重要性分析**：
- `s2` (sinθ₂) 权重最高 → 第二摆角度最重要
- `d1, d2` (角速度) 次要
- `c1, c2` (cos) 再次之

---

## 技术亮点

### 1. 显存优化策略
- 小网格 B-spline (grid_size=5)
- 分批次计算避免大矩阵广播
- 峰值显存占用 **< 10MB**

### 2. 梯度稳定化
- 梯度裁剪 (clip_norm=1.0)
- 层归一化 (LayerNorm)
- 网格定期更新机制

### 3. L1 稀疏化
- 训练时施加 L1 正则
- 鼓励权重归零
- 便于符号公式提取

---

## 项目结构

```
kan_rl_double_pendulum/
├── checkpoints/
│   ├── bc_kan_model.pt          # BC 最佳模型 (-439, 88%)
│   ├── sac_kan_model.pt         # SAC 冠军模型 (-194, 96%) ⭐
│   ├── ppo_kan_model_v2.pt      # PPO 调优版
│   └── dagger_kan_model.pt      # DAgger 30轮
├── results/
│   ├── symbolic_formula.py      # BC 符号公式
│   ├── sac_symbolic_formula.py  # SAC 符号公式
│   ├── algorithm_comparison.png # 对比图
│   └── sac_training_detail.png  # SAC 细节
├── models/kan_policy.py         # KAN 核心实现
├── agents/
│   ├── bc_agent.py              # 行为克隆
│   ├── ppo_agent.py             # PPO 实现
│   ├── sac_agent.py             # SAC 实现 ⭐
│   └── dagger_agent.py          # DAgger 实现
├── scripts/
│   ├── 1_generate_expert.py     # 专家数据生成
│   ├── 2_train_bc.py            # BC 训练
│   ├── 3_train_ppo.py           # PPO 训练
│   ├── 4_train_sac.py           # SAC 训练 ⭐
│   ├── 5_train_dagger.py        # DAgger 训练
│   ├── evaluate.py              # 策略评估
│   ├── extract_sac_symbolic.py  # SAC 符号提取
│   └── plot_comparison.py       # 生成对比图
├── config.yaml                  # 全局配置
└── README.md                    # 本文件
```

---

## 使用指南

### 训练流程

```bash
# 1. 生成专家数据
python scripts/1_generate_expert.py --algorithm heuristic --n_trajectories 1000

# 2. 训练 BC（快速基线）
python scripts/2_train_bc.py

# 3. 训练 SAC（推荐）
python scripts/4_train_sac.py --total_timesteps 100000

# 4. 训练 PPO
python scripts/3_train_ppo.py --bc_checkpoint checkpoints/bc_kan_model.pt

# 5. 训练 DAgger
python scripts/5_train_dagger.py --expert_algorithm heuristic --n_iterations 10
```

### 评估策略

```bash
# 评估 SAC
python scripts/evaluate.py --model checkpoints/sac_kan_model.pt --n_episodes 100

# 评估 BC
python scripts/evaluate.py --model checkpoints/bc_kan_model.pt --n_episodes 100
```

### 提取符号公式

```bash
# 提取 SAC 公式
python scripts/extract_sac_symbolic.py

# 提取 BC 公式
python utils/symbolic.py
```

---

## 结论与展望

### 主要结论

1. **KAN 可以用于 RL**：成功实现了 [6,8,1] 极简结构的连续控制
2. **SAC 是最佳搭档**：Off-Policy + 自适应熵调节超越所有对比算法
3. **可解释性达成**：43% 稀疏化，可提取显式数学公式
4. **4GB 显存可行**：通过小网格和分块计算实现

### 未来方向

1. **更复杂的任务**：CartPole Swing-up, Pendulum 等
2. **更深的 KAN**：探索 [6, 16, 8, 1] 等多层结构
3. **网格自适应优化**：动态调整 B-spline 网格
4. **自动符号简化**：用 SymPy 进一步简化提取的公式

---

## 参考文献

1. Liu, Z., et al. (2024). "KAN: Kolmogorov-Arnold Networks". arXiv:2404.19756.
2. Haarnoja, T., et al. (2018). "Soft Actor-Critic". ICML.
3. Schulman, J., et al. (2017). "Proximal Policy Optimization". arXiv:1707.06347.
4. Ross, S., et al. (2011). "A Reduction of Imitation Learning to Active Learning". AISTATS.

---

## 硬件信息

- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4GB VRAM)
- **训练时间**: SAC ~56分钟, BC ~2分钟, PPO ~60分钟, DAgger ~90分钟
- **完成日期**: 2024

---

**项目状态**：✅ 已完成，**SAC 策略 (-193.7, 96%)** 为最终推荐方案
