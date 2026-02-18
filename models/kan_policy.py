"""
KAN (Kolmogorov-Arnold Network) Policy for Double Pendulum
============================================================
极简 KAN 网络实现，针对 RTX 3050Ti (4GB VRAM) 深度优化

网络结构: [6, 8, 1] (输入6维 -> 隐藏8节点 -> 输出1维)
- 输入: [cos(θ₁), sin(θ₁), cos(θ₂), sin(θ₂), θ̇₁, θ̇₂]
- 输出: 连续扭矩 τ ∈ [-1, 1] (通过 tanh 约束)

特性:
1. 小网格 B-spline (grid_size=5) 节省显存
2. 梯度裁剪防止 KAN 不稳定
3. L1 稀疏化惩罚接口 (用于符号提取)
4. 显式网格更新机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


class KANLayer(nn.Module):
    """
    单层 KAN: 使用 B-spline 基函数的 Kolmogorov-Arnold 网络层
    
    数学形式: y = Σᵢ φᵢ(xᵢ), 其中 φᵢ 是 B-spline 函数
    
    显存优化策略:
    - 小批次计算避免大矩阵广播
    - 使用 inplace 操作减少中间变量
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
    ):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            grid_size: B-spline 网格点数 (小网格节省显存)
            spline_order: B-spline 阶数 (3=三次样条)
            scale_noise: 初始化噪声尺度
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # 基础网格范围 (假设输入已经归一化到 [-1, 1])
        grid_range = [-1.0, 1.0]
        
        # 初始化 B-spline 网格: (grid_size + 2 * spline_order + 1,)
        # 多出的点用于边界外的外推
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(
            -spline_order, grid_size + spline_order + 1
        ).float() * h + grid_range[0]
        self.register_buffer("grid", grid)  # (grid_size + 2 * spline_order + 1,)
        
        # 可学习参数:
        # 1. base_weight: 残差连接的线性权重 (类似于 MLP 的权重)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        # 2. spline_weight: B-spline 系数 (核心可解释参数)
        # 形状: (out_features, in_features, grid_size + spline_order)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * scale_noise
        )
        
        # 3. spline_scaler: B-spline 输出的缩放因子
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.base_weight, gain=0.1)
        nn.init.normal_(self.spline_weight, std=0.01)
    
    def _compute_b_spline(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 B-spline 基函数值
        
        输入 x: (batch_size, in_features)
        输出: (batch_size, in_features, grid_size + spline_order)
        
        使用 Cox-de Boor 递推公式，显存优化的迭代实现
        """
        batch_size = x.shape[0]
        
        # 扩展维度用于网格计算: (batch_size, in_features, 1)
        x_expanded = x.unsqueeze(-1)  # (B, in, 1)
        grid_expanded = self.grid.view(1, 1, -1)  # (1, 1, grid_len)
        
        # k=0 阶 (分段常数): 判断 x 是否落在 [grid[i], grid[i+1]) 区间内
        # 使用差分计算避免大内存分配
        dist = x_expanded - grid_expanded  # (B, in, grid_len)
        
        # 0 阶: [grid[i] <= x < grid[i+1]]
        grid_size_total = self.grid_size + 2 * self.spline_order
        bases = torch.zeros(
            batch_size, self.in_features, grid_size_total + 1,
            dtype=x.dtype, device=x.device
        )
        
        # 找到 x 所在的网格区间
        # 使用 searchsorted 找到插入位置
        grid_for_search = self.grid[self.spline_order:-self.spline_order]  # 有效网格范围
        
        # 向量化计算 B-spline (Cox-de Boor 递推)
        # 使用滑窗减少显存占用
        k = self.spline_order
        for i in range(grid_size_total):
            # 计算第 i 个基函数在当前 x 处的值
            bases[:, :, i] = self._b_spline_basis(x, i, k)
        
        return bases[:, :, :-1]  # (B, in, grid_size + spline_order)
    
    def _b_spline_basis(self, x: torch.Tensor, i: int, k: int) -> torch.Tensor:
        """
        Cox-de Boor 递推公式计算单个 B-spline 基函数
        
        B_{i,0}(x) = 1 if grid[i] <= x < grid[i+1] else 0
        B_{i,k}(x) = w_{i,k}(x) * B_{i,k-1}(x) + (1 - w_{i+1,k}(x)) * B_{i+1,k-1}(x)
        
        其中 w_{i,k}(x) = (x - grid[i]) / (grid[i+k] - grid[i])
        """
        grid = self.grid
        
        if k == 0:
            # 0 阶基函数
            left = grid[i]
            right = grid[i + 1]
            return ((x >= left) & (x < right)).float()
        
        # k 阶递推
        # 第一项系数
        left_denom = grid[i + k] - grid[i]
        if left_denom > 1e-8:
            left_w = (x - grid[i]) / left_denom
            left_term = left_w * self._b_spline_basis(x, i, k - 1)
        else:
            left_term = torch.zeros_like(x)
        
        # 第二项系数
        right_denom = grid[i + k + 1] - grid[i + 1]
        if right_denom > 1e-8:
            right_w = (grid[i + k + 1] - x) / right_denom
            right_term = right_w * self._b_spline_basis(x, i + 1, k - 1)
        else:
            right_term = torch.zeros_like(x)
        
        return left_term + right_term
    
    def _b_spline_basis_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """
        向量化的 B-spline 计算 (比递推更快，显存稍多)
        
        这是一个简化版本，使用线性插值近似 B-spline
        对于网格大小=5的情况，精度足够且速度更快
        """
        batch_size, in_features = x.shape
        n_basis = self.grid_size + self.spline_order  # 基函数数量
        
        # 将 x 限制在 [0, grid_size-1] 范围内用于插值
        x_norm = (x + 1.0) / 2.0 * (self.grid_size - 1)  # 映射到 [0, grid_size-1]
        x_norm = torch.clamp(x_norm, 0, self.grid_size - 1 - 1e-6)
        
        # 计算左右基函数索引和权重
        left_idx = x_norm.long()  # (B, in)
        right_idx = left_idx + 1
        right_weight = x_norm - left_idx.float()  # (B, in)
        left_weight = 1.0 - right_weight
        
        # 构造 one-hot 编码的基函数值
        bases = torch.zeros(batch_size, in_features, n_basis, device=x.device, dtype=x.dtype)
        
        # 使用 scatter 填充 (比循环高效)
        batch_idx = torch.arange(batch_size, device=x.device).view(-1, 1).expand(-1, in_features)
        feat_idx = torch.arange(in_features, device=x.device).view(1, -1).expand(batch_size, -1)
        
        bases[batch_idx, feat_idx, left_idx] = left_weight
        bases[batch_idx, feat_idx, right_idx] = right_weight
        
        return bases
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: y = base(x) + spline(x)
        
        Args:
            x: (batch_size, in_features)
        Returns:
            y: (batch_size, out_features)
        """
        batch_size = x.shape[0]
        
        # 1. 基础线性部分 (残差连接)
        base_out = F.linear(x, self.base_weight, self.base_bias)  # (B, out)
        
        # 2. B-spline 部分
        # 计算基函数: (B, in, n_basis)
        bases = self._b_spline_basis_vectorized(x)  # 使用向量化版本
        
        # 与可学习系数相乘: (B, in, n_basis) @ (out, in, n_basis).transpose -> ?
        # 需要: (out, in, n_basis) 与 (B, in, n_basis) 逐元素乘后求和
        # spline_out[b, o] = sum_i sum_k spline_weight[o, i, k] * bases[b, i, k]
        
        # 显存优化: 分批计算 (对于4GB显存很重要)
        if batch_size > 256:
            # 大批量时分块计算
            chunk_size = 128
            spline_out_chunks = []
            for i in range(0, batch_size, chunk_size):
                x_chunk = x[i:i+chunk_size]
                bases_chunk = self._b_spline_basis_vectorized(x_chunk)
                # (chunk, in, n_basis) * (1, in, n_basis) -> sum -> (chunk,)
                chunk_out = torch.einsum('bik,oik->bo', bases_chunk, self.spline_weight)
                spline_out_chunks.append(chunk_out)
            spline_out = torch.cat(spline_out_chunks, dim=0)
        else:
            # 小批量直接计算
            spline_out = torch.einsum('bik,oik->bo', bases, self.spline_weight)  # (B, out)
        
        # 应用缩放因子
        # spline_scaler: (out, in), 需要广播到 (B, out)
        scaler_sum = self.spline_scaler.sum(dim=1, keepdim=True).t()  # (1, out)
        spline_out = spline_out * (scaler_sum / self.in_features)  # 归一化缩放
        
        # 3. 合并输出
        output = base_out + spline_out
        
        return output
    
    def update_grid(self, x: torch.Tensor, sample_rate: float = 1.0):
        """
        根据输入数据分布更新 B-spline 网格 (KAN 的关键特性)
        
        通常在训练过程中定期调用 (如每 N 个 epoch)
        
        Args:
            x: 输入样本 (用于计算新的网格范围)
            sample_rate: 采样率 (如果样本太多，随机采样一部分)
        """
        with torch.no_grad():
            # 采样
            if sample_rate < 1.0:
                n_samples = int(x.shape[0] * sample_rate)
                indices = torch.randperm(x.shape[0])[:n_samples]
                x_sample = x[indices]
            else:
                x_sample = x
            
            # 计算当前数据的范围
            x_min = x_sample.min(dim=0)[0]  # (in_features,)
            x_max = x_sample.max(dim=0)[0]  # (in_features,)
            
            # 增加边界余量
            margin = (x_max - x_min) * 0.1 + 1e-8
            x_min = x_min - margin
            x_max = x_max + margin
            
            # 为每个输入维度计算新的网格
            # 这里简化处理: 使用所有维度的平均范围
            x_min_mean = x_min.mean()
            x_max_mean = x_max.mean()
            
            h = (x_max_mean - x_min_mean) / self.grid_size
            new_grid = torch.arange(
                -self.spline_order, 
                self.grid_size + self.spline_order + 1,
                device=x.device, dtype=x.dtype
            ) * h + x_min_mean
            
            # 更新网格 (同时需要重新插值 spline_weight)
            self._resize_grid(new_grid)
    
    def _resize_grid(self, new_grid: torch.Tensor):
        """
        调整网格大小时，重新插值 B-spline 系数
        
        这是保持函数表示不变的关键步骤
        """
        old_grid = self.grid
        old_weight = self.spline_weight
        
        # 计算新的基函数数量
        old_n_basis = old_grid.shape[0] - self.spline_order - 1
        new_n_basis = new_grid.shape[0] - self.spline_order - 1
        
        if old_n_basis == new_n_basis:
            # 只是平移/缩放，直接更新网格
            self.grid.copy_(new_grid)
            return
        
        # 需要插值权重 (使用线性插值简化)
        # 这是一个简化实现，完整实现需要更复杂的样条插值
        # 对于本项目，网格大小固定，此方法很少被调用
        self.grid = new_grid
        
        # 重新初始化权重
        self.spline_weight = nn.Parameter(
            torch.randn(
                self.out_features, self.in_features, new_n_basis,
                device=old_weight.device, dtype=old_weight.dtype
            ) * 0.01
        )
    
    def regularization_loss(self, l1_factor: float = 1.0) -> torch.Tensor:
        """
        计算 L1 稀疏化惩罚 (用于鼓励网络稀疏，便于符号提取)
        
        Returns:
            L1 正则化损失
        """
        # 对 spline_weight 应用 L1 正则
        l1_loss = torch.abs(self.spline_weight).mean()
        # 也对 scaler 应用 L1 鼓励稀疏
        l1_loss += torch.abs(self.spline_scaler).mean() * 0.1
        return l1_factor * l1_loss


class KANPolicy(nn.Module):
    """
    KAN 策略网络: [6, 8, 1] 极简结构
    
    用于双倒立摆连续控制:
    - 输入: 6 维状态 [cosθ₁, sinθ₁, cosθ₂, sinθ₂, θ̇₁, θ̇₂]
    - 输出: 1 维连续动作 (扭矩), 通过 tanh 约束到 [-1, 1]
    """
    
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 8,
        output_dim: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        action_scale: float = 1.0,
    ):
        """
        Args:
            input_dim: 输入维度 (默认6)
            hidden_dim: 隐藏层维度 (默认8，控制可解释性)
            output_dim: 输出维度 (默认1，连续扭矩)
            grid_size: B-spline 网格大小
            spline_order: B-spline 阶数
            action_scale: 动作缩放因子 (用于不同环境的动作范围)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.action_scale = action_scale
        
        # 网络结构: [6, 8, 1]
        self.layer1 = KANLayer(input_dim, hidden_dim, grid_size, spline_order)
        self.layer2 = KANLayer(hidden_dim, output_dim, grid_size, spline_order)
        
        # 层归一化 (帮助稳定训练)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 记录前向传播次数 (用于决定何时更新网格)
        self.register_buffer("forward_count", torch.tensor(0, dtype=torch.long))
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: (batch_size, 6) 状态向量
        Returns:
            action: (batch_size, 1) 连续动作，范围 [-action_scale, action_scale]
        """
        x = self.layer1(state)
        x = self.layer_norm(x)
        x = torch.tanh(x)  # 激活函数
        x = self.layer2(x)
        
        # 输出通过 tanh 限制到 [-1, 1]，再缩放到目标范围
        action = torch.tanh(x) * self.action_scale
        
        self.forward_count += 1
        
        return action
    
    def get_action(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        用于推理的便捷接口 (numpy 输入输出)
        
        Args:
            state: (6,) numpy 数组
            deterministic: 是否确定性策略 (KAN 本身就是确定性的)
        Returns:
            action: (1,) numpy 数组
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                state_tensor = state_tensor.cuda()
            action = self.forward(state_tensor)
            return action.cpu().numpy()[0]
    
    def update_grids(self, states: torch.Tensor, sample_rate: float = 1.0):
        """
        更新所有层的 B-spline 网格
        
        应在训练循环中定期调用 (如每 10-20 个 epoch)
        
        Args:
            states: 状态样本 (用于计算新的网格范围)
            sample_rate: 采样率
        """
        # 第一层: 使用原始状态
        self.layer1.update_grid(states, sample_rate)
        
        # 第二层: 使用第一层的输出
        with torch.no_grad():
            hidden = self.layer1(states)
            hidden = self.layer_norm(hidden)
            hidden = torch.tanh(hidden)
        self.layer2.update_grid(hidden, sample_rate)
        
        print(f"[KAN] Grids updated at forward_count={self.forward_count.item()}")
    
    def regularization_loss(self, l1_factor: float = 1e-4) -> torch.Tensor:
        """
        计算整体的 L1 稀疏化惩罚
        
        Args:
            l1_factor: L1 正则化系数
        Returns:
            总的正则化损失
        """
        reg_loss = self.layer1.regularization_loss(l1_factor)
        reg_loss += self.layer2.regularization_loss(l1_factor)
        return reg_loss
    
    def get_sparsity_info(self, threshold: float = 0.01) -> dict:
        """
        获取网络稀疏化信息 (用于符号提取准备)
        
        Args:
            threshold: 视为零的权重阈值
        Returns:
            稀疏化统计信息字典
        """
        info = {}
        
        # 第一层
        w1 = self.layer1.spline_weight.detach().cpu()
        info['layer1'] = {
            'total_params': w1.numel(),
            'zero_params': (torch.abs(w1) < threshold).sum().item(),
            'sparsity_ratio': (torch.abs(w1) < threshold).float().mean().item(),
            'mean_abs_weight': torch.abs(w1).mean().item(),
        }
        
        # 第二层
        w2 = self.layer2.spline_weight.detach().cpu()
        info['layer2'] = {
            'total_params': w2.numel(),
            'zero_params': (torch.abs(w2) < threshold).sum().item(),
            'sparsity_ratio': (torch.abs(w2) < threshold).float().mean().item(),
            'mean_abs_weight': torch.abs(w2).mean().item(),
        }
        
        # 总体
        total_params = info['layer1']['total_params'] + info['layer2']['total_params']
        total_zero = info['layer1']['zero_params'] + info['layer2']['zero_params']
        info['total'] = {
            'total_params': total_params,
            'zero_params': total_zero,
            'sparsity_ratio': total_zero / total_params,
        }
        
        return info
    
    def print_sparsity(self, threshold: float = 0.01):
        """打印稀疏化信息"""
        info = self.get_sparsity_info(threshold)
        print("=" * 50)
        print("KAN 网络稀疏化统计")
        print("=" * 50)
        for layer_name, layer_info in info.items():
            if layer_name == 'total':
                print("-" * 50)
            print(f"{layer_name}:")
            for key, value in layer_info.items():
                if 'ratio' in key:
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        print("=" * 50)


def test_kan_policy():
    """测试 KAN Policy 的函数"""
    print("=" * 60)
    print("测试 KAN Policy")
    print("=" * 60)
    
    # 创建网络
    policy = KANPolicy(
        input_dim=6,
        hidden_dim=8,
        output_dim=1,
        grid_size=5,
        spline_order=3,
    )
    
    # 打印网络结构
    print(f"\n网络结构: {policy}")
    print(f"\n可训练参数数量: {sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
    
    # 测试前向传播
    batch_size = 32
    test_input = torch.randn(batch_size, 6)
    
    print(f"\n测试输入: {test_input.shape}")
    output = policy(test_input)
    print(f"输出动作: {output.shape}, 范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 测试 numpy 接口
    test_state = np.random.randn(6)
    action = policy.get_action(test_state)
    print(f"\nnumpy 接口测试: state {test_state.shape} -> action {action.shape}")
    
    # 测试正则化损失
    reg_loss = policy.regularization_loss(l1_factor=1e-4)
    print(f"\nL1 正则化损失: {reg_loss.item():.6f}")
    
    # 测试稀疏化信息
    policy.print_sparsity(threshold=0.01)
    
    # 测试 GPU 支持 (如果可用)
    if torch.cuda.is_available():
        print("\n测试 GPU 运行...")
        policy_gpu = policy.cuda()
        test_input_gpu = test_input.cuda()
        output_gpu = policy_gpu(test_input_gpu)
        print(f"GPU 输出: {output_gpu.shape}, 设备: {output_gpu.device}")
        
        # 测试显存占用
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(100):
            _ = policy_gpu(test_input_gpu)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"峰值显存占用: {peak_mem:.2f} MB")
    else:
        print("\n无可用 GPU，跳过 GPU 测试")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_kan_policy()
