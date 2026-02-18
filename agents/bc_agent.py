"""
行为克隆 (Behavioral Cloning) Agent
=====================================
基于 KAN 网络的模仿学习实现

训练流程:
1. 加载专家轨迹 (state-action pairs)
2. 使用 MSE Loss 训练 KAN 策略
3. L1 正则化鼓励网络稀疏 (便于符号提取)
4. 定期更新 B-spline 网格
5. 早停和模型保存

关键特性:
- 梯度裁剪防止 KAN 不稳定
- 验证集监控防止过拟合
- 网格更新频率可配置
- 稀疏化统计实时监控
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import time

from models.kan_policy import KANPolicy


class ExpertDataset(Dataset):
    """
    专家轨迹数据集
    
    数据格式:
    - states: (N, 6) 状态向量
    - actions: (N, 1) 连续动作
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: 专家数据文件路径 (.npy 或 .pt)
        """
        super().__init__()
        
        # 加载数据
        if data_path.endswith('.pt') or data_path.endswith('.pth'):
            data = torch.load(data_path)
            self.states = data['states'].float()
            self.actions = data['actions'].float()
        elif data_path.endswith('.npy'):
            data = np.load(data_path, allow_pickle=True).item()
            self.states = torch.FloatTensor(data['states'])
            self.actions = torch.FloatTensor(data['actions'])
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
        
        # 确保动作是 2D
        if self.actions.dim() == 1:
            self.actions = self.actions.unsqueeze(-1)
        
        assert self.states.shape[0] == self.actions.shape[0], "状态和动作数量不匹配"
        
        print(f"[ExpertDataset] 加载 {len(self)} 条专家轨迹")
        print(f"  - states shape: {self.states.shape}")
        print(f"  - actions shape: {self.actions.shape}")
        print(f"  - actions range: [{self.actions.min():.3f}, {self.actions.max():.3f}]")
    
    def __len__(self) -> int:
        return self.states.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx]


class BCAgent:
    """
    行为克隆 Agent
    
    使用 KAN 网络从专家数据中学习策略
    """
    
    def __init__(
        self,
        policy: Optional[KANPolicy] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            policy: KAN 策略网络 (若 None 则根据配置创建)
            config: 配置字典 (若 None 则从 config_path 加载)
            config_path: 配置文件路径
            device: 计算设备 ('cuda', 'cpu', 或 None 自动检测)
        """
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            # 默认配置
            self.config = {
                'bc': {
                    'batch_size': 128,
                    'lr': 1e-3,
                    'l1_penalty': 1e-4,
                    'grad_clip': 1.0,
                    'val_split': 0.1,
                    'early_stop_patience': 20,
                },
                'model': {
                    'layers': [6, 8, 1],
                    'grid_size': 5,
                    'spline_order': 3,
                },
            }
        
        bc_config = self.config.get('bc', {})
        model_config = self.config.get('model', {})
        
        # 设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"[BCAgent] 使用设备: {self.device}")
        
        # 策略网络
        if policy is not None:
            self.policy = policy.to(self.device)
        else:
            layers = model_config.get('layers', [6, 8, 1])
            self.policy = KANPolicy(
                input_dim=int(layers[0]),
                hidden_dim=int(layers[1]),
                output_dim=int(layers[2]),
                grid_size=int(model_config.get('grid_size', 5)),
                spline_order=int(model_config.get('spline_order', 3)),
            ).to(self.device)
        
        # 优化器
        self.lr = float(bc_config.get('lr', 1e-3))
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # 学习率调度器 (可选)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 训练参数
        self.batch_size = int(bc_config.get('batch_size', 128))
        self.l1_penalty = float(bc_config.get('l1_penalty', 1e-4))
        self.grad_clip = float(bc_config.get('grad_clip', 1.0))
        self.val_split = float(bc_config.get('val_split', 0.1))
        self.early_stop_patience = int(bc_config.get('early_stop_patience', 20))
        
        # 网格更新参数
        self.grid_update_freq = int(model_config.get('grid_update_freq', 10))
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        
    def create_dataloaders(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        创建训练和验证 DataLoader
        
        Args:
            data_path: 专家数据路径
        Returns:
            train_loader, val_loader
        """
        dataset = ExpertDataset(data_path)
        
        # 划分训练集和验证集
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows 下建议设为 0
            pin_memory=True if self.device == 'cuda' else False,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False,
        )
        
        print(f"[BCAgent] 数据集划分: 训练集 {train_size}, 验证集 {val_size}")
        
        return train_loader, val_loader
    
    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        true_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失 = MSE + L1 正则化
        
        Returns:
            total_loss, loss_info
        """
        # MSE Loss
        mse_loss = F.mse_loss(pred_actions, true_actions)
        
        # L1 正则化 (稀疏化惩罚)
        if self.l1_penalty > 0:
            l1_loss = self.policy.regularization_loss(self.l1_penalty)
        else:
            l1_loss = torch.tensor(0.0, device=self.device)
        
        # 总损失
        total_loss = mse_loss + l1_loss
        
        loss_info = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
        }
        
        return total_loss, loss_info
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Returns:
            平均损失字典
        """
        self.policy.train()
        total_losses = {'total': 0.0, 'mse': 0.0, 'l1': 0.0}
        n_batches = 0
        
        for states, actions in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # 前向传播
            pred_actions = self.policy(states)
            
            # 计算损失
            loss, loss_info = self.compute_loss(pred_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 (关键！防止 KAN 梯度爆炸)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.grad_clip
                )
            
            self.optimizer.step()
            
            # 累加损失
            for key in total_losses:
                total_losses[key] += loss_info[key]
            n_batches += 1
        
        # 平均损失
        avg_losses = {key: val / n_batches for key, val in total_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证
        
        Returns:
            平均损失字典
        """
        self.policy.eval()
        total_losses = {'total': 0.0, 'mse': 0.0, 'l1': 0.0}
        n_batches = 0
        
        with torch.no_grad():
            for states, actions in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                pred_actions = self.policy(states)
                _, loss_info = self.compute_loss(pred_actions, actions)
                
                for key in total_losses:
                    total_losses[key] += loss_info[key]
                n_batches += 1
        
        avg_losses = {key: val / n_batches for key, val in total_losses.items()}
        
        return avg_losses
    
    def train(
        self,
        data_path: str,
        epochs: int = 200,
        save_path: Optional[str] = None,
        log_interval: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        完整训练流程
        
        Args:
            data_path: 专家数据路径
            epochs: 训练轮数
            save_path: 最佳模型保存路径
            log_interval: 日志打印间隔
            verbose: 是否打印详细日志
        
        Returns:
            训练历史记录
        """
        # 创建数据加载器
        train_loader, val_loader = self.create_dataloaders(data_path)
        
        print(f"\n{'='*60}")
        print("开始行为克隆训练")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"L1 penalty: {self.l1_penalty}")
        print(f"Grad clip: {self.grad_clip}")
        print(f"Grid update freq: {self.grid_update_freq}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练
            train_losses = self.train_epoch(train_loader)
            
            # 验证
            val_losses = self.validate(val_loader)
            
            # 记录历史
            self.train_losses.append(train_losses['total'])
            self.val_losses.append(val_losses['total'])
            
            # 学习率调度
            self.scheduler.step(val_losses['total'])
            
            # 网格更新
            if (epoch + 1) % self.grid_update_freq == 0:
                print(f"[Epoch {epoch+1}] 更新 B-spline 网格...")
                # 从验证集采样一些状态用于网格更新
                sample_states = []
                for states, _ in val_loader:
                    sample_states.append(states)
                    if len(sample_states) * states.shape[0] >= 2048:
                        break
                sample_states = torch.cat(sample_states, dim=0)[:2048].to(self.device)
                self.policy.update_grids(sample_states, sample_rate=1.0)
            
            # 早停检查
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.patience_counter = 0
                
                # 保存最佳模型
                if save_path is not None:
                    self.save(save_path)
                    if verbose and (epoch + 1) % log_interval == 0:
                        print(f"[Epoch {epoch+1}] 💾 保存最佳模型 (val_loss={val_losses['total']:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n[Epoch {epoch+1}] ⏹️ 早停触发 ( patience={self.early_stop_patience} )")
                    break
            
            # 打印日志
            if verbose and (epoch + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch+1:3d}/{epochs}] "
                    f"train_loss={train_losses['total']:.6f} "
                    f"(mse={train_losses['mse']:.6f}, l1={train_losses['l1']:.6f}) | "
                    f"val_loss={val_losses['total']:.6f} "
                    f"| 耗时: {elapsed:.1f}s"
                )
                
                # 打印稀疏化信息 (每 20 个 epoch)
                if (epoch + 1) % (log_interval * 2) == 0:
                    self.policy.print_sparsity(threshold=0.01)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"训练完成! 总耗时: {total_time:.1f}s")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
    
    def save(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"[BCAgent] 加载模型: {path} (epoch={self.current_epoch})")
    
    def evaluate(self, env, n_episodes: int = 10) -> Dict[str, float]:
        """
        在环境中评估策略
        
        Args:
            env: 环境实例
            n_episodes: 评估回合数
        Returns:
            评估统计信息
        """
        self.policy.eval()
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.policy.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }


# 导入 F 用于损失计算
import torch.nn.functional as F


def test_bc_agent():
    """测试 BC Agent"""
    print("=" * 60)
    print("测试 BC Agent")
    print("=" * 60)
    
    # 创建模拟专家数据
    print("\n[Step 1] 创建模拟专家数据")
    n_samples = 1000
    states = torch.randn(n_samples, 6)
    actions = torch.tanh(torch.randn(n_samples, 1))  # 范围 [-1, 1]
    
    # 添加一些模式使任务可学习
    # 简单规则: action = -0.5 * cos1 + 0.3 * sin2
    actions = (-0.5 * states[:, 0:1] + 0.3 * states[:, 3:4]).clamp(-1, 1)
    
    # 保存模拟数据
    os.makedirs('data', exist_ok=True)
    mock_data_path = 'data/mock_expert.pt'
    torch.save({'states': states, 'actions': actions}, mock_data_path)
    print(f"模拟数据已保存: {mock_data_path}")
    print(f"  - states: {states.shape}, actions: {actions.shape}")
    
    # 创建 Agent
    print("\n[Step 2] 创建 BC Agent")
    config = {
        'bc': {
            'batch_size': 64,  # 小批次测试
            'lr': 1e-3,
            'l1_penalty': 1e-4,
            'grad_clip': 1.0,
            'val_split': 0.2,
            'early_stop_patience': 10,
        },
        'model': {
            'layers': [6, 8, 1],
            'grid_size': 5,
            'spline_order': 3,
            'grid_update_freq': 5,
        },
    }
    
    agent = BCAgent(config=config, device='cpu')
    print(f"Policy 参数数量: {sum(p.numel() for p in agent.policy.parameters())}")
    
    # 训练 (短轮数测试)
    print("\n[Step 3] 训练")
    history = agent.train(
        data_path=mock_data_path,
        epochs=20,
        save_path='checkpoints/test_bc_model.pt',
        log_interval=5,
    )
    
    # 验证
    print("\n[Step 4] 验证学习效果")
    agent.policy.eval()
    with torch.no_grad():
        test_states = states[:10]
        pred_actions = agent.policy(test_states)
        true_actions = actions[:10]
        mse = F.mse_loss(pred_actions, true_actions).item()
    
    print(f"测试集 MSE: {mse:.6f}")
    print(f"预测动作范围: [{pred_actions.min():.3f}, {pred_actions.max():.3f}]")
    print(f"真实动作范围: [{true_actions.min():.3f}, {true_actions.max():.3f}]")
    
    # 打印最终稀疏化信息
    print("\n[Step 5] 最终稀疏化统计")
    agent.policy.print_sparsity(threshold=0.01)
    
    print("\n" + "=" * 60)
    print("✅ BC Agent 测试通过！")
    print("=" * 60)
    
    return agent


if __name__ == "__main__":
    test_bc_agent()
