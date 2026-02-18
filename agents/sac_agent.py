"""
SAC (Soft Actor-Critic) Agent
==============================
基于 KAN 网络的 SAC 实现

特性:
- 自适应熵调节 (Auto-tuning temperature)
- 双 Q 网络 + Target 网络
- 经验回放 (Experience Replay)
- 支持从 BC 冷启动

参考文献: Haarnoja et al. "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor" (2018)
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from collections import deque

from models.kan_policy import KANPolicy


class ReplayBuffer:
    """
    SAC 经验回放缓冲区
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        self.states = torch.zeros(buffer_size, state_dim, device=device)
        self.actions = torch.zeros(buffer_size, action_dim, device=device)
        self.rewards = torch.zeros(buffer_size, device=device)
        self.next_states = torch.zeros(buffer_size, state_dim, device=device)
        self.dones = torch.zeros(buffer_size, device=device)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.states[self.ptr] = torch.FloatTensor(state)
        self.actions[self.ptr] = torch.FloatTensor(action)
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = torch.FloatTensor(next_state)
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int) -> Tuple:
        """随机采样"""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
    
    def __len__(self):
        return self.size


class KANQNetwork(nn.Module):
    """
    KAN Q 网络
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 8,
        grid_size: int = 5,
        spline_order: int = 3,
    ):
        super().__init__()
        
        from models.kan_policy import KANLayer
        
        # Q(s, a) 输入是状态和动作的拼接
        input_dim = state_dim + action_dim
        
        self.layer1 = KANLayer(input_dim, hidden_dim, grid_size, spline_order)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer2 = KANLayer(hidden_dim, 1, grid_size, spline_order)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """返回 Q(s, a)"""
        x = torch.cat([state, action], dim=-1)
        x = self.layer1(x)
        x = self.layer_norm(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x.squeeze(-1)


class KANGaussianPolicy(nn.Module):
    """
    KAN 高斯策略 (用于 SAC)
    
    输出动作的高斯分布参数 (mean, log_std)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 8,
        grid_size: int = 5,
        spline_order: int = 3,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super().__init__()
        
        from models.kan_policy import KANLayer
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 共享特征层
        self.feature = KANLayer(state_dim, hidden_dim, grid_size, spline_order)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 均值头
        self.mean_layer = KANLayer(hidden_dim, action_dim, grid_size, spline_order)
        
        # 标准差头
        self.log_std_layer = KANLayer(hidden_dim, action_dim, grid_size, spline_order)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回动作的均值和对数标准差
        """
        x = self.feature(state)
        x = self.layer_norm(x)
        x = torch.relu(x)
        
        mean = self.mean_layer(x)
        mean = torch.tanh(mean)  # 限制在 [-1, 1]
        
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Returns:
            action, log_prob
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterized sample
        action = torch.tanh(x_t)
        
        # 计算 log_prob (考虑 tanh 的 Jacobian)
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)
        
        return action, log_prob
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """用于推理"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            mean, log_std = self.forward(state_tensor)
            
            if deterministic:
                action = mean
            else:
                std = log_std.exp()
                action = torch.tanh(mean + std * torch.randn_like(mean))
            
            return action.cpu().numpy()[0]


class SACAgent:
    """
    SAC Agent with KAN networks
    """
    
    def __init__(
        self,
        policy: Optional[KANGaussianPolicy] = None,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            policy: 策略网络
            config: 配置字典
            config_path: 配置文件路径
            device: 计算设备
        """
        # 加载配置
        if config is not None:
            self.config = config
        elif config_path is not None:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        sac_config = self.config.get('sac', {})
        model_config = self.config.get('model', {})
        
        # 设备
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SACAgent] Using device: {self.device}")
        
        # 网络结构参数
        state_dim = int(model_config.get('state_dim', 6))
        action_dim = int(model_config.get('action_dim', 1))
        hidden_dim = int(model_config.get('hidden_dim', 8))
        grid_size = int(model_config.get('grid_size', 5))
        spline_order = int(model_config.get('spline_order', 3))
        
        # 策略网络
        if policy is not None:
            self.policy = policy.to(self.device)
        else:
            self.policy = KANGaussianPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                grid_size=grid_size,
                spline_order=spline_order,
            ).to(self.device)
        
        # Q 网络 (双 Q)
        self.q1 = KANQNetwork(state_dim, action_dim, hidden_dim, grid_size, spline_order).to(self.device)
        self.q2 = KANQNetwork(state_dim, action_dim, hidden_dim, grid_size, spline_order).to(self.device)
        
        # Target Q 网络
        self.q1_target = KANQNetwork(state_dim, action_dim, hidden_dim, grid_size, spline_order).to(self.device)
        self.q2_target = KANQNetwork(state_dim, action_dim, hidden_dim, grid_size, spline_order).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # 优化器
        self.lr = float(sac_config.get('lr', 3e-4))
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr)
        
        # 自适应熵温度
        self.automatic_entropy_tuning = sac_config.get('automatic_entropy_tuning', True)
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_dim  # -dim(A)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = float(sac_config.get('alpha', 0.2))
        
        # SAC 超参数
        self.gamma = float(sac_config.get('gamma', 0.99))
        self.tau = float(sac_config.get('tau', 0.005))
        self.batch_size = int(sac_config.get('batch_size', 64))
        self.buffer_size = int(sac_config.get('buffer_size', 100000))
        self.warmup_steps = int(sac_config.get('warmup_steps', 1000))
        self.update_every = int(sac_config.get('update_every', 1))
        self.l1_penalty = float(sac_config.get('l1_penalty', 1e-5))
        self.grad_clip = float(sac_config.get('grad_clip', 0.5))
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )
        
        # 训练状态
        self.total_steps = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'sac': {
                'buffer_size': 100000,
                'batch_size': 64,
                'lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'automatic_entropy_tuning': True,
                'warmup_steps': 1000,
                'update_every': 1,
                'l1_penalty': 1e-5,
                'grad_clip': 0.5,
            },
            'model': {
                'state_dim': 6,
                'action_dim': 1,
                'hidden_dim': 8,
                'grid_size': 5,
                'spline_order': 3,
            },
        }
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """选择动作"""
        if self.total_steps < self.warmup_steps and not deterministic:
            # 随机动作预热
            return np.random.uniform(-1, 1, size=(1,))
        
        return self.policy.get_action(state, deterministic)
    
    def update(self) -> Dict[str, float]:
        """更新网络"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # ==================== 更新 Q 网络 ====================
        with torch.no_grad():
            # 采样下一个动作
            next_actions, next_log_probs = self.policy.sample(next_states)
            
            # Target Q 值
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next
        
        # Q1 损失
        q1_pred = self.q1(states, actions)
        q1_loss = F.mse_loss(q1_pred, q_target)
        
        # Q2 损失
        q2_pred = self.q2(states, actions)
        q2_loss = F.mse_loss(q2_pred, q_target)
        
        # 更新 Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.grad_clip)
        self.q1_optimizer.step()
        
        # 更新 Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.grad_clip)
        self.q2_optimizer.step()
        
        # ==================== 更新策略 ====================
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        # L1 正则化
        l1_loss = sum(torch.abs(p).mean() for p in self.policy.parameters())
        policy_loss += self.l1_penalty * l1_loss
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()
        
        # ==================== 更新温度 alpha ====================
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_value = self.alpha.item()
        else:
            alpha_value = self.alpha
        
        # ==================== 软更新 Target 网络 ====================
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': alpha_value,
        }
    
    def train(
        self,
        env,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None,
        log_interval: int = 1000,
    ) -> Dict:
        """训练"""
        print(f"\n{'='*60}")
        print("Start SAC Training")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Buffer size: {self.buffer_size}")
        print(f"Batch size: {self.batch_size}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        episode_reward = 0
        episode_length = 0
        obs, _ = env.reset()
        
        while self.total_steps < total_timesteps:
            # 选择动作
            action = self.select_action(obs, deterministic=False)
            
            # 执行动作
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # 存储经验
            self.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
            
            obs = next_obs
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # 更新网络
            if self.total_steps >= self.warmup_steps and self.total_steps % self.update_every == 0:
                update_info = self.update()
            else:
                update_info = {}
            
            # 回合结束
            if terminated or truncated:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # 日志
                if len(self.episode_rewards) % 10 == 0:
                    elapsed = time.time() - start_time
                    recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                    mean_reward = np.mean(recent_rewards)
                    
                    info_str = f"[Step {self.total_steps}/{total_timesteps}] Reward: {mean_reward:.1f}"
                    if update_info:
                        info_str += f" | Q1: {update_info.get('q1_loss', 0):.3f} | Policy: {update_info.get('policy_loss', 0):.3f} | Alpha: {update_info.get('alpha', 0):.3f}"
                    info_str += f" | FPS: {self.total_steps / elapsed:.0f}"
                    print(info_str)
                
                episode_reward = 0
                episode_length = 0
                obs, _ = env.reset()
        
        if save_path:
            self.save(save_path)
            print(f"\nModel saved: {save_path}")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed! Total time: {total_time:.1f}s")
        print(f"{'='*60}\n")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
        }
    
    def save(self, path: str):
        """保存模型"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'total_steps': self.total_steps,
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        print(f"[SACAgent] Model loaded: {path}")


if __name__ == "__main__":
    print("SAC Agent module - run scripts/4_train_sac.py for training")
