"""
环境包装器 (Environment Wrapper)
================================
用于双倒立摆 (Acrobot) 连续控制任务

核心功能:
1. 特征解耦: 将角度转换为三角函数连续特征
2. 动作空间适配: 将离散动作转换为连续扭矩
3. 状态归一化: 可选的状态标准化
4. 奖励塑形: 可选的奖励工程

注意: Gymnasium Acrobot-v1 原生输出已经是三角函数形式:
  [cos(theta1), sin(theta1), cos(theta2), sin(theta2), dot_theta1, dot_theta2]
本 Wrapper 主要用于适配连续控制接口和可能的预处理。
"""

import gymnasium as gym
import numpy as np
from typing import Tuple, Optional, Dict, Any


class ContinuousAcrobotWrapper(gym.Wrapper):
    """
    Acrobot 连续控制包装器
    
    将 Gymnasium 的离散动作 (3个: -1, 0, +1) 映射到连续动作空间 [-1, 1]
    同时提供状态预处理接口。
    
    原始环境:
    - 观测空间: Box([-1, -1, -1, -1, -4π, -9π], [1, 1, 1, 1, 4π, 9π], (6,), float32)
    - 动作空间: Discrete(3) -> {0: -1.0, 1: 0.0, 2: +1.0}
    
    包装后:
    - 观测空间: Box([-1, -1, -1, -1, -4π, -9π], [1, 1, 1, 1, 4π, 9π], (6,), float32) (可选归一化)
    - 动作空间: Box([-1.0], [1.0], (1,), float32)
    """
    
    def __init__(
        self,
        env: gym.Env,
        normalize_obs: bool = False,
        obs_norm_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            env: 原始 Acrobot 环境
            normalize_obs: 是否对观测进行归一化
            obs_norm_stats: 归一化统计信息 {'mean': ..., 'std': ...}
        """
        super().__init__(env)
        self.normalize_obs = normalize_obs
        
        # 动作空间转换为连续 [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        # 观测空间保持不变 (已经是三角函数形式)
        # 但我们可以选择是否归一化
        self.observation_space = env.observation_space
        
        # 归一化统计信息
        if obs_norm_stats is not None:
            self.obs_mean = obs_norm_stats['mean']
            self.obs_std = obs_norm_stats['std']
        else:
            # Acrobot 的默认范围估计
            # [cos, sin, cos, sin, dot1, dot2]
            self.obs_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            self.obs_std = np.array([1.0, 1.0, 1.0, 1.0, 6.0, 10.0], dtype=np.float32)
        
        # 记录轨迹统计
        self.episode_step = 0
        self.episode_reward = 0.0
        
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        self.episode_step = 0
        self.episode_reward = 0.0
        
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 连续动作，范围 [-1, 1]，形状 (1,) 或标量
        Returns:
            obs, reward, terminated, truncated, info
        """
        # 确保动作是标量或正确的形状
        if isinstance(action, np.ndarray):
            action_value = float(np.clip(action[0], -1.0, 1.0))
        else:
            action_value = float(np.clip(action, -1.0, 1.0))
        
        # 将连续动作 [-1, 1] 映射到离散动作 {0, 1, 2}
        # -1.0 -> 0, 0.0 -> 1, +1.0 -> 2
        # 使用线性映射: discrete = int((continuous + 1) * 1.0)
        discrete_action = int((action_value + 1.0))
        discrete_action = np.clip(discrete_action, 0, 2)
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(discrete_action)
        
        # 状态处理
        if self.normalize_obs:
            obs = self._normalize_obs(obs)
        
        # 记录统计
        self.episode_step += 1
        self.episode_reward += reward
        
        info['action_continuous'] = action_value
        info['action_discrete'] = discrete_action
        
        return obs, reward, terminated, truncated, info
    
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """归一化观测"""
        return (obs - self.obs_mean) / (self.obs_std + 1e-8)
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """获取当前回合统计"""
        return {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
        }


class AngleFeatureWrapper(gym.ObservationWrapper):
    """
    角度特征转换包装器
    
    对于某些双摆环境，如果原始输出是角度而非三角函数，
    使用此 Wrapper 将其转换为 [sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), θ̇₁, θ̇₂]
    
    Gymnasium Acrobot-v1 原生就是三角函数形式，所以此 Wrapper 主要作为备用。
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        
        # 假设原始观测是 [theta1, theta2, dot_theta1, dot_theta2]
        # 转换为 [sin1, cos1, sin2, cos2, dot1, dot2]
        original_low = env.observation_space.low
        original_high = env.observation_space.high
        
        # 新的观测空间: sin/cos 在 [-1, 1]，角速度保持不变
        new_low = np.array([-1.0, -1.0, -1.0, -1.0, original_low[2], original_low[3]], dtype=np.float32)
        new_high = np.array([1.0, 1.0, 1.0, 1.0, original_high[2], original_high[3]], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(
            low=new_low,
            high=new_high,
            dtype=np.float32
        )
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        转换观测: [theta1, theta2, dot1, dot2] -> [sin1, cos1, sin2, cos2, dot1, dot2]
        """
        theta1, theta2, dot1, dot2 = obs[0], obs[1], obs[2], obs[3]
        
        return np.array([
            np.sin(theta1),
            np.cos(theta1),
            np.sin(theta2),
            np.cos(theta2),
            dot1,
            dot2
        ], dtype=np.float32)


class RewardShapingWrapper(gym.Wrapper):
    """
    奖励塑形包装器 (可选)
    
    对原始奖励进行变换，可能有助于训练:
    1. 角度惩罚: 惩罚偏离直立位置的角度
    2. 角速度惩罚: 惩罚过大的角速度 (能量节省)
    3. 存活奖励: 每步给予小正奖励
    """
    
    def __init__(
        self,
        env: gym.Env,
        angle_penalty_coef: float = 0.0,
        velocity_penalty_coef: float = 0.0,
        alive_bonus: float = 0.0,
    ):
        """
        Args:
            angle_penalty_coef: 角度惩罚系数
            velocity_penalty_coef: 角速度惩罚系数
            alive_bonus: 每步存活奖励
        """
        super().__init__(env)
        self.angle_penalty_coef = angle_penalty_coef
        self.velocity_penalty_coef = velocity_penalty_coef
        self.alive_bonus = alive_bonus
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 原始观测 (假设是三角函数形式)
        # obs = [cos1, sin1, cos2, sin2, dot1, dot2]
        cos1, sin1, cos2, sin2 = obs[0], obs[1], obs[2], obs[3]
        dot1, dot2 = obs[4], obs[5]
        
        # 计算角度 (用于惩罚)
        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)
        
        # 奖励塑形
        shaped_reward = reward
        
        # 角度惩罚 (惩罚远离 [0, 0] 的状态，即双摆下垂)
        if self.angle_penalty_coef > 0:
            angle_penalty = -(theta1**2 + theta2**2) * self.angle_penalty_coef
            shaped_reward += angle_penalty
            info['angle_penalty'] = angle_penalty
        
        # 角速度惩罚 (能量节省)
        if self.velocity_penalty_coef > 0:
            velocity_penalty = -(dot1**2 + dot2**2) * self.velocity_penalty_coef
            shaped_reward += velocity_penalty
            info['velocity_penalty'] = velocity_penalty
        
        # 存活奖励
        if self.alive_bonus > 0:
            shaped_reward += self.alive_bonus
            info['alive_bonus'] = self.alive_bonus
        
        info['original_reward'] = reward
        info['shaped_reward'] = shaped_reward
        
        return obs, shaped_reward, terminated, truncated, info


def make_acrobot_env(
    continuous: bool = True,
    normalize_obs: bool = False,
    angle_transform: bool = False,
    reward_shaping: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> gym.Env:
    """
    创建 Acrobot 环境的工厂函数
    
    Args:
        continuous: 是否使用连续动作空间
        normalize_obs: 是否归一化观测
        angle_transform: 是否进行角度到三角函数的转换 (Acrobot-v1 不需要)
        reward_shaping: 奖励塑形参数字典 {'angle_penalty_coef': ..., ...}
        seed: 随机种子
    
    Returns:
        包装后的环境
    """
    # 创建基础环境
    env = gym.make('Acrobot-v1')
    
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    
    # 角度特征转换 (通常不需要，因为 Acrobot-v1 原生就是三角函数)
    if angle_transform:
        env = AngleFeatureWrapper(env)
    
    # 奖励塑形 (可选)
    if reward_shaping is not None:
        env = RewardShapingWrapper(env, **reward_shaping)
    
    # 连续控制适配
    if continuous:
        env = ContinuousAcrobotWrapper(env, normalize_obs=normalize_obs)
    
    return env


def compute_obs_stats(env: gym.Env, n_samples: int = 10000) -> Dict[str, np.ndarray]:
    """
    计算观测的均值和标准差 (用于归一化)
    
    Args:
        env: 环境实例
        n_samples: 采样步数
    Returns:
        {'mean': np.ndarray, 'std': np.ndarray}
    """
    observations = []
    obs, _ = env.reset()
    
    for _ in range(n_samples):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    observations = np.array(observations)
    mean = observations.mean(axis=0)
    std = observations.std(axis=0)
    
    return {'mean': mean, 'std': std}


def test_env_wrapper():
    """测试环境包装器"""
    print("=" * 60)
    print("测试环境包装器")
    print("=" * 60)
    
    # 测试 1: 连续动作环境
    print("\n[Test 1] 连续动作环境")
    env = make_acrobot_env(continuous=True, seed=42)
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    obs, info = env.reset()
    print(f"初始观测: {obs}")
    print(f"观测形状: {obs.shape}")
    
    # 测试连续动作
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action[0]:+.3f} -> discrete={info['action_discrete']}, reward={reward:.3f}")
    
    env.close()
    
    # 测试 2: 归一化环境
    print("\n[Test 2] 归一化观测环境")
    env = make_acrobot_env(continuous=True, normalize_obs=True, seed=42)
    obs, _ = env.reset()
    print(f"归一化后观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
    env.close()
    
    # 测试 3: 奖励塑形
    print("\n[Test 3] 奖励塑形环境")
    reward_shaping = {
        'angle_penalty_coef': 0.01,
        'velocity_penalty_coef': 0.001,
        'alive_bonus': 0.1,
    }
    env = make_acrobot_env(continuous=True, reward_shaping=reward_shaping, seed=42)
    obs, _ = env.reset()
    action = np.array([0.5])
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"原始奖励: {info['original_reward']:.3f}")
    print(f"塑形后奖励: {info['shaped_reward']:.3f}")
    print(f"  - 角度惩罚: {info.get('angle_penalty', 0):.3f}")
    print(f"  - 速度惩罚: {info.get('velocity_penalty', 0):.3f}")
    print(f"  - 存活奖励: {info.get('alive_bonus', 0):.3f}")
    env.close()
    
    # 测试 4: 与 KAN Policy 集成
    print("\n[Test 4] 与 KAN Policy 集成测试")
    from models.kan_policy import KANPolicy
    
    env = make_acrobot_env(continuous=True, seed=42)
    policy = KANPolicy(input_dim=6, hidden_dim=8, output_dim=1, action_scale=1.0)
    
    obs, _ = env.reset()
    total_reward = 0
    
    for step in range(100):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode finished at step {step+1}, total_reward={total_reward:.2f}")
            break
    else:
        print(f"Completed 100 steps, total_reward={total_reward:.2f}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("✅ 所有环境测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_env_wrapper()
