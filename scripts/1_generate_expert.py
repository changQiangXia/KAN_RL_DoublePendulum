"""
专家数据生成脚本 (1_generate_expert.py)
=========================================
生成用于行为克隆 (BC) 训练的专家轨迹

支持的专家策略:
1. random: 随机策略 (用于测试)
2. heuristic: 启发式策略 (基于能量和角度)
3. mlp_ppo: 预训练的 MLP-PPO 模型 (需先训练)

输出格式:
- data/expert_trajectories.pt
  {
    'states': torch.Tensor (N, 6),
    'actions': torch.Tensor (N, 1),
    'rewards': torch.Tensor (N,),
    'episode_lengths': List[int],
  }

使用方法:
  python scripts/1_generate_expert.py --algorithm heuristic --n_trajectories 1000
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.wrapper import make_acrobot_env
from utils.experts import RandomExpert as ImportedRandomExpert, HeuristicExpert as ImportedHeuristicExpert


# 保留本地定义以保持向后兼容
class RandomExpert:
    """随机专家策略 (用于测试 BC 流程)"""
    
    def __init__(self, action_dim: int = 1):
        self.action_dim = action_dim
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """返回随机动作 [-1, 1]"""
        return np.random.uniform(-1, 1, size=(self.action_dim,))
    
    def eval(self):
        pass
    
    def __call__(self, state):
        """使其可调用"""
        return self.get_action(state)


class HeuristicExpert:
    """
    启发式专家策略 (基于物理直觉)
    
    策略逻辑:
    - 如果下摆角度大，施加扭矩使其回正
    - 考虑角速度进行阻尼控制
    - 类似一个简单的 PD 控制器
    
    状态: [cos1, sin1, cos2, sin2, dot1, dot2]
    """
    
    def __init__(
        self,
        kp1: float = 2.0,   # 第一摆角度比例增益
        kd1: float = 0.5,   # 第一摆角速度阻尼
        kp2: float = 4.0,   # 第二摆角度比例增益
        kd2: float = 0.3,   # 第二摆角速度阻尼
    ):
        self.kp1 = kp1
        self.kd1 = kd1
        self.kp2 = kp2
        self.kd2 = kd2
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        根据状态计算动作
        
        目标: 将双摆摆动到直立位置 (cos1≈1, sin1≈0, cos2≈1, sin2≈0)
        """
        cos1, sin1, cos2, sin2, dot1, dot2 = state
        
        # 计算角度 (从三角函数恢复)
        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)
        
        # PD 控制: 目标是 theta1=0, theta2=0 (直立)
        # 动作 = -kp * theta - kd * dot
        action = (
            -self.kp1 * theta1 - self.kd1 * dot1
            - self.kp2 * theta2 - self.kd2 * dot2
        )
        
        # 限制到 [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        
        return np.array([action], dtype=np.float32)
    
    def eval(self):
        pass
    
    def __call__(self, state):
        """使其可调用"""
        return self.get_action(state)


class MLPPPOExpert:
    """
    预训练 MLP-PPO 专家 (占位实现)
    
    需要先训练一个 MLP-PPO 模型，然后加载
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: 加载 MLP-PPO 模型
        # self.model = load_model(model_path)
        raise NotImplementedError(
            "MLP-PPO 专家需要预训练模型。"
            "请先训练 MLP-PPO 或选择其他专家类型。"
        )
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        # with torch.no_grad():
        #     action = self.model(state)
        pass
    
    def eval(self):
        pass


def collect_trajectories(
    env,
    expert,
    n_trajectories: int = 1000,
    max_steps: int = 500,
    render: bool = False,
) -> Dict:
    """
    收集专家轨迹
    
    Args:
        env: 环境实例
        expert: 专家策略
        n_trajectories: 收集轨迹数量
        max_steps: 每条轨迹最大步数
        render: 是否渲染 (仅适用于有 GUI 的环境)
    
    Returns:
        {
            'states': np.ndarray (N, 6),
            'actions': np.ndarray (N, 1),
            'rewards': np.ndarray (N,),
            'episode_lengths': List[int],
            'episode_rewards': List[float],
        }
    """
    all_states = []
    all_actions = []
    all_rewards = []
    episode_lengths = []
    episode_rewards = []
    
    pbar = tqdm(total=n_trajectories, desc="收集轨迹")
    
    for ep in range(n_trajectories):
        obs, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            # 获取专家动作
            action = expert.get_action(obs)
            
            # 存储状态-动作对
            all_states.append(obs.copy())
            all_actions.append(action.copy())
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            all_rewards.append(reward)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        episode_lengths.append(step + 1)
        episode_rewards.append(episode_reward)
        
        pbar.update(1)
        pbar.set_postfix({
            'avg_reward': np.mean(episode_rewards[-100:]),
            'avg_length': np.mean(episode_lengths[-100:]),
        })
    
    pbar.close()
    
    return {
        'states': np.array(all_states, dtype=np.float32),
        'actions': np.array(all_actions, dtype=np.float32),
        'rewards': np.array(all_rewards, dtype=np.float32),
        'episode_lengths': episode_lengths,
        'episode_rewards': episode_rewards,
    }


def main():
    parser = argparse.ArgumentParser(description="生成专家轨迹数据")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='heuristic',
        choices=['random', 'heuristic', 'mlp_ppo'],
        help='专家算法类型'
    )
    parser.add_argument(
        '--n_trajectories',
        type=int,
        default=1000,
        help='生成轨迹数量'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=500,
        help='每条轨迹最大步数'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出文件路径 (默认从 config 读取)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='是否渲染环境 (慢)'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print(f"警告: 配置文件 {args.config} 不存在，使用默认配置")
        config = {}
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建输出目录
    output_path = args.output or config.get('expert', {}).get('save_path', 'data/expert_trajectories.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建环境
    print(f"创建环境: Acrobot-v1 (连续控制)")
    env = make_acrobot_env(continuous=True, seed=args.seed)
    
    # 创建专家
    print(f"创建专家策略: {args.algorithm}")
    if args.algorithm == 'random':
        expert = RandomExpert(action_dim=1)
    elif args.algorithm == 'heuristic':
        expert = HeuristicExpert()
    elif args.algorithm == 'mlp_ppo':
        model_path = config.get('expert', {}).get('model_path', 'checkpoints/expert_mlp_ppo.pt')
        expert = MLPPPOExpert(model_path)
    else:
        raise ValueError(f"未知的专家算法: {args.algorithm}")
    
    expert.eval()
    
    # 收集轨迹
    print(f"\n开始收集 {args.n_trajectories} 条轨迹...")
    print(f"每条轨迹最大步数: {args.max_steps}")
    print("-" * 60)
    
    data = collect_trajectories(
        env=env,
        expert=expert,
        n_trajectories=args.n_trajectories,
        max_steps=args.max_steps,
        render=args.render,
    )
    
    env.close()
    
    # 统计信息
    print("\n" + "=" * 60)
    print("数据收集完成!")
    print("=" * 60)
    print(f"总样本数: {len(data['states'])}")
    print(f"平均回合长度: {np.mean(data['episode_lengths']):.1f} ± {np.std(data['episode_lengths']):.1f}")
    print(f"平均回合奖励: {np.mean(data['episode_rewards']):.1f} ± {np.std(data['episode_rewards']):.1f}")
    print(f"动作范围: [{data['actions'].min():.3f}, {data['actions'].max():.3f}]")
    print(f"奖励范围: [{data['rewards'].min():.3f}, {data['rewards'].max():.3f}]")
    
    # 保存数据
    data_to_save = {
        'states': torch.FloatTensor(data['states']),
        'actions': torch.FloatTensor(data['actions']),
        'rewards': torch.FloatTensor(data['rewards']),
        'episode_lengths': data['episode_lengths'],
        'episode_rewards': data['episode_rewards'],
        'algorithm': args.algorithm,
        'config': config,
    }
    
    torch.save(data_to_save, output_path)
    print(f"\n💾 数据已保存: {output_path}")
    print(f"   文件大小: {os.path.getsize(output_path) / 1024**2:.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
