"""
策略评估脚本 (evaluate.py)
==========================
在环境中测试训练好的 KAN 策略

使用方法:
  # 评估 BC 策略
  python scripts/evaluate.py --model checkpoints/bc_kan_model.pt

  # 渲染可视化 (慢)
  python scripts/evaluate.py --model checkpoints/bc_kan_model.pt --render
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.kan_policy import KANPolicy
from envs.wrapper import make_acrobot_env


def evaluate_policy(
    policy,
    env,
    n_episodes: int = 100,
    max_steps: int = 500,
    render: bool = False,
):
    """
    评估策略在环境中的表现
    
    Returns:
        {
            'episode_rewards': List[float],
            'episode_lengths': List[int],
            'mean_reward': float,
            'std_reward': float,
            'mean_length': float,
            'success_rate': float,  # 成功完成回合的比例
        }
    """
    policy.eval()
    episode_rewards = []
    episode_lengths = []
    
    pbar = tqdm(range(n_episodes), desc="评估中")
    
    for ep in pbar:
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            if render:
                env.render()
            
            with torch.no_grad():
                action = policy.get_action(obs)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        pbar.set_postfix({
            'avg_reward': np.mean(episode_rewards),
            'avg_length': np.mean(episode_lengths),
        })
    
    pbar.close()
    
    # 计算统计信息
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    # 成功定义: 在少于 500 步内完成 (即奖励 > -500)
    success_rate = sum(1 for r in episode_rewards if r > -500) / n_episodes
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_length': mean_length,
        'success_rate': success_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="评估 KAN 策略")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='模型路径 (如 checkpoints/bc_kan_model.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--n_episodes',
        type=int,
        default=100,
        help='评估回合数'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='是否渲染环境 (慢)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建环境
    print("\n创建环境...")
    env = make_acrobot_env(continuous=True, seed=args.seed)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model_config = config.get('model', {})
    layers = model_config.get('layers', [6, 8, 1])
    
    policy = KANPolicy(
        input_dim=int(layers[0]),
        hidden_dim=int(layers[1]),
        output_dim=int(layers[2]),
        grid_size=int(model_config.get('grid_size', 5)),
        spline_order=int(model_config.get('spline_order', 3)),
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(args.model, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    print(f"模型加载完成 (训练 epoch: {checkpoint.get('epoch', 'unknown')})")
    
    # 打印模型信息
    print("\n模型信息:")
    policy.print_sparsity(threshold=0.01)
    
    # 评估
    print(f"\n开始评估 ({args.n_episodes} 回合)...")
    print("-" * 60)
    
    results = evaluate_policy(
        policy=policy,
        env=env,
        n_episodes=args.n_episodes,
        render=args.render,
    )
    
    env.close()
    
    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"平均回合奖励: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"平均回合长度: {results['mean_length']:.1f} 步")
    print(f"成功率: {results['success_rate']:.1%}")
    print("-" * 60)
    print(f"最短回合: {min(results['episode_lengths'])} 步")
    print(f"最长回合: {max(results['episode_lengths'])} 步")
    print(f"最佳奖励: {max(results['episode_rewards']):.1f}")
    print(f"最差奖励: {min(results['episode_rewards']):.1f}")
    print("=" * 60)
    
    # 与专家数据对比
    expert_data_path = config.get('expert', {}).get('save_path', 'data/expert_trajectories.pt')
    if os.path.exists(expert_data_path):
        expert_data = torch.load(expert_data_path)
        expert_rewards = expert_data.get('episode_rewards', [])
        if expert_rewards:
            print(f"\n与专家对比:")
            print(f"  专家平均奖励: {np.mean(expert_rewards):.1f} ± {np.std(expert_rewards):.1f}")
            print(f"  当前平均奖励: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
            print(f"  性能比例: {results['mean_reward'] / np.mean(expert_rewards):.1%}")
    
    print("\n✅ 评估完成!")


if __name__ == "__main__":
    main()
