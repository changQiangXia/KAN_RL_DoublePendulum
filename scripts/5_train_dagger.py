"""
DAgger 训练脚本 (5_train_dagger.py)
===================================
Dataset Aggregation: 在线模仿学习

在训练过程中让策略与环境交互，遇到新状态时请专家标注，
逐步扩展数据集，解决分布偏移问题。

使用方法:
  python scripts/5_train_dagger.py --expert_algorithm heuristic --n_iterations 10
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

from agents.bc_agent import BCAgent, ExpertDataset
from envs.wrapper import make_acrobot_env
from utils.experts import HeuristicExpert, RandomExpert


def collect_rollout(env, policy, expert, n_steps: int = 1000):
    """
    使用当前策略收集轨迹，并由专家标注动作
    
    Returns:
        states, expert_actions
    """
    states = []
    expert_actions = []
    
    obs, _ = env.reset()
    for _ in range(n_steps):
        # 使用当前策略选择动作
        action = policy.get_action(obs)
        
        # 记录状态和专家动作
        states.append(obs.copy())
        expert_action = expert.get_action(obs)
        expert_actions.append(expert_action.copy())
        
        # 执行策略动作
        obs, _, terminated, truncated, _ = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    return np.array(states, dtype=np.float32), np.array(expert_actions, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train KAN-DAgger")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--expert_algorithm',
        type=str,
        default='heuristic',
        choices=['heuristic', 'random'],
        help='Expert algorithm type'
    )
    parser.add_argument(
        '--n_iterations',
        type=int,
        default=10,
        help='Number of DAgger iterations'
    )
    parser.add_argument(
        '--steps_per_iter',
        type=int,
        default=10000,
        help='Steps to collect per iteration'
    )
    parser.add_argument(
        '--bc_epochs',
        type=int,
        default=50,
        help='BC training epochs per iteration'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
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
    print(f"Using device: {device}")
    
    # 创建环境
    print("\nCreating environment...")
    env = make_acrobot_env(continuous=True, seed=args.seed)
    
    # 创建专家
    print(f"Creating expert: {args.expert_algorithm}")
    if args.expert_algorithm == 'heuristic':
        expert = HeuristicExpert()
    else:
        expert = RandomExpert()
    expert.eval()
    
    # 初始化数据集 (可选: 从现有专家数据加载)
    print("\n" + "=" * 60)
    print("DAgger Training")
    print("=" * 60)
    print(f"Iterations: {args.n_iterations}")
    print(f"Steps per iteration: {args.steps_per_iter}")
    print(f"BC epochs per iteration: {args.bc_epochs}")
    print("=" * 60 + "\n")
    
    # 创建初始数据集
    all_states = []
    all_actions = []
    
    # 迭代 DAgger
    for iteration in range(args.n_iterations):
        print(f"\n[DAgger Iteration {iteration + 1}/{args.n_iterations}]")
        print("-" * 60)
        
        # 收集新数据
        print("Collecting rollout with current policy...")
        if iteration == 0:
            # 第一次迭代: 随机策略
            states = []
            actions = []
            obs, _ = env.reset()
            for _ in range(args.steps_per_iter):
                action = expert.get_action(obs)
                states.append(obs.copy())
                actions.append(action.copy())
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()
            states = np.array(states, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
        else:
            # 后续迭代: 使用当前策略
            states, actions = collect_rollout(env, agent.policy, expert, args.steps_per_iter)
        
        print(f"Collected {len(states)} new state-action pairs")
        
        # 添加到数据集
        all_states.append(states)
        all_actions.append(actions)
        
        # 合并数据集
        combined_states = np.concatenate(all_states, axis=0)
        combined_actions = np.concatenate(all_actions, axis=0)
        
        print(f"Total dataset size: {len(combined_states)}")
        
        # 保存临时数据
        temp_data_path = 'data/dagger_temp.pt'
        os.makedirs('data', exist_ok=True)
        torch.save({
            'states': torch.FloatTensor(combined_states),
            'actions': torch.FloatTensor(combined_actions),
        }, temp_data_path)
        
        # 创建/更新 BC Agent
        if iteration == 0:
            agent = BCAgent(config=config, device=device)
        
        # 训练 BC
        print(f"\nTraining BC for {args.bc_epochs} epochs...")
        agent.train(
            data_path=temp_data_path,
            epochs=args.bc_epochs,
            save_path=f'checkpoints/dagger_iter_{iteration + 1}.pt',
            log_interval=max(1, args.bc_epochs // 5),
            verbose=True,
        )
        
        # 评估
        print(f"\nEvaluating...")
        episode_rewards = []
        for _ in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.policy.get_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_rewards.append(episode_reward)
        
        mean_reward = np.mean(episode_rewards)
        print(f"Mean reward: {mean_reward:.1f}")
        
        # 稀疏化统计
        agent.policy.print_sparsity(threshold=0.01)
    
    env.close()
    
    # 保存最终模型
    final_path = 'checkpoints/dagger_kan_model.pt'
    agent.save(final_path)
    
    # 最终评估
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    env = make_acrobot_env(continuous=True, seed=args.seed)
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(50):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        while not done:
            action = agent.policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    env.close()
    
    print(f"Mean reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f}")
    print(f"Success rate: {sum(1 for r in episode_rewards if r > -500) / len(episode_rewards):.1%}")
    print("=" * 60)
    print(f"\nFinal model saved: {final_path}")


if __name__ == "__main__":
    main()
