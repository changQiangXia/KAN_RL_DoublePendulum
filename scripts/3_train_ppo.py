"""
PPO 训练脚本 (3_train_ppo.py)
==============================
使用 PPO 算法训练或微调 KAN 策略

支持从 BC checkpoint 冷启动:
  python scripts/3_train_ppo.py --bc_checkpoint checkpoints/bc_kan_model.pt

从头训练:
  python scripts/3_train_ppo.py

使用方法:
  python scripts/3_train_ppo.py --total_timesteps 100000 --log_interval 5
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent, KANValueNetwork
from models.kan_policy import KANPolicy
from envs.wrapper import make_acrobot_env


def main():
    parser = argparse.ArgumentParser(description="Train KAN-PPO")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Config file path'
    )
    parser.add_argument(
        '--bc_checkpoint',
        type=str,
        default=None,
        help='Load BC checkpoint for warm start'
    )
    parser.add_argument(
        '--ppo_checkpoint',
        type=str,
        default=None,
        help='Resume PPO training from checkpoint'
    )
    parser.add_argument(
        '--total_timesteps',
        type=int,
        default=None,
        help='Total training timesteps'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='Log interval (in updates)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"Loading config: {args.config}")
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config not found: {args.config}")
    
    # 命令行参数覆盖配置
    if args.total_timesteps:
        config['ppo']['total_timesteps'] = args.total_timesteps
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建环境
    print("\nCreating environment...")
    env = make_acrobot_env(continuous=True, seed=args.seed)
    
    # 确定是否从 checkpoint 加载
    policy = None
    value_net = None
    
    if args.ppo_checkpoint and os.path.exists(args.ppo_checkpoint):
        # 从 PPO checkpoint 恢复
        print(f"\nResuming PPO from: {args.ppo_checkpoint}")
        agent = PPOAgent(config=config, device=device)
        agent.load(args.ppo_checkpoint)
    
    elif args.bc_checkpoint and os.path.exists(args.bc_checkpoint):
        # 从 BC checkpoint 冷启动
        print(f"\nWarm starting from BC: {args.bc_checkpoint}")
        
        model_config = config.get('model', {})
        layers = model_config.get('layers', [6, 8, 1])
        
        # 创建策略网络并加载 BC 权重
        policy = KANPolicy(
            input_dim=int(layers[0]),
            hidden_dim=int(layers[1]),
            output_dim=int(layers[2]),
            grid_size=int(model_config.get('grid_size', 5)),
            spline_order=int(model_config.get('spline_order', 3)),
        )
        
        checkpoint = torch.load(args.bc_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        print(f"BC policy loaded (trained {checkpoint.get('epoch', 'unknown')} epochs)")
        
        # 创建价值网络 (随机初始化)
        value_net = KANValueNetwork(
            input_dim=int(layers[0]),
            hidden_dim=int(layers[1]),
            grid_size=int(model_config.get('grid_size', 5)),
            spline_order=int(model_config.get('spline_order', 3)),
        )
        print("Value network initialized randomly")
        
        agent = PPOAgent(
            policy=policy,
            value_net=value_net,
            config=config,
            device=device,
        )
    
    else:
        # 从头训练
        if args.bc_checkpoint:
            print(f"Warning: BC checkpoint {args.bc_checkpoint} not found, training from scratch")
        print("\nTraining from scratch...")
        agent = PPOAgent(config=config, device=device)
    
    # 打印配置
    print("\n" + "=" * 60)
    print("PPO Configuration")
    print("=" * 60)
    ppo_config = config['ppo']
    print(f"Total timesteps: {ppo_config['total_timesteps']}")
    print(f"Buffer size (n_steps): {ppo_config['n_steps']}")
    print(f"Mini batch size: {ppo_config['mini_batch_size']}")
    print(f"PPO epochs: {ppo_config['ppo_epochs']}")
    print(f"Learning rate: {ppo_config['lr']}")
    print(f"Gamma: {ppo_config['gamma']}")
    print(f"GAE lambda: {ppo_config['gae_lambda']}")
    print(f"Clip range: {ppo_config['clip_range']}")
    print(f"L1 penalty: {ppo_config['l1_penalty']}")
    print(f"Save path: {ppo_config['save_path']}")
    print("=" * 60 + "\n")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(ppo_config['save_path']), exist_ok=True)
    
    # 开始训练
    history = agent.train(
        env=env,
        total_timesteps=ppo_config['total_timesteps'],
        save_path=ppo_config['save_path'],
        log_interval=args.log_interval,
    )
    
    # 评估
    print("\nEvaluating trained policy...")
    from scripts.evaluate import evaluate_policy
    
    results = evaluate_policy(
        policy=agent.policy,
        env=env,
        n_episodes=50,
    )
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    print(f"Mean reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"Mean length: {results['mean_length']:.1f}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print("=" * 60)
    
    # 打印最终稀疏化
    print("\nFinal sparsity:")
    agent.policy.print_sparsity(threshold=0.01)
    
    # 保存训练历史
    history_path = ppo_config['save_path'].replace('.pt', '_history.pt')
    torch.save({
        'episode_rewards': history['episode_rewards'],
        'episode_lengths': history['episode_lengths'],
        'config': config,
    }, history_path)
    print(f"\nHistory saved: {history_path}")
    
    print("\n" + "=" * 60)
    print("PPO Training Completed!")
    print(f"Model: {ppo_config['save_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
