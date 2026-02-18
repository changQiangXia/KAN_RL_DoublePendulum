"""
SAC 训练脚本 (4_train_sac.py)
=============================
使用 SAC 算法训练 KAN 策略

使用方法:
  # 从头训练
  python scripts/4_train_sac.py --total_timesteps 100000

  # 从 BC 冷启动 (加载 BC 策略权重初始化)
  python scripts/4_train_sac.py --bc_checkpoint checkpoints/bc_kan_model.pt
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.sac_agent import SACAgent, KANGaussianPolicy
from envs.wrapper import make_acrobot_env


def main():
    parser = argparse.ArgumentParser(description="Train KAN-SAC")
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
        help='Load BC checkpoint for warm start (optional)'
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
        help='Log interval (in episodes)'
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
        config['sac']['total_timesteps'] = args.total_timesteps
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 创建设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 创建环境
    print("\nCreating environment...")
    env = make_acrobot_env(continuous=True, seed=args.seed)
    
    # 创建 Agent
    policy = None
    
    if args.bc_checkpoint and os.path.exists(args.bc_checkpoint):
        # 从 BC 冷启动: 用 BC 策略初始化 SAC 策略的均值网络
        print(f"\nWarm starting from BC: {args.bc_checkpoint}")
        print("Note: SAC uses Gaussian policy, BC weights initialize the mean network only")
        
        model_config = config.get('model', {})
        
        # 创建策略网络
        policy = KANGaussianPolicy(
            state_dim=int(model_config.get('state_dim', 6)),
            action_dim=int(model_config.get('action_dim', 1)),
            hidden_dim=int(model_config.get('hidden_dim', 8)),
            grid_size=int(model_config.get('grid_size', 5)),
            spline_order=int(model_config.get('spline_order', 3)),
        )
        
        # 加载 BC 权重 (仅用于初始化 mean_layer)
        checkpoint = torch.load(args.bc_checkpoint, map_location=device)
        bc_state_dict = checkpoint['policy_state_dict']
        
        # 尝试加载兼容的层
        try:
            policy.feature.load_state_dict(bc_state_dict['layer1.state_dict'])
            policy.mean_layer.load_state_dict(bc_state_dict['layer2.state_dict'])
            print("BC weights loaded successfully for mean network initialization")
        except Exception as e:
            print(f"Warning: Could not load all BC weights: {e}")
            print("Training from scratch for incompatible layers")
    
    # 创建 SAC Agent
    agent = SACAgent(
        policy=policy,
        config=config,
        device=device,
    )
    
    # 打印配置
    print("\n" + "=" * 60)
    print("SAC Configuration")
    print("=" * 60)
    sac_config = config['sac']
    print(f"Total timesteps: {sac_config['total_timesteps']}")
    print(f"Buffer size: {sac_config['buffer_size']}")
    print(f"Batch size: {sac_config['batch_size']}")
    print(f"Learning rate: {sac_config['lr']}")
    print(f"Gamma: {sac_config['gamma']}")
    print(f"Tau: {sac_config['tau']}")
    print(f"Auto entropy tuning: {sac_config['automatic_entropy_tuning']}")
    print(f"Warmup steps: {sac_config['warmup_steps']}")
    print(f"L1 penalty: {sac_config['l1_penalty']}")
    print(f"Save path: {sac_config['save_path']}")
    print("=" * 60 + "\n")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(sac_config['save_path']), exist_ok=True)
    
    # 开始训练
    history = agent.train(
        env=env,
        total_timesteps=sac_config['total_timesteps'],
        save_path=sac_config['save_path'],
        log_interval=args.log_interval,
    )
    
    # 评估
    print("\nEvaluating trained policy...")
    from scripts.evaluate import evaluate_policy
    
    # 包装策略以适配 evaluate_policy
    class PolicyWrapper:
        def __init__(self, policy):
            self.policy = policy
        
        def get_action(self, state, deterministic=True):
            return self.policy.get_action(state, deterministic)
        
        def eval(self):
            pass
        
        def print_sparsity(self, threshold=0.01):
            # SAC 策略的稀疏化统计
            print("=" * 50)
            print("SAC Policy sparsity not fully implemented")
            print("=" * 50)
    
    wrapped_policy = PolicyWrapper(agent.policy)
    
    results = evaluate_policy(
        policy=wrapped_policy,
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
    
    # 保存训练历史
    history_path = sac_config['save_path'].replace('.pt', '_history.pt')
    torch.save({
        'episode_rewards': history['episode_rewards'],
        'episode_lengths': history['episode_lengths'],
        'config': config,
    }, history_path)
    print(f"\nHistory saved: {history_path}")
    
    print("\n" + "=" * 60)
    print("SAC Training Completed!")
    print(f"Model: {sac_config['save_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
