"""
行为克隆训练脚本 (2_train_bc.py)
================================
使用专家轨迹训练 KAN 策略

使用方法:
  # 基本用法 (使用 config.yaml 中的默认配置)
  python scripts/2_train_bc.py

  # 自定义参数
  python scripts/2_train_bc.py --data data/expert_trajectories.pt --epochs 300 --l1_penalty 1e-4

  # 从指定 checkpoint 继续训练
  python scripts/2_train_bc.py --resume checkpoints/bc_kan_model.pt --epochs 100
"""

import os
import sys
import argparse
import yaml
import torch
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.bc_agent import BCAgent
from models.kan_policy import KANPolicy


def main():
    parser = argparse.ArgumentParser(description="训练 KAN-BC 策略")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='专家数据路径 (默认从 config 读取)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='训练轮数 (默认从 config 读取)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='批次大小 (默认从 config 读取)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='学习率 (默认从 config 读取)'
    )
    parser.add_argument(
        '--l1_penalty',
        type=float,
        default=None,
        help='L1 正则化系数 (默认从 config 读取)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='模型保存路径 (默认从 config 读取)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从 checkpoint 恢复训练'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='计算设备 (cuda/cpu，默认自动检测)'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='日志打印间隔'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    print(f"加载配置: {args.config}")
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 命令行参数覆盖配置
    if args.data:
        config['bc']['expert_data_path'] = args.data
    if args.epochs:
        config['bc']['epochs'] = args.epochs
    if args.batch_size:
        config['bc']['batch_size'] = args.batch_size
    if args.lr:
        config['bc']['lr'] = args.lr
    if args.l1_penalty:
        config['bc']['l1_penalty'] = args.l1_penalty
    if args.output:
        config['bc']['save_path'] = args.output
    
    # 确定设备
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确定数据路径
    data_path = args.data or config['bc']['expert_data_path']
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"专家数据不存在: {data_path}\n"
            f"请先运行: python scripts/1_generate_expert.py"
        )
    
    # 创建 Agent
    print("\n创建 BC Agent...")
    if args.resume and os.path.exists(args.resume):
        print(f"从 checkpoint 恢复: {args.resume}")
        agent = BCAgent(config=config, device=device)
        agent.load(args.resume)
    else:
        if args.resume:
            print(f"警告: checkpoint {args.resume} 不存在，从头开始训练")
        agent = BCAgent(config=config, device=device)
    
    # 打印配置
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"数据路径: {data_path}")
    print(f"训练轮数: {config['bc']['epochs']}")
    print(f"批次大小: {config['bc']['batch_size']}")
    print(f"学习率: {config['bc']['lr']}")
    print(f"L1 惩罚: {config['bc']['l1_penalty']}")
    print(f"梯度裁剪: {config['bc']['grad_clip']}")
    print(f"验证集比例: {config['bc']['val_split']}")
    print(f"早停耐心: {config['bc']['early_stop_patience']}")
    print(f"网格更新频率: {config['model']['grid_update_freq']} epochs")
    print(f"模型保存路径: {config['bc']['save_path']}")
    print("=" * 60 + "\n")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(config['bc']['save_path']), exist_ok=True)
    
    # 开始训练
    history = agent.train(
        data_path=data_path,
        epochs=config['bc']['epochs'],
        save_path=config['bc']['save_path'],
        log_interval=args.log_interval,
        verbose=True,
    )
    
    # 加载最佳模型进行最终评估
    print("\n加载最佳模型进行最终评估...")
    agent.load(config['bc']['save_path'])
    
    # 打印最终稀疏化信息
    print("\n最终模型稀疏化统计:")
    agent.policy.print_sparsity(threshold=0.01)
    
    # 保存训练历史
    history_path = config['bc']['save_path'].replace('.pt', '_history.pt')
    torch.save({
        'train_losses': history['train_losses'],
        'val_losses': history['val_losses'],
        'config': config,
    }, history_path)
    print(f"\n训练历史已保存: {history_path}")
    
    print("\n" + "=" * 60)
    print("✅ BC 训练完成!")
    print(f"最佳模型: {config['bc']['save_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
