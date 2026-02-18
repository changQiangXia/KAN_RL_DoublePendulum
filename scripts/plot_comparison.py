"""
生成算法对比可视化
绘制训练曲线和性能对比图
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def load_history(path):
    """加载训练历史"""
    if os.path.exists(path):
        data = torch.load(path)
        return data.get('episode_rewards', []), data.get('episode_lengths', [])
    return [], []

def plot_training_curves():
    """绘制各算法的训练曲线"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('KAN-RL Double Pendulum: Algorithm Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    # 加载数据
    bc_rewards, bc_lengths = load_history('checkpoints/bc_kan_model_history.pt')
    ppo_rewards, ppo_lengths = load_history('checkpoints/ppo_kan_model_v2_history.pt')
    sac_rewards, sac_lengths = load_history('checkpoints/sac_kan_model_history.pt')
    
    # 1. 奖励曲线对比
    ax1 = axes[0, 0]
    
    # SAC 曲线
    if sac_rewards:
        # 滑动平均
        window = 50
        sac_smooth = np.convolve(sac_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(sac_smooth, label='SAC', color='green', linewidth=2.5)
        ax1.axhline(y=np.mean(sac_rewards[-100:]), color='green', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # PPO 曲线
    if ppo_rewards:
        window = 20
        ppo_smooth = np.convolve(ppo_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(ppo_smooth, label='PPO', color='red', linewidth=2.5)
        ax1.axhline(y=np.mean(ppo_rewards[-100:]), color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Episode Reward', fontsize=11)
    ax1.set_title('Training Curves (Reward)', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)
    
    # 2. 最终性能柱状图
    ax2 = axes[0, 1]
    
    algorithms = ['SAC', 'Expert', 'BC', 'DAgger', 'PPO']
    rewards = [-193.7, -265.8, -439.2, -443.2, -466.5]
    colors = ['gold', 'silver', '#3498db', '#9b59b6', '#e74c3c']
    
    bars = ax2.barh(algorithms, rewards, color=colors, edgecolor='black', linewidth=1.5, height=0.6)
    
    # 添加数值标签 (放在条形右侧)
    for i, (bar, reward) in enumerate(zip(bars, rewards)):
        width = bar.get_width()
        # 根据数值调整标签位置
        if reward > -300:  # 好成绩在左边
            ax2.text(width - 10, bar.get_y() + bar.get_height()/2, 
                    f'{reward:.1f}', ha='right', va='center', fontweight='bold', fontsize=11, color='black')
        else:  # 差成绩在右边
            ax2.text(width + 10, bar.get_y() + bar.get_height()/2, 
                    f'{reward:.1f}', ha='left', va='center', fontweight='bold', fontsize=11, color='black')
    
    ax2.set_xlabel('Average Reward (higher is better)', fontsize=11)
    ax2.set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    ax2.axvline(x=-265.8, color='gray', linestyle='--', alpha=0.7, linewidth=1.5, label='Expert Baseline')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_xlim([-500, -150])  # 调整x轴范围
    
    # 3. 成功率对比
    ax3 = axes[1, 0]
    
    success_rates = [96, 100, 88, 82, 50]  # Expert assumed 100%
    x_pos = np.arange(len(algorithms))
    
    bars = ax3.bar(x_pos, success_rates, color=colors, edgecolor='black', linewidth=1.5, width=0.6)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(algorithms, rotation=0, fontsize=10)  # 不旋转
    ax3.set_ylabel('Success Rate (%)', fontsize=11)
    ax3.set_title('Success Rate Comparison', fontsize=13, fontweight='bold')
    ax3.set_ylim([0, 115])
    
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{rate}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(labelsize=9)
    
    # 4. 稀疏化和参数量
    ax4 = axes[1, 1]
    
    sparsity = [42, 42, 42, 47, 42]  # Approximate
    
    bars = ax4.bar(x_pos, sparsity, color='skyblue', edgecolor='black', linewidth=1.5, width=0.6)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(algorithms, rotation=0, fontsize=10)
    ax4.set_ylabel('Sparsity (%)', fontsize=11)
    ax4.set_title('Network Sparsity', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 60])
    
    for bar, sp in zip(bars, sparsity):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{sp}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.tick_params(labelsize=9)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], pad=2.0, h_pad=3.0, w_pad=2.5)
    
    # 保存
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/algorithm_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/algorithm_comparison.pdf', bbox_inches='tight')
    print("[OK] Comparison plot saved: results/algorithm_comparison.png")
    
    plt.show()

def plot_sac_detailed():
    """绘制 SAC 详细训练曲线"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('SAC Training Details (Champion Algorithm)', fontsize=15, fontweight='bold', y=0.98)
    
    sac_rewards, sac_lengths = load_history('checkpoints/sac_kan_model_history.pt')
    
    if sac_rewards:
        # 奖励曲线
        ax1 = axes[0]
        ax1.plot(sac_rewards, alpha=0.3, color='green', label='Raw')
        
        # 滑动平均
        window = 100
        if len(sac_rewards) > window:
            sac_smooth = np.convolve(sac_rewards, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(sac_rewards)), sac_smooth, 
                    color='darkgreen', linewidth=2, label=f'MA({window})')
        
        ax1.axhline(y=-193.7, color='gold', linestyle='--', linewidth=2, label='Final: -193.7')
        ax1.axhline(y=-265.8, color='gray', linestyle='--', alpha=0.5, label='Expert: -265.8')
        
        ax1.set_xlabel('Episode', fontsize=11)
        ax1.set_ylabel('Episode Reward', fontsize=11)
        ax1.set_title('SAC Reward Curve', fontsize=13, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(labelsize=9)
        
        # 回合长度
        ax2 = axes[1]
        ax2.plot(sac_lengths, alpha=0.3, color='blue', label='Raw')
        
        if len(sac_lengths) > window:
            len_smooth = np.convolve(sac_lengths, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(sac_lengths)), len_smooth, 
                    color='darkblue', linewidth=2, label=f'MA({window})')
        
        ax2.axhline(y=194.7, color='gold', linestyle='--', linewidth=2, label='Final: 195')
        ax2.axhline(y=266.8, color='gray', linestyle='--', alpha=0.5, label='Expert: 267')
        
        ax2.set_xlabel('Episode', fontsize=11)
        ax2.set_ylabel('Episode Length (steps)', fontsize=11)
        ax2.set_title('SAC Episode Length', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=9)
    
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94], pad=2.0, h_pad=2.5, w_pad=2.5)
    plt.savefig('results/sac_training_detail.png', dpi=300, bbox_inches='tight')
    print("[OK] SAC detail plot saved: results/sac_training_detail.png")
    
    plt.show()

if __name__ == "__main__":
    print("Generating comparison plots...")
    plot_training_curves()
    plot_sac_detailed()
    print("\nAll plots generated!")
