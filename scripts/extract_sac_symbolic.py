"""
提取 SAC 策略的符号化公式
SAC 使用高斯策略，提取均值网络的公式
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from models.kan_policy import KANPolicy
from agents.sac_agent import KANGaussianPolicy

def extract_sac_symbolic(checkpoint_path='checkpoints/sac_kan_model.pt', 
                         output_path='results/sac_symbolic_formula.py'):
    """提取 SAC 策略的符号化公式"""
    
    print("=" * 60)
    print("SAC 策略符号化提取")
    print("=" * 60)
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建策略网络
    policy = KANGaussianPolicy(
        state_dim=6,
        action_dim=1,
        hidden_dim=8,
        grid_size=5,
        spline_order=3,
    )
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy.eval()
    
    print(f"\n模型加载完成: {checkpoint_path}")
    print(f"训练步数: {checkpoint.get('total_steps', 'unknown')}")
    
    # 提取权重
    feature_w = policy.feature.base_weight.detach().numpy()
    feature_b = policy.feature.base_bias.detach().numpy()
    
    mean_w = policy.mean_layer.base_weight.detach().numpy()
    mean_b = policy.mean_layer.base_bias.detach().numpy()
    
    # 计算每层稀疏化
    feature_sparse = (np.abs(feature_w) < 0.01).sum() / feature_w.size
    mean_sparse = (np.abs(mean_w) < 0.01).sum() / mean_w.size
    
    print(f"\n稀疏化统计:")
    print(f"  Feature 层: {feature_sparse*100:.1f}%")
    print(f"  Mean 层: {mean_sparse*100:.1f}%")
    
    # 生成代码
    var_names = ['c1', 's1', 'c2', 's2', 'd1', 'd2']
    
    code_lines = [
        '"""',
        'SAC 策略的符号化实现 (均值网络)',
        '由 extract_sac_symbolic.py 自动生成',
        '"""',
        '',
        'import numpy as np',
        '',
        'def sac_policy(state, deterministic=True):',
        '    """',
        '    SAC 策略函数 (提取的均值网络)',
        '    输入: state = [c1, s1, c2, s2, d1, d2]',
        '    输出: action (标量), log_std',
        '    """',
        f'    {", ".join(var_names)} = state',
        '',
        '    # Feature Layer',
    ]
    
    # Feature Layer
    for i in range(8):
        terms = []
        for j in range(6):
            w = feature_w[i, j]
            if abs(w) > 0.01:
                terms.append(f"{w:.6f} * {var_names[j]}")
        if terms:
            code_lines.append(f'    h{i} = {" + ".join(terms)} + {feature_b[i]:.6f}')
        else:
            code_lines.append(f'    h{i} = {feature_b[i]:.6f}')
    
    code_lines.extend([
        '',
        '    # Apply ReLU',
    ])
    for i in range(8):
        code_lines.append(f'    h{i} = max(0, h{i})')
    
    # Mean Layer
    code_lines.extend([
        '',
        '    # Mean Layer (输出)',
    ])
    
    terms = []
    for i in range(8):
        w = mean_w[0, i]
        if abs(w) > 0.01:
            terms.append(f"{w:.6f} * h{i}")
    
    if terms:
        code_lines.append(f'    mean = {" + ".join(terms)} + {mean_b[0]:.6f}')
    else:
        code_lines.append(f'    mean = {mean_b[0]:.6f}')
    
    code_lines.extend([
        '',
        '    # Tanh 限制',
        '    mean = np.tanh(mean)',
        '',
        '    # 简化的 log_std (常数近似)',
        '    log_std = -1.0  # 近似值',
        '',
        '    if deterministic:',
        '        return mean, log_std',
        '    else:',
        '        std = np.exp(log_std)',
        '        action = np.tanh(mean + std * np.random.randn())',
        '        return action, log_std',
        '',
        '',
        'def test_policy():',
        '    """测试策略"""',
        '    for i in range(5):',
        '        state = np.random.randn(6)',
        '        action, _ = sac_policy(state)',
        '        print(f"State {i+1}: {state.round(2)} -> Action: {action:.4f}")',
        '',
        '',
        'if __name__ == "__main__":',
        '    test_policy()',
    ])
    
    # 写入文件
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(code_lines))
    
    print(f"\n[OK] SAC 符号化公式已保存: {output_path}")
    
    # 打印关键公式
    print("\n" + "=" * 60)
    print("提取的核心公式 (均值网络)")
    print("=" * 60)
    
    # 找最重要的特征
    feature_importance = np.abs(feature_w).sum(axis=0)
    print("\n输入特征重要性:")
    for i, (name, imp) in enumerate(zip(var_names, feature_importance)):
        print(f"  {name}: {imp:.3f}")
    
    # 找最重要的隐藏节点
    mean_importance = np.abs(mean_w[0])
    top_nodes = np.argsort(mean_importance)[-3:][::-1]
    print(f"\n最重要的隐藏节点: {top_nodes}")
    for node in top_nodes:
        print(f"  h{node}: weight = {mean_w[0, node]:.4f}")
    
    print("=" * 60)

if __name__ == "__main__":
    extract_sac_symbolic()
