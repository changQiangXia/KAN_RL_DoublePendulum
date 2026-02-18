"""
符号化公式提取工具 (symbolic.py)
================================
将训练好的 KAN 权重转换为可读的数学公式

核心功能:
1. 提取稀疏化的 B-spline 系数
2. 将非零系数转换为 SymPy 符号表达式
3. 生成人类可读的 Python 函数

输出示例:
    action = -0.5 * sin(theta1) + 0.3 * cos(theta2) - 0.1 * dot_theta1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional

from models.kan_policy import KANPolicy


class KANSymbolicExtractor:
    """
    KAN 符号化提取器
    
    将 KAN 网络的权重转换为显式数学公式
    """
    
    def __init__(
        self,
        policy: KANPolicy,
        var_names: Optional[List[str]] = None,
        threshold: float = 0.01,
    ):
        """
        Args:
            policy: 训练好的 KAN 策略
            var_names: 变量名称列表 (默认: ['c1', 's1', 'c2', 's2', 'd1', 'd2'])
            threshold: 视为零的权重阈值
        """
        self.policy = policy
        self.policy.eval()
        self.threshold = threshold
        
        # 变量名
        self.var_names = var_names or ['c1', 's1', 'c2', 's2', 'd1', 'd2']
        
        # 创建 SymPy 符号
        self.symbols = [sp.Symbol(name) for name in self.var_names]
        
        # 为隐藏层创建符号
        self.hidden_symbols = [sp.Symbol(f"h{i}") for i in range(policy.hidden_dim)]
        
        print(f"[SymbolicExtractor] 初始化完成")
        print(f"  - 输入变量: {self.var_names}")
        print(f"  - 隐藏变量: {[f'h{i}' for i in range(policy.hidden_dim)]}")
        print(f"  - 稀疏阈值: {threshold}")
    
    def extract_layer_formula(
        self,
        layer_idx: int,
        output_idx: Optional[int] = None,
    ) -> Dict:
        """
        提取单层网络的公式
        
        Args:
            layer_idx: 层索引 (1 或 2)
            output_idx: 输出节点索引 (None 表示提取所有)
        
        Returns:
            公式信息字典
        """
        if layer_idx == 1:
            layer = self.policy.layer1
            symbols = self.symbols
            var_names = self.var_names
        elif layer_idx == 2:
            layer = self.policy.layer2
            symbols = self.hidden_symbols
            var_names = [f"h{i}" for i in range(layer.in_features)]
        else:
            raise ValueError(f"无效层索引: {layer_idx}")
        
        # 获取权重
        base_weight = layer.base_weight.detach().cpu().numpy()  # (out, in)
        spline_weight = layer.spline_weight.detach().cpu().numpy()  # (out, in, n_basis)
        scaler = layer.spline_scaler.detach().cpu().numpy()  # (out, in)
        
        in_features = layer.in_features
        out_features = layer.out_features
        n_basis = spline_weight.shape[2]
        
        # 确定要提取的输出节点
        if output_idx is None:
            output_indices = range(out_features)
        else:
            output_indices = [output_idx]
        
        formulas = []
        
        for o_idx in output_indices:
            # 构建该输出的表达式
            terms = []
            
            # 1. 基础线性部分
            for i_idx in range(in_features):
                w = base_weight[o_idx, i_idx]
                if abs(w) > self.threshold:
                    term = w * symbols[i_idx]
                    terms.append((w, var_names[i_idx], term))
            
            # 2. B-spline 部分 (简化: 视为多个基函数的加权和)
            for i_idx in range(in_features):
                # 计算该输入边的总权重
                spline_w = spline_weight[o_idx, i_idx]  # (n_basis,)
                scaler_w = scaler[o_idx, i_idx]
                
                total_weight = np.abs(spline_w).mean() * scaler_w
                
                if total_weight > self.threshold:
                    # 简化表示
                    term_str = f"BS({var_names[i_idx]})"
                    terms.append((total_weight, term_str, total_weight * symbols[i_idx]))
            
            # 排序 (按权重绝对值降序)
            terms.sort(key=lambda x: abs(x[0]), reverse=True)
            
            # 构建 SymPy 表达式
            sympy_expr = sum(term[2] for term in terms) if terms else sp.Float(0)
            
            # 添加偏置
            bias = layer.base_bias[o_idx].item()
            if abs(bias) > self.threshold:
                sympy_expr += bias
            
            formulas.append({
                'output_idx': o_idx,
                'sympy_expr': sympy_expr,
                'terms': terms,
                'n_terms': len(terms),
                'bias': bias,
            })
        
        return {
            'layer_idx': layer_idx,
            'formulas': formulas,
            'in_features': in_features,
            'out_features': out_features,
        }
    
    def extract_full_formula(self) -> sp.Expr:
        """
        提取完整的端到端公式
        
        由于有两层网络，完整公式是嵌套的:
        h = tanh(BS1(x) + b1)
        y = tanh(BS2(h) + b2)
        
        这里我们尝试简化表示
        """
        print("\n提取完整公式...")
        
        # 提取每层
        layer1_info = self.extract_layer_formula(1)
        layer2_info = self.extract_layer_formula(2)
        
        # 对于极简 [6,8,1] 网络，我们可以尝试近似
        # 但完整 B-spline 组合非常复杂
        
        # 返回简化描述
        print(f"\nLayer 1 (6 -> 8):")
        for f in layer1_info['formulas'][:3]:  # 只显示前3个
            print(f"  h{f['output_idx']} = {f['sympy_expr']}")
        if len(layer1_info['formulas']) > 3:
            print(f"  ... ({len(layer1_info['formulas'])} 个隐藏节点)")
        
        print(f"\nLayer 2 (8 -> 1):")
        f = layer2_info['formulas'][0]
        print(f"  action = tanh({f['sympy_expr']})")
        
        return layer1_info, layer2_info
    
    def generate_python_function(self, output_path: str):
        """
        生成可执行的 Python 函数代码
        
        Args:
            output_path: 输出文件路径
        """
        # 提取权重
        w1_base = self.policy.layer1.base_weight.detach().cpu().numpy()
        w1_spline = self.policy.layer1.spline_weight.detach().cpu().numpy()
        b1 = self.policy.layer1.base_bias.detach().cpu().numpy()
        
        w2_base = self.policy.layer2.base_weight.detach().cpu().numpy()
        w2_spline = self.policy.layer2.spline_weight.detach().cpu().numpy()
        b2 = self.policy.layer2.base_bias.detach().cpu().numpy()
        
        # 生成代码
        code_lines = [
            '"""',
            'KAN 策略的符号化实现',
            '由 symbolic.py 自动生成',
            '"""',
            '',
            'import numpy as np',
            '',
            'def kan_policy(state):',
            '    """',
            '    KAN 策略函数',
            '    输入: state = [c1, s1, c2, s2, d1, d2]',
            '    输出: action (标量)',
            '    """',
            f'    {", ".join(self.var_names)} = state',
            '',
            '    # Layer 1: Linear part',
        ]
        
        # Layer 1 线性部分
        for i in range(8):
            terms = []
            for j in range(6):
                w = w1_base[i, j]
                if abs(w) > self.threshold:
                    terms.append(f"{w:.6f} * {self.var_names[j]}")
            if terms:
                code_lines.append(f'    h{i}_linear = {" + ".join(terms)} + {b1[i]:.6f}')
            else:
                code_lines.append(f'    h{i}_linear = {b1[i]:.6f}')
        
        code_lines.extend([
            '',
            '    # Layer 1: B-spline part (simplified)',
            '    # Note: Full B-spline computation is complex,',
            '    # this is a linear approximation',
        ])
        
        # Layer 1 B-spline 部分 (简化)
        for i in range(8):
            terms = []
            for j in range(6):
                # 简化为平均权重
                w_spline = np.abs(w1_spline[i, j]).mean()
                if w_spline > self.threshold:
                    terms.append(f"{w_spline:.6f} * {self.var_names[j]}")
            if terms:
                code_lines.append(f'    h{i}_spline = {" + ".join(terms)}')
                code_lines.append(f'    h{i} = np.tanh(h{i}_linear + h{i}_spline)')
            else:
                code_lines.append(f'    h{i} = np.tanh(h{i}_linear)')
        
        code_lines.extend([
            '',
            '    # Layer 2: Linear part',
        ])
        
        # Layer 2
        terms = []
        for i in range(8):
            w = w2_base[0, i]
            if abs(w) > self.threshold:
                terms.append(f"{w:.6f} * h{i}")
        
        if terms:
            code_lines.append(f'    action_linear = {" + ".join(terms)} + {b2[0]:.6f}')
        else:
            code_lines.append(f'    action_linear = {b2[0]:.6f}')
        
        code_lines.extend([
            '',
            '    # Layer 2: B-spline part (simplified)',
        ])
        
        terms = []
        for i in range(8):
            w_spline = np.abs(w2_spline[0, i]).mean()
            if w_spline > self.threshold:
                terms.append(f"{w_spline:.6f} * h{i}")
        
        if terms:
            code_lines.append(f'    action_spline = {" + ".join(terms)}')
            code_lines.append('    action = np.tanh(action_linear + action_spline)')
        else:
            code_lines.append('    action = np.tanh(action_linear)')
        
        code_lines.extend([
            '',
            '    return action',
            '',
            '',
            'def test_policy():',
            '    """测试策略函数"""',
            '    # 随机测试',
            '    for _ in range(5):',
            '        state = np.random.randn(6)',
            '        action = kan_policy(state)',
            '        print(f"State: {state}, Action: {action:.4f}")',
            '',
            '',
            'if __name__ == "__main__":',
            '    test_policy()',
        ])
        
        # 写入文件
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(code_lines))
        
        print(f"\n[OK] Python function saved: {output_path}")
    
    def print_formulas(self):
        """打印提取的公式"""
        print("\n" + "=" * 60)
        print("KAN 符号化公式提取")
        print("=" * 60)
        
        self.extract_full_formula()
        
        print("\n" + "=" * 60)


def extract_from_checkpoint(
    checkpoint_path: str,
    config_path: str = 'config.yaml',
    output_path: str = 'results/symbolic_formula.py',
    threshold: float = 0.01,
):
    """
    从 checkpoint 提取符号化公式
    
    Args:
        checkpoint_path: 模型检查点路径
        config_path: 配置文件路径
        output_path: 输出 Python 文件路径
        threshold: 稀疏化阈值
    """
    import yaml
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_config = config.get('model', {})
    layers = model_config.get('layers', [6, 8, 1])
    
    # 创建模型
    policy = KANPolicy(
        input_dim=int(layers[0]),
        hidden_dim=int(layers[1]),
        output_dim=int(layers[2]),
        grid_size=int(model_config.get('grid_size', 5)),
        spline_order=int(model_config.get('spline_order', 3)),
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    policy.load_state_dict(checkpoint['policy_state_dict'])
    
    print(f"加载模型: {checkpoint_path}")
    print(f"训练 epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # 打印稀疏化信息
    policy.print_sparsity(threshold=threshold)
    
    # 提取公式
    extractor = KANSymbolicExtractor(
        policy=policy,
        threshold=threshold,
    )
    
    extractor.print_formulas()
    extractor.generate_python_function(output_path)
    
    print(f"\n[OK] Symbolic extraction completed!")


if __name__ == "__main__":
    # 测试: 从 BC 模型提取
    extract_from_checkpoint(
        checkpoint_path='checkpoints/bc_kan_model.pt',
        output_path='results/symbolic_formula.py',
    )
