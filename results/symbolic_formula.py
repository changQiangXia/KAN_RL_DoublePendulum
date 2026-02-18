"""
KAN 策略的符号化实现
由 symbolic.py 自动生成
"""

import numpy as np

def kan_policy(state):
    """
    KAN 策略函数
    输入: state = [c1, s1, c2, s2, d1, d2]
    输出: action (标量)
    """
    c1, s1, c2, s2, d1, d2 = state

    # Layer 1: Linear part
    h0_linear = 0.162183 * c1 + 1.005088 * s1 + -0.116296 * c2 + 0.052023 * s2 + 0.359153 * d1 + 0.228279 * d2 + -0.151030
    h1_linear = 0.147088 * c1 + -0.071551 * s1 + -0.309561 * c2 + 1.409106 * s2 + 0.025754 * d2 + 0.048537
    h2_linear = -0.464081 * c1 + 0.359153 * c2 + 1.246001 * s2 + 0.034609 * d2 + -0.014625
    h3_linear = 0.118725 * c1 + -0.041987 * s1 + -0.331197 * c2 + -2.101315 * s2 + -0.018041 * d1 + 0.007717
    h4_linear = -0.681627 * c1 + -0.018415 * s1 + 0.170762 * c2 + 0.678853 * s2 + -0.132140 * d1 + -0.056743 * d2 + -0.233061
    h5_linear = -0.167543 * c1 + -0.449696 * s1 + -0.723556 * c2 + -0.788266 * s2 + -0.219579 * d1 + -0.102616 * d2 + -0.057037
    h6_linear = 0.084192 * c1 + 0.152986 * s1 + 0.390115 * c2 + -1.227925 * s2 + 0.108885 * d1 + 0.065018 * d2 + 0.097261
    h7_linear = 0.413665 * c1 + -0.346269 * s1 + 0.573355 * c2 + -0.753869 * s2 + -0.148358 * d1 + -0.073389 * d2 + 0.038848

    # Layer 1: B-spline part (simplified)
    # Note: Full B-spline computation is complex,
    # this is a linear approximation
    h0_spline = 0.156124 * c1 + 0.264974 * s1 + 0.236902 * c2 + 0.209554 * s2 + 0.035007 * d1 + 0.036606 * d2
    h0 = np.tanh(h0_linear + h0_spline)
    h1_spline = 0.065303 * c1 + 0.025406 * s1 + 0.144846 * c2 + 1.429443 * s2 + 0.010019 * d2
    h1 = np.tanh(h1_linear + h1_spline)
    h2_spline = 0.174258 * c1 + 0.029226 * s1 + 0.212591 * c2 + 0.524195 * s2 + 0.015253 * d1 + 0.014239 * d2
    h2 = np.tanh(h2_linear + h2_spline)
    h3_spline = 0.049055 * c1 + 0.063773 * s1 + 0.449445 * c2 + 1.256738 * s2 + 0.010891 * d2
    h3 = np.tanh(h3_linear + h3_spline)
    h4_spline = 0.280918 * c1 + 0.071240 * s1 + 0.092458 * c2 + 0.312548 * s2 + 0.019108 * d1 + 0.043697 * d2
    h4 = np.tanh(h4_linear + h4_spline)
    h5_spline = 0.138036 * c1 + 0.152645 * s1 + 0.248549 * c2 + 0.294150 * s2 + 0.015158 * d1 + 0.019354 * d2
    h5 = np.tanh(h5_linear + h5_spline)
    h6_spline = 0.094359 * c1 + 0.100606 * s1 + 0.388119 * c2 + 0.734453 * s2 + 0.036782 * d1 + 0.028968 * d2
    h6 = np.tanh(h6_linear + h6_spline)
    h7_spline = 0.136284 * c1 + 0.127732 * s1 + 0.308953 * c2 + 0.262492 * s2 + 0.013709 * d1 + 0.013610 * d2
    h7 = np.tanh(h7_linear + h7_spline)

    # Layer 2: Linear part
    action_linear = -0.416719 * h0 + -1.067622 * h1 + 0.258915 * h2 + 0.526561 * h3 + 0.562473 * h5 + 0.257550 * h6 + 0.542674 * h7 + -0.014131

    # Layer 2: B-spline part (simplified)
    action_spline = 0.148701 * h0 + 0.441916 * h1 + 0.274892 * h2 + 0.313119 * h3 + 0.178218 * h4 + 0.249720 * h5 + 0.149300 * h6 + 0.306050 * h7
    action = np.tanh(action_linear + action_spline)

    return action


def test_policy():
    """测试策略函数"""
    # 随机测试
    for _ in range(5):
        state = np.random.randn(6)
        action = kan_policy(state)
        print(f"State: {state}, Action: {action:.4f}")


if __name__ == "__main__":
    test_policy()