"""
SAC 策略的符号化实现 (均值网络)
由 extract_sac_symbolic.py 自动生成
"""

import numpy as np

def sac_policy(state, deterministic=True):
    """
    SAC 策略函数 (提取的均值网络)
    输入: state = [c1, s1, c2, s2, d1, d2]
    输出: action (标量), log_std
    """
    c1, s1, c2, s2, d1, d2 = state

    # Feature Layer
    h0 = 0.107003 * c1 + 0.138368 * s1 + 0.091532 * c2 + 0.115129 * s2 + 0.185895 * d1 + 0.053713 * d2 + 0.167746
    h1 = 0.016972 * c1 + -0.158019 * s1 + -0.041465 * c2 + -0.136124 * s2 + -0.082276 * d1 + -0.110737 * d2 + -0.000524
    h2 = 0.096806 * c1 + 0.048667 * s1 + 0.172431 * c2 + 0.086338 * s2 + 0.107583 * d1 + 0.180125 * d2 + 0.027530
    h3 = 0.037688 * c1 + -0.074815 * s1 + 0.084276 * c2 + -0.055972 * s2 + 0.044695 * d2 + 0.055894
    h4 = 0.047659 * c1 + 0.152232 * s1 + -0.080153 * c2 + 0.201116 * s2 + -0.136030 * d1 + -0.259800 * d2 + 0.059829
    h5 = 0.089144 * c1 + 0.173076 * s1 + 0.172352 * c2 + -0.012369 * s2 + 0.072508 * d1 + 0.117991 * d2 + 0.044362
    h6 = 0.097164 * c1 + 0.121789 * s1 + 0.084263 * c2 + -0.030178 * s2 + 0.051126 * d1 + -0.036046 * d2 + 0.065629
    h7 = -0.050383 * s1 + 0.077466 * c2 + -0.091291 * s2 + -0.104048 * d1 + -0.034176 * d2 + -0.037597

    # Apply ReLU
    h0 = max(0, h0)
    h1 = max(0, h1)
    h2 = max(0, h2)
    h3 = max(0, h3)
    h4 = max(0, h4)
    h5 = max(0, h5)
    h6 = max(0, h6)
    h7 = max(0, h7)

    # Mean Layer (输出)
    mean = -0.124099 * h0 + -0.807738 * h1 + 0.630071 * h2 + -0.010198 * h4 + 0.051166 * h5 + -0.023224 * h6 + -0.459503 * h7 + -0.020716

    # Tanh 限制
    mean = np.tanh(mean)

    # 简化的 log_std (常数近似)
    log_std = -1.0  # 近似值

    if deterministic:
        return mean, log_std
    else:
        std = np.exp(log_std)
        action = np.tanh(mean + std * np.random.randn())
        return action, log_std


def test_policy():
    """测试策略"""
    for i in range(5):
        state = np.random.randn(6)
        action, _ = sac_policy(state)
        print(f"State {i+1}: {state.round(2)} -> Action: {action:.4f}")


if __name__ == "__main__":
    test_policy()