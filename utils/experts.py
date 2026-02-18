"""
专家策略实现
"""
import numpy as np


class RandomExpert:
    """随机专家策略"""
    
    def __init__(self, action_dim: int = 1):
        self.action_dim = action_dim
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        return np.random.uniform(-1, 1, size=(self.action_dim,))
    
    def eval(self):
        pass
    
    def __call__(self, state):
        return self.get_action(state)


class HeuristicExpert:
    """启发式专家策略 (基于物理直觉)"""
    
    def __init__(
        self,
        kp1: float = 2.0,
        kd1: float = 0.5,
        kp2: float = 4.0,
        kd2: float = 0.3,
    ):
        self.kp1 = kp1
        self.kd1 = kd1
        self.kp2 = kp2
        self.kd2 = kd2
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        cos1, sin1, cos2, sin2, dot1, dot2 = state
        theta1 = np.arctan2(sin1, cos1)
        theta2 = np.arctan2(sin2, cos2)
        
        action = (
            -self.kp1 * theta1 - self.kd1 * dot1
            - self.kp2 * theta2 - self.kd2 * dot2
        )
        action = np.clip(action, -1.0, 1.0)
        
        return np.array([action], dtype=np.float32)
    
    def eval(self):
        pass
    
    def __call__(self, state):
        return self.get_action(state)
