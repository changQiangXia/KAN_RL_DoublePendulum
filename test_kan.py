"""
KAN Policy 测试脚本
验证模型在 RTX 3050Ti 上能否正常运行
"""
import sys
sys.path.insert(0, '.')

from models.kan_policy import KANPolicy, test_kan_policy
import torch

if __name__ == "__main__":
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print()
    test_kan_policy()
