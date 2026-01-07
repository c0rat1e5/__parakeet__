#!/usr/bin/env python3
"""CLIテスト用"""
import torch
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# 簡単なテスト
if torch.cuda.is_available():
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f"Matrix multiply test: OK ({y.shape})")
    del x, y
    torch.cuda.empty_cache()
    print("CUDA basic test passed!")
