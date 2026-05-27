#!/usr/bin/env python3
"""Pure PyTorch GPU stress test — no MuJoCo, no EGL, no OpenVLA.

Runs matrix multiply + simple MLP forward-backward for configurable duration.
Logs any CUDA errors.
"""
import os
import sys
import time
import argparse
import torch

def stress_test(duration_minutes: int = 10, dim: int = 4096, check_errors: bool = True):
    device = torch.device("cuda:0")
    print(f"[{time.strftime('%H:%M:%S')}] GPU stress starting on {torch.cuda.get_device_name(device)}")
    print(f"  Duration: {duration_minutes} min, Matrix dim: {dim}, Device: {device}")

    # Allocate test tensors
    try:
        a = torch.randn(dim, dim, device=device, dtype=torch.float32)
        b = torch.randn(dim, dim, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        print(f"  Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MiB")
    except RuntimeError as e:
        print(f"  FATAL CUDA ERROR during allocation: {e}")
        return 1

    # Simple MLP for forward-backward
    mlp = torch.nn.Sequential(
        torch.nn.Linear(dim, dim // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(dim // 2, dim),
    ).to(device)

    opt = torch.optim.SGD(mlp.parameters(), lr=0.01)

    start = time.time()
    deadline = start + duration_minutes * 60
    iteration = 0
    errors = []

    try:
        while time.time() < deadline:
            # Matrix multiply
            c = torch.mm(a, b)
            # MLP forward-backward
            x = torch.randn(128, dim, device=device, dtype=torch.float32)
            y = mlp(x)
            loss = y.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Sync and check
            torch.cuda.synchronize()
            if check_errors and iteration % 100 == 0:
                # torch.cuda.synchronize() above already checks for errors
                pass
            iteration += 1
            if iteration % 500 == 0:
                elapsed = time.time() - start
                mem = torch.cuda.memory_allocated(device) / 1024**2
                print(f"  [{time.strftime('%H:%M:%S')}] iter={iteration}, "
                      f"elapsed={elapsed:.0f}s, mem={mem:.1f}MiB")
    except RuntimeError as e:
        errors.append(f"Iteration {iteration}: {e}")
        print(f"  CUDA ERROR at iteration {iteration}: {e}")
    except KeyboardInterrupt:
        print("  Interrupted")

    elapsed = time.time() - start
    print(f"\n  Completed: {iteration} iterations in {elapsed:.1f}s")
    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors:
            print(f"    {e}")
        return 1

    # Quick correctness check: compute CPU reference
    a_cpu = a.cpu()
    b_cpu = b.cpu()
    c_cpu = torch.mm(a_cpu, b_cpu)
    c_gpu = c.cpu()
    max_diff = (c_gpu - c_cpu).abs().max().item()
    print(f"  Correctness check: max GPU-CPU diff = {max_diff:.6e}")
    if max_diff > 1e-3:
        print(f"  WARNING: Large GPU-CPU discrepancy!")
        return 1

    print("  GPU STRESS PASSED")
    return 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=10, help="Duration in minutes")
    p.add_argument("--dim", type=int, default=4096, help="Matrix dimension")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Log GPU info
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    rc = stress_test(duration_minutes=args.duration, dim=args.dim)
    sys.exit(rc)
