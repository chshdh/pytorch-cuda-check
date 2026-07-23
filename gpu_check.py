import time

import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable")

device = torch.device("cuda")
properties = torch.cuda.get_device_properties(0)
size = 4096

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU:", properties.name)
print("VRAM:", round(properties.total_memory / 1024**3, 2), "GiB")

a = torch.randn((size, size), device=device)
b = torch.randn((size, size), device=device)

# 第一次运算用于初始化和预热
_ = a @ b
torch.cuda.synchronize()

start = time.perf_counter()
result = a @ b
torch.cuda.synchronize()
elapsed = time.perf_counter() - start

tflops = 2 * size**3 / elapsed / 1e12

print("Result device:", result.device)
print("Matrix shape:", tuple(result.shape))
print("Elapsed:", round(elapsed, 4), "seconds")
print("Approx performance:", round(tflops, 2), "TFLOPS")
print("Result checksum:", round(result[0, 0].item(), 4))
