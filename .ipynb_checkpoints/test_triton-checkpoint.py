import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

n = 1024
x = torch.randn(n, device='cuda')
y = torch.randn(n, device='cuda')
out = torch.empty_like(x)

grid = (triton.cdiv(n, 1024),)
add_kernel[grid](x, y, out, n, 1024)
torch.cuda.synchronize()
print("Triton 内核运行成功")
print("结果正确:", torch.allclose(out, x + y))
