import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 当前 Triton Program 的编号
    pid = tl.program_id(axis=0)

    # 2. 当前 Program 负责的数据下标
    offsets = pid * BLOCK_SIZE + tl.arange(
        0, BLOCK_SIZE
    )

    # 3. 过滤越界下标
    mask = offsets < n_elements

    # 4. 从 GPU 全局内存读取
    x = tl.load(x_ptr + offsets, mask=mask, other=0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0)

    # 5. 并行计算
    z = x + y

    # 6. 把结果写回 GPU
    tl.store(out_ptr + offsets, z, mask=mask)


def vector_add(x, y, block_size=256):
    assert x.is_cuda and y.is_cuda
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.is_contiguous() and y.is_contiguous()

    out = torch.empty_like(x)
    n = x.numel()

    if n == 0:
        return out

    grid = (triton.cdiv(n, block_size),)

    vector_add_kernel[grid](
        x,
        y,
        out,
        n,
        BLOCK_SIZE=block_size,
    )

    return out


def test_correctness():
    device = "cuda"

    test_sizes = [
        1,
        31,
        256,
        257,
        1000,
        4096,
        1000000,
    ]

    for n in test_sizes:
        x = torch.randn(n, device=device)
        y = torch.randn(n, device=device)

        expected = x + y
        actual = vector_add(x, y)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-6,
        )

        print(f"[PASS] N={n}")

    print("All correctness tests passed!")


def benchmark():
    print("\nBenchmark: Triton vs PyTorch")

    for n in [1024, 65536, 1048576, 4194304]:
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")

        # 预分配输出，避免把分配成本混入测速
        out = torch.empty_like(x)

        def triton_fn():
            vector_add_kernel[
                (triton.cdiv(n, 256),)
            ](
                x, y, out, n,
                BLOCK_SIZE=256,
            )

        def torch_fn():
            torch.add(x, y, out=out)

        # Triton testing.do_bench 使用 GPU event 测时
        ms_triton = triton.testing.do_bench(triton_fn)
        ms_torch = triton.testing.do_bench(torch_fn)

        speedup = ms_torch / ms_triton

        print(
            f"N={n:<9} "
            f"Triton={ms_triton:.5f} ms  "
            f"PyTorch={ms_torch:.5f} ms  "
            f"Speedup={speedup:.2f}x"
        )


if __name__ == "__main__":
    test_correctness()
    benchmark()
