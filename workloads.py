"""Pre-defined TVM workloads for tuning experiments.

All TVM script complexity is encapsulated here. Interact only through
get_workload(name) and list_workloads().
"""
from typing import List

import tvm
from tvm.script import tir as T


# ---------------------------------------------------------------------------
# Matmul workloads at various sizes
# ---------------------------------------------------------------------------

@tvm.script.ir_module
class _Matmul256:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (256, 256), "float32")
        B = T.match_buffer(b, (256, 256), "float32")
        C = T.match_buffer(c, (256, 256), "float32")
        for i, j, k in T.grid(256, 256, 256):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class _Matmul512:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (512, 512), "float32")
        B = T.match_buffer(b, (512, 512), "float32")
        C = T.match_buffer(c, (512, 512), "float32")
        for i, j, k in T.grid(512, 512, 512):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class _Matmul1024:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


# ---------------------------------------------------------------------------
# Conv2d NCHW workloads (stride=1, no padding)
#
# Layout: A[N, C, H, W], W[CO, CI, KH, KW], B[N, CO, OH, OW]
# Output spatial: OH = H - KH + 1, OW = W - KW + 1
# Representative of ResNet-style layers.
# ---------------------------------------------------------------------------

@tvm.script.ir_module
class _Conv2d56:
    @T.prim_func
    def main(a: T.handle, w: T.handle, b: T.handle) -> None:
        # N=1, CI=64, H=W=56, CO=64, KH=KW=3 → output 1×64×54×54
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1, 64, 56, 56), "float32")
        W = T.match_buffer(w, (64, 64, 3, 3), "float32")
        B = T.match_buffer(b, (1, 64, 54, 54), "float32")
        for n, co, oh, ow, ci, kh, kw in T.grid(1, 64, 54, 54, 64, 3, 3):
            with T.block("conv2d"):
                vn, vco, voh, vow, vci, vkh, vkw = T.axis.remap(
                    "SSSSRRR", [n, co, oh, ow, ci, kh, kw]
                )
                with T.init():
                    B[vn, vco, voh, vow] = 0.0
                B[vn, vco, voh, vow] = (
                    B[vn, vco, voh, vow]
                    + A[vn, vci, voh + vkh, vow + vkw] * W[vco, vci, vkh, vkw]
                )


@tvm.script.ir_module
class _Conv2d28:
    @T.prim_func
    def main(a: T.handle, w: T.handle, b: T.handle) -> None:
        # N=1, CI=128, H=W=28, CO=128, KH=KW=3 → output 1×128×26×26
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1, 128, 28, 28), "float32")
        W = T.match_buffer(w, (128, 128, 3, 3), "float32")
        B = T.match_buffer(b, (1, 128, 26, 26), "float32")
        for n, co, oh, ow, ci, kh, kw in T.grid(1, 128, 26, 26, 128, 3, 3):
            with T.block("conv2d"):
                vn, vco, voh, vow, vci, vkh, vkw = T.axis.remap(
                    "SSSSRRR", [n, co, oh, ow, ci, kh, kw]
                )
                with T.init():
                    B[vn, vco, voh, vow] = 0.0
                B[vn, vco, voh, vow] = (
                    B[vn, vco, voh, vow]
                    + A[vn, vci, voh + vkh, vow + vkw] * W[vco, vci, vkh, vkw]
                )


@tvm.script.ir_module
class _Conv2dBiasAdd56:
    @T.prim_func
    def main(
        a: T.handle, w: T.handle, bias: T.handle, res: T.handle, b: T.handle
    ) -> None:
        # Conv2d (same spatial as _Conv2d56) + bias broadcast + residual add.
        # Models a ResNet-style: out = conv(a, w) + bias + res
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A    = T.match_buffer(a,    (1, 64, 56, 56), "float32")
        W    = T.match_buffer(w,    (64, 64, 3, 3),  "float32")
        Bias = T.match_buffer(bias, (64,),            "float32")
        Res  = T.match_buffer(res,  (1, 64, 54, 54), "float32")
        B    = T.match_buffer(b,    (1, 64, 54, 54), "float32")
        Conv = T.alloc_buffer(      (1, 64, 54, 54), "float32")
        for n, co, oh, ow, ci, kh, kw in T.grid(1, 64, 54, 54, 64, 3, 3):
            with T.block("conv2d"):
                vn, vco, voh, vow, vci, vkh, vkw = T.axis.remap(
                    "SSSSRRR", [n, co, oh, ow, ci, kh, kw]
                )
                with T.init():
                    Conv[vn, vco, voh, vow] = 0.0
                Conv[vn, vco, voh, vow] = (
                    Conv[vn, vco, voh, vow]
                    + A[vn, vci, voh + vkh, vow + vkw] * W[vco, vci, vkh, vkw]
                )
        for n, co, oh, ow in T.grid(1, 64, 54, 54):
            with T.block("bias_add"):
                vn, vco, voh, vow = T.axis.remap("SSSS", [n, co, oh, ow])
                B[vn, vco, voh, vow] = (
                    Conv[vn, vco, voh, vow] + Bias[vco] + Res[vn, vco, voh, vow]
                )


# ---------------------------------------------------------------------------
# Workload registry
# ---------------------------------------------------------------------------

_WORKLOAD_REGISTRY = {
    # Matmul: 2 * N^3 multiply-adds
    "matmul_256":  (_Matmul256,  256,  2 * 256**3),
    "matmul_1024": (_Matmul1024, 1024, 2 * 1024**3),
    # Conv2d NCHW: 2 * N * CO * OH * OW * CI * KH * KW multiply-adds
    "conv2d_56": (_Conv2d56, 56,
        2 * 1 * 64 * 54 * 54 * 64 * 3 * 3),           # ≈ 215 MFLOPs
    "conv2d_bias_add_56": (_Conv2dBiasAdd56, 56,
        2 * 1 * 64 * 54 * 54 * 64 * 3 * 3              # conv
        + 2 * 1 * 64 * 54 * 54),                        # bias + residual add
}


def get_workload(name: str) -> tvm.IRModule:
    """Get a workload IRModule by name.

    Parameters
    ----------
    name : str
        One of the names returned by list_workloads().

    Returns
    -------
    mod : tvm.IRModule
    """
    if name not in _WORKLOAD_REGISTRY:
        raise ValueError(
            f"Unknown workload: {name!r}. Available: {list_workloads()}"
        )
    mod, _, _ = _WORKLOAD_REGISTRY[name]
    return mod


def get_workload_flops(name: str) -> int:
    """Get the number of floating-point operations for a workload.

    Used for computing GFLOPS = flops / runtime / 1e9.
    """
    if name not in _WORKLOAD_REGISTRY:
        raise ValueError(
            f"Unknown workload: {name!r}. Available: {list_workloads()}"
        )
    _, _, flops = _WORKLOAD_REGISTRY[name]
    return flops


def list_workloads() -> List[str]:
    """List all available workload names."""
    return sorted(_WORKLOAD_REGISTRY.keys())
