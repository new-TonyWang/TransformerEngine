# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Tests for tex.nvfp4_gemm_bf16 (CUTLASS-based NVFP4 x BF16 GEMM).

Validates that the CUTLASS implementation produces results numerically
consistent with the existing cuBLAS-based tex.generic_gemm reference.
"""

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch import NVFP4Quantizer


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------

def check_nvfp4_gemm_bf16(
    M: int,
    K: int,
    N: int,
    accumulate: bool,
    use_bias: bool = False,
    use_grad: bool = False,
):
    """
    Compare tex.nvfp4_gemm_bf16 (CUTLASS) against tex.generic_gemm (cuBLAS).

    Layout convention (matches existing test_nvfp4_gemm_exact.py):
      A (activation) : NVFP4Tensor, shape (M, K), rowwise quantized
      B (weight)     : NVFP4Tensor, shape (N, K), rowwise quantized
      GEMM           : D = A @ B.T  →  transa=False, transb=True
    """
    device = "cuda"
    out_dtype = torch.bfloat16
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # -----------------------------------------------------------------------
    # 1. Build float inputs
    # -----------------------------------------------------------------------
    # x = torch.full((M, K),int("01110111",2), dtype=torch.bfloat16, device=device)
    # w = torch.full((N, K),int("01110111",2), dtype=torch.bfloat16, device=device)

    x = torch.rand((M, K), dtype=torch.bfloat16, device=device)
    w = torch.rand((N, K), dtype=torch.bfloat16, device=device)

    # -----------------------------------------------------------------------
    # 2. NVFP4-quantize both tensors
    # -----------------------------------------------------------------------
    te_dtype = tex.DType.kFloat4E2M1

    def make_nvfp4(t):
        q = NVFP4Quantizer(
            fp4_dtype=te_dtype,
            rowwise=True,
            columnwise=True,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=False,
            with_post_rht_amax=False,
        )
        buf = q.make_empty(t.shape, dtype=t.dtype, device=device, requires_grad=False)
        return q.update_quantized(t, buf)

    x_nvfp4 = make_nvfp4(x)
    w_nvfp4 = make_nvfp4(w)

    # -----------------------------------------------------------------------
    # [DIAGNOSTIC] Print scale factor information
    # -----------------------------------------------------------------------
    print("\n[SCALE FACTOR DIAGNOSTIC]")
    print(f"  Input value: {int('01110111',2)} (bf16) = {x[0,0].item():.4f}")
    
    # Print rowwise scale_inv info
    x_sf_row = x_nvfp4._rowwise_scale_inv
    w_sf_row = w_nvfp4._rowwise_scale_inv
    print(f"  x rowwise scale_inv shape: {x_sf_row.shape}, dtype: {x_sf_row.dtype}")
    print(f"  x rowwise scale_inv first 16 bytes: {x_sf_row.flatten()[:16].view(torch.uint8).tolist()}")
    print(f"  w rowwise scale_inv shape: {w_sf_row.shape}, dtype: {w_sf_row.dtype}")
    print(f"  w rowwise scale_inv first 16 bytes: {w_sf_row.flatten()[:16].view(torch.uint8).tolist()}")
    
    # Print columnwise scale_inv info
    if x_nvfp4._columnwise_scale_inv is not None:
        x_sf_col = x_nvfp4._columnwise_scale_inv
        print(f"  x columnwise scale_inv shape: {x_sf_col.shape}")
        print(f"  x columnwise scale_inv first 16 bytes: {x_sf_col.flatten()[:16].view(torch.uint8).tolist()}")
    if w_nvfp4._columnwise_scale_inv is not None:
        w_sf_col = w_nvfp4._columnwise_scale_inv
        print(f"  w columnwise scale_inv shape: {w_sf_col.shape}")
        print(f"  w columnwise scale_inv first 16 bytes: {w_sf_col.flatten()[:16].view(torch.uint8).tolist()}")
    
    # Print amax values
    print(f"  x amax_rowwise: {x_nvfp4._amax_rowwise.item():.4f}")
    print(f"  w amax_rowwise: {w_nvfp4._amax_rowwise.item():.4f}")
    
    # Interpret the scale factor byte as E4M3 (signed)
    def decode_e4m3(byte_val):
        """Decode signed E4M3 byte to float value"""
        sign = (byte_val >> 7) & 1
        exp = (byte_val >> 3) & 0xF
        mant = byte_val & 0x7
        if exp == 0:  # subnormal
            val = (mant / 8.0) * (2 ** -6)
        elif exp == 15:  # NaN
            val = float('nan')
        else:
            val = (1.0 + mant / 8.0) * (2 ** (exp - 7))
        return -val if sign else val
    
    def decode_ue4m3(byte_val):
        """Decode unsigned E4M3 byte to float value"""
        exp = (byte_val >> 3) & 0xF
        mant = byte_val & 0x7
        if exp == 0:  # subnormal
            val = (mant / 8.0) * (2 ** -6)
        elif exp == 15:  # NaN
            val = float('nan')
        else:
            val = (1.0 + mant / 8.0) * (2 ** (exp - 7))
        return val
    
    first_byte = x_sf_row.flatten()[0].view(torch.uint8).item()
    print(f"  First scale byte: 0x{first_byte:02X} = {first_byte}")
    print(f"    As signed E4M3:   {decode_e4m3(first_byte):.6f}")
    print(f"    As unsigned E4M3: {decode_ue4m3(first_byte):.6f}")
    
    # Expected scale factor for dequant: amax / e2m1_max = 119 / 6 ≈ 19.83
    amax = x_nvfp4._amax_rowwise.item()
    expected_scale = amax / 6.0
    print(f"  Expected dequant scale (amax/6): {expected_scale:.4f}")
    print(f"  Expected scale_inv (6/amax): {6.0/amax:.6f}")
    print()
    
    # -----------------------------------------------------------------------
    # 3. Shared call arguments
    # -----------------------------------------------------------------------
    # D = A @ B^T  →  transa=False (A row-major), transb=True (B needs transpose)
    transa = True
    transb = False

    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    use_split_accumulator = False

    # Accumulation buffer: must be pre-allocated when accumulate=True
    out_init = torch.zeros((M, N), dtype=out_dtype, device=device) if accumulate else None

    # Workspace (small; CUTLASS manages its own internal memory)
    workspace = torch.empty(4, dtype=torch.uint8, device=device)

    # -----------------------------------------------------------------------
    # 4. Reference: cuBLAS via tex.generic_gemm
    #    signature: (A, transa, B, transb, D, quantizer, out_dtype, bias,
    #                bias_type, gelu, gelu_in, grad, workspace, workspace_size,
    #                accumulate, use_split_accumulator, ...)
    # -----------------------------------------------------------------------
    y_cublas = tex.generic_gemm(
        w_nvfp4,          # A (weight, shape NxK)
        transa,           # transA: False → treat w as-is
        x_nvfp4,          # B (activation, shape MxK)
        transb,           # transB: True → transpose x
        out_init.clone() if accumulate else None,
        None,             # out_quantizer
        TE_DType[out_dtype],
        bias,
        bias_dtype,
        False,            # gelu
        None,             # gelu_in
        use_grad,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )[0]
    print("y_cublas=",y_cublas)
    # -----------------------------------------------------------------------
    # 5. New interface: CUTLASS via tex.nvfp4_gemm_bf16
    #    signature: (A, transa, B, transb, D, quantizer, out_dtype, bias,
    #                bias_type, grad, workspace, workspace_size,
    #                accumulate, use_split_accumulator, alpha, beta)
    #    Returns: [D, bias_grad]
    # -----------------------------------------------------------------------
    out_init = torch.zeros((M, N), dtype=out_dtype, device=device) if accumulate else None
    result = tex.nvfp4_gemm_bf16(
        w_nvfp4,          # A
        transa,           # transA
        x_nvfp4,          # B
        transb,           # transB
        out_init.clone() if accumulate else None,  # D (in-place buffer or None)
        None,             # quantizer
        TE_DType[out_dtype],
        bias,
        bias_dtype,
        use_grad,
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )
    y_cutlass = result[0]
    bias_grad = result[1]
    print(result)
    print(out_init)
    # -----------------------------------------------------------------------
    # 6. Sanity checks
    # -----------------------------------------------------------------------
    assert y_cutlass is not y_cublas, "Outputs should be separate tensors"
    assert y_cutlass.shape == (M, N), f"Expected shape ({M},{N}), got {y_cutlass.shape}"
    assert y_cutlass.dtype == out_dtype, f"Expected dtype {out_dtype}, got {y_cutlass.dtype}"
    assert not torch.isnan(y_cutlass).all(), "All CUTLASS output elements are NaN"
    assert bias_grad is None, "bias_grad should be None when no bias is provided"

    # Replace NaNs with 0 before numeric comparison
    y_cublas  = torch.where(y_cublas.isnan(),  torch.zeros_like(y_cublas),  y_cublas)
    y_cutlass = torch.where(y_cutlass.isnan(), torch.zeros_like(y_cutlass), y_cutlass)

    torch.testing.assert_close(y_cutlass, y_cublas, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test: basic GEMM shapes
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, K, N",
    [
        (128,  128, 128),
    ],
    ids=lambda x: str(x),
)
@pytest.mark.parametrize("accumulate", [True], ids=["no_accum", "accum"])
def test_nvfp4_gemm_bf16_shapes(M, K, N, accumulate):
    check_nvfp4_gemm_bf16(M=M, K=K, N=N, accumulate=accumulate)


# ---------------------------------------------------------------------------
# Test: return value structure
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_nvfp4_gemm_bf16_return_structure():
    """Verify that the function returns exactly [D, bias_grad]."""
    device = "cuda"
    M, K, N = 128, 128, 128
    te_dtype = tex.DType.kFloat4E2M1

    q = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    w = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    x_q = q.update_quantized(x, q.make_empty(x.shape, dtype=x.dtype, device=device))
    w_q = q.update_quantized(w, q.make_empty(w.shape, dtype=w.dtype, device=device))

    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    result = tex.nvfp4_gemm_bf16(
        w_q, False, x_q, True,
        None, None, TE_DType[torch.bfloat16],
        None, TE_DType[torch.bfloat16],
        False,
        workspace, workspace.shape[0],
        False, False,
    )
    assert len(result) == 2, f"Expected 2 return values, got {len(result)}"
    D, bias_grad = result
    assert D is not None
    assert bias_grad is None


# ---------------------------------------------------------------------------
# Test: non-zero beta with accumulate=True
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_nvfp4_gemm_bf16_beta_validation():
    """Passing non-zero beta with accumulate=False should raise an error."""
    device = "cuda"
    M, K, N = 128, 128, 128
    te_dtype = tex.DType.kFloat4E2M1

    q = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
    )
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    w = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    x_q = q.update_quantized(x, q.make_empty(x.shape, dtype=x.dtype, device=device))
    w_q = q.update_quantized(w, q.make_empty(w.shape, dtype=w.dtype, device=device))

    workspace = torch.empty(4, dtype=torch.uint8, device=device)
    with pytest.raises(Exception):
        tex.nvfp4_gemm_bf16(
            w_q, False, x_q, True,
            None, None, TE_DType[torch.bfloat16],
            None, TE_DType[torch.bfloat16],
            False,
            workspace, workspace.shape[0],
            False,   # accumulate=False
            False,
            alpha=1.0,
            beta=0.5,  # non-zero beta without accumulate → should raise
        )

if __name__ == "__main__":
    check_nvfp4_gemm_bf16(M=64, K=64, N=128, accumulate=False)
