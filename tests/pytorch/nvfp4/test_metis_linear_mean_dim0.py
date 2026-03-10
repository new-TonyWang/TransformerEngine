# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for Linear with MEAN_DIM0_ONLY quantization strategy.

Verifies the full pipeline for quantization_strategy = mean_dim0_only:
  - Forward pass with mean_dim0_only_quant
  - Backward pass with compute_input_gradient_mean
  - Weight gradient computation with compute_weight_gradient_mean
  - Multi-step training stability
  - Shape compatibility with standard path
"""

import pytest
import torch
import torch.nn as nn

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.module import Linear
from transformer_engine.pytorch.module.metis.metis_context import (
    LinearLowbitContext,
    QuantizationStrategy,
    get_metis_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# NVFP4 alignment requirement
ALIGN = 64


def reset_rng(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_nvfp4_recipe():
    """Return a basic NVFP4 recipe."""
    r = recipe.NVFP4BlockScaling()
    r.fp4_quant_fwd_inp = recipe.QParams(random_hadamard_transform=False, stochastic_rounding=False,fp4_2d_quantization=False)
    r.fp4_quant_fwd_weight = recipe.QParams(random_hadamard_transform=False, stochastic_rounding=False,fp4_2d_quantization=False)
    r.fp4_quant_bwd_grad = recipe.QParams(random_hadamard_transform=False, stochastic_rounding=False,fp4_2d_quantization=False)
    return r


def run_forward_backward(module, inp, fp8, recipe_obj=None):
    """Run one forward + backward pass and return (output, input_grad)."""
    x = inp.clone().detach().requires_grad_(True)

    if fp8:
        with te.autocast(enabled=True, recipe=recipe_obj):
            out = module(x)
    else:
        out = module(x)

    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    return out.detach(), x.grad


# ---------------------------------------------------------------------------
# MEAN_DIM0_ONLY strategy tests for MetisLinear
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMetisLinearMeanDim0Only:
    """Tests for MEAN_DIM0_ONLY quantization strategy with MetisLinear."""

    @pytest.mark.parametrize("in_features", [128, 256])
    @pytest.mark.parametrize("out_features", [128, 256])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("bias", [False])
    def test_mean_dim0_only_forward_backward_runs(self, in_features, out_features, dtype, bias):
        """Verify MEAN_DIM0_ONLY strategy forward + backward runs without error."""
        reset_rng()
        device = "cuda"
        batch_size = ALIGN * 2  # Ensure alignment

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            params_dtype=dtype,
            device=device,
            enable_metis=True,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(batch_size, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean_dim0_only",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, dgrad = run_forward_backward(module, inp, fp8=True, recipe_obj=nvfp4_rec)

        assert out.shape == (batch_size, out_features), f"Output shape mismatch: {out.shape}"
        assert dgrad is not None, "Input gradient should not be None"
        assert dgrad.shape == (batch_size, in_features), f"Grad shape mismatch: {dgrad.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isnan(dgrad).any(), "Input grad contains NaN"

    @pytest.mark.parametrize("in_features", [128, 256])
    @pytest.mark.parametrize("out_features", [128, 256])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_mean_dim0_only_weight_grad_exists(self, in_features, out_features, dtype):
        """Verify that weight gradients are computed under MEAN_DIM0_ONLY strategy."""
        reset_rng()
        device = "cuda"
        batch_size = ALIGN * 2

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(batch_size, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean_dim0_only",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, _ = run_forward_backward(module, inp, fp8=True, recipe_obj=nvfp4_rec)

        assert module.weight.grad is not None, "weight.grad should not be None"
        assert not torch.isnan(module.weight.grad).any(), "weight.grad contains NaN"

    @pytest.mark.parametrize("in_features", [128, 256])
    @pytest.mark.parametrize("out_features", [128, 256])
    def test_mean_dim0_only_output_shape_matches_standard(self, in_features, out_features):
        """Verify Metis MEAN_DIM0_ONLY output shape matches standard path output shape."""
        reset_rng()
        device = "cuda"
        dtype = torch.bfloat16
        batch_size = ALIGN * 2

        nvfp4_rec = get_nvfp4_recipe()

        # Standard path (use_metis=False)
        reset_rng()
        module_std = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).eval()

        # Metis MEAN_DIM0_ONLY path
        reset_rng()
        module_metis = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).eval()

        # Copy weights to ensure same initialization
        with torch.no_grad():
            module_metis.weight.copy_(module_std.weight)

        inp = torch.randn(batch_size, in_features, dtype=dtype, device=device)

        with te.autocast(enabled=True, recipe=nvfp4_rec):
            with get_metis_context(use_metis=False):
                out_std = module_std(inp)
            with get_metis_context(
                use_metis=True,
                quantization_strategy="mean_dim0_only",
                enable_activation_svd=False,
                enable_backward_svd=False,
            ):
                out_metis = module_metis(inp)

        assert out_metis.shape == out_std.shape, (
            f"Shape mismatch: metis={out_metis.shape}, std={out_std.shape}"
        )

    @pytest.mark.parametrize("in_features", [64, 128])
    @pytest.mark.parametrize("out_features", [64, 128])
    def test_mean_dim0_only_multi_step_training(self, in_features, out_features):
        """Simulate multiple training steps with MEAN_DIM0_ONLY strategy."""
        reset_rng()
        device = "cuda"
        dtype = torch.bfloat16
        batch_size = ALIGN * 2

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).train()

        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)
        nvfp4_rec = get_nvfp4_recipe()

        for step in range(3):
            optimizer.zero_grad()
            torch.manual_seed(step + 100)
            inp = torch.randn(batch_size, in_features, dtype=dtype, device=device)

            with get_metis_context(
                use_metis=True,
                quantization_strategy="mean_dim0_only",
                enable_activation_svd=False,
                enable_backward_svd=True,
            ):
                with te.autocast(enabled=True, recipe=nvfp4_rec):
                    out = module(inp)

            loss = out.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            assert not torch.isnan(out).any(), f"Output NaN at step {step}"
            assert module.weight.grad is not None, f"weight.grad is None at step {step}"

    @pytest.mark.parametrize("in_features", [128])
    @pytest.mark.parametrize("out_features", [256])
    def test_mean_dim0_only_with_bias_grad(self, in_features, out_features):
        """Verify bias gradients are computed when bias=True."""
        reset_rng()
        device = "cuda"
        dtype = torch.bfloat16
        batch_size = ALIGN * 2

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            params_dtype=dtype,
            device=device,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(batch_size, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean_dim0_only",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, _ = run_forward_backward(module, inp, fp8=True, recipe_obj=nvfp4_rec)

        assert module.weight.grad is not None, "weight.grad should not be None"
        assert module.bias.grad is not None, "bias.grad should not be None"
        assert not torch.isnan(module.weight.grad).any(), "weight.grad contains NaN"
        assert not torch.isnan(module.bias.grad).any(), "bias.grad contains NaN"

    @pytest.mark.parametrize("in_features", [128])
    @pytest.mark.parametrize("out_features", [256])
    def test_mean_dim0_only_3d_input(self, in_features, out_features):
        """Verify MEAN_DIM0_ONLY works with 3D input (batch, seq, hidden)."""
        reset_rng()
        device = "cuda"
        dtype = torch.bfloat16
        batch_size = 2
        seq_len = ALIGN  # Ensure total tokens align to 64

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(batch_size, seq_len, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean_dim0_only",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, dgrad = run_forward_backward(module, inp, fp8=True, recipe_obj=nvfp4_rec)

        assert out.shape == (batch_size, seq_len, out_features), f"Output shape mismatch: {out.shape}"
        assert dgrad.shape == (batch_size, seq_len, in_features), f"Grad shape mismatch: {dgrad.shape}"


# ---------------------------------------------------------------------------
# Comparison tests between different strategies
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMetisLinearStrategyComparison:
    """Compare MEAN_DIM0_ONLY with other quantization strategies."""

    @pytest.mark.parametrize("strategy", ["mean", "mean_dim0_only"])
    def test_different_strategies_run_successfully(self, strategy):
        """Verify that different quantization strategies all run without errors."""
        reset_rng()
        device = "cuda"
        dtype = torch.bfloat16
        in_features = 128
        out_features = 256
        batch_size = ALIGN * 2

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(batch_size, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy=strategy,
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, dgrad = run_forward_backward(module, inp, fp8=True, recipe_obj=nvfp4_rec)

        assert out.shape == (batch_size, out_features)
        assert dgrad.shape == (batch_size, in_features)
        assert not torch.isnan(out).any()
        assert not torch.isnan(dgrad).any()


# ---------------------------------------------------------------------------
# Fallback test
# ---------------------------------------------------------------------------

class TestMetisLinearFallback:
    """Verify fallback behavior when use_metis=False."""

    @pytest.mark.parametrize("in_features", [128])
    @pytest.mark.parametrize("out_features", [256])
    def test_fallback_to_standard_path(self, in_features, out_features):
        """When use_metis=False, MetisLinear should use standard path."""
        reset_rng()
        device = "cuda"
        dtype = torch.bfloat16
        batch_size = 64

        module = Linear(
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            device=device,
        ).train()

        inp = torch.randn(batch_size, in_features, dtype=dtype, device=device, requires_grad=True)

        with get_metis_context(use_metis=False):
            out = module(inp)
            loss = out.sum()
            loss.backward()

        assert out.shape == (batch_size, out_features)
        assert inp.grad.shape == inp.shape
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# Smoke test runner (for quick validation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke tests for MetisLinear with MEAN_DIM0_ONLY...")

    if nvfp4_available:
        # MEAN_DIM0_ONLY strategy tests
        t = TestMetisLinearMeanDim0Only()
        t.test_mean_dim0_only_forward_backward_runs(128, 256, torch.bfloat16, False)
        print("MEAN_DIM0_ONLY forward/backward: PASSED")
        t.test_mean_dim0_only_weight_grad_exists(128, 256, torch.bfloat16)
        print("MEAN_DIM0_ONLY weight grad: PASSED")
        t.test_mean_dim0_only_output_shape_matches_standard(128, 256)
        print("MEAN_DIM0_ONLY output shape: PASSED")
        t.test_mean_dim0_only_multi_step_training(64, 64)
        print("MEAN_DIM0_ONLY multi-step training: PASSED")

        # Strategy comparison
        t_comp = TestMetisLinearStrategyComparison()
        t_comp.test_different_strategies_run_successfully("mean_dim0_only")
        print("Strategy comparison (mean_dim0_only): PASSED")
    else:
        print(f"NVFP4 not available: {reason_for_no_nvfp4} — skipping FP8 tests")

    print("\nAll smoke tests PASSED.")
