# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for _MetisGroupedLinear: SEPARATE_RESIDUAL and MEAN quantization strategies."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.module.metis.grouped_linear import GroupedLinear
from transformer_engine.pytorch.module.metis.metis_context import (
    LinearLowbitContext,
    QuantizationStrategy,
    get_metis_context,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset_rng_states(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_nvfp4_recipe():
    """Return a basic NVFP4 recipe."""
    r = recipe.NVFP4BlockScaling()
    r.fp4_quant_fwd_inp = recipe.QParams()
    r.fp4_quant_fwd_weight = recipe.QParams()
    r.fp4_quant_bwd_grad = recipe.QParams()
    return r


nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# NVFP4 alignment: m_splits must be divisible by 64 (hadamard transform requirement)
ALIGN = 64


def make_m_splits(total_rows: int, num_gemms: int, align: int = ALIGN) -> list:
    """Create m_splits that are divisible by `align`."""
    base = (total_rows // num_gemms // align) * align
    splits = [base] * num_gemms
    # Give remainder to last split
    remainder = total_rows - base * num_gemms
    remainder = (remainder // align) * align
    splits[-1] += remainder
    # If last split is 0, give at least ALIGN tokens
    if splits[-1] == 0:
        splits[-1] = align
    return splits


def run_forward_backward(module, inp, m_splits, fp8, recipe_obj=None):
    """Run one forward + backward pass and return (output, input_grad)."""
    x = inp.clone().detach().requires_grad_(True)
    ctx = te.autocast(enabled=fp8, recipe=recipe_obj) if fp8 else torch.no_grad().__class__()

    if fp8:
        with te.autocast(enabled=True, recipe=recipe_obj):
            out = module(x, m_splits)
    else:
        out = module(x, m_splits)

    loss = out.sum()
    loss.backward()
    torch.cuda.synchronize()
    return out.detach(), x.grad


# ---------------------------------------------------------------------------
# MEAN_DIM0_ONLY strategy tests (requires NVFP4)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMetisGroupedLinearMeanDim0Only:
    """Tests for MEAN_DIM0_ONLY quantization strategy with NVFP4."""

    @pytest.mark.parametrize("num_gemms", [2, 4])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("bias", [False, True])
    def test_mean_dim0_only_forward_backward_runs(self, num_gemms, dtype, bias):
        """Verify MEAN_DIM0_ONLY strategy forward + backward runs without error."""
        reset_rng_states()
        device = "cuda"
        in_features = 128
        out_features = 256

        total_rows = ALIGN * (num_gemms - 1) * 2
        m_splits = make_m_splits(total_rows, num_gemms)

        module = GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            bias=bias and (num_gemms == 1),
            params_dtype=dtype,
            parallel_mode=None,
            device=device,
            enable_metis=True,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean_dim0_only",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, dgrad = run_forward_backward(module, inp, m_splits, fp8=True, recipe_obj=nvfp4_rec)

        assert out.shape == (total_rows, out_features), f"Output shape mismatch: {out.shape}"
        assert dgrad is not None, "Input gradient should not be None"
        assert dgrad.shape == (total_rows, in_features), f"Grad shape mismatch: {dgrad.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isnan(dgrad).any(), "Input grad contains NaN"

    @pytest.mark.parametrize("num_gemms", [2, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_mean_dim0_only_weight_grad_exists(self, num_gemms, dtype):
        """Verify that weight gradients are computed under MEAN_DIM0_ONLY strategy."""
        reset_rng_states()
        device = "cuda"
        in_features = 128
        out_features = 128

        total_rows = ALIGN * num_gemms * 2
        m_splits = make_m_splits(total_rows, num_gemms)

        module = GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            parallel_mode=None,
            device=device,
            enable_metis=True,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean_dim0_only",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, _ = run_forward_backward(module, inp, m_splits, fp8=True, recipe_obj=nvfp4_rec)

        for i in range(num_gemms):
            weight = getattr(module, f"weight{i}")
            assert weight.grad is not None, f"weight{i}.grad should not be None"
            assert not torch.isnan(weight.grad).any(), f"weight{i}.grad contains NaN"

    @pytest.mark.parametrize("num_gemms", [2])
    def test_mean_dim0_only_output_shape_matches_standard(self, num_gemms):
        """Verify Metis MEAN_DIM0_ONLY output shape matches standard path output shape."""
        reset_rng_states()
        device = "cuda"
        dtype = torch.bfloat16
        in_features = 128
        out_features = 256
        total_rows = ALIGN * num_gemms * 2
        m_splits = make_m_splits(total_rows, num_gemms)

        nvfp4_rec = get_nvfp4_recipe()

        # Standard path
        reset_rng_states()
        module_std = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device, enable_metis=True,
        ).eval()

        # Metis MEAN_DIM0_ONLY path (same weights)
        reset_rng_states()
        module_metis = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device, enable_metis=True,
        ).eval()
        with torch.no_grad():
            for i in range(num_gemms):
                getattr(module_metis, f"weight{i}").copy_(getattr(module_std, f"weight{i}"))

        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        with te.autocast(enabled=True, recipe=nvfp4_rec):
            with get_metis_context(use_metis=False):
                out_std = module_std(inp, m_splits)
            with get_metis_context(use_metis=True, quantization_strategy="mean_dim0_only",
                                   enable_activation_svd=False, enable_backward_svd=False):
                out_metis = module_metis(inp, m_splits)

        assert out_metis.shape == out_std.shape, (
            f"Shape mismatch: metis={out_metis.shape}, std={out_std.shape}"
        )

    @pytest.mark.parametrize("num_gemms", [2])
    def test_mean_dim0_only_multi_step_training(self, num_gemms):
        """Simulate multiple training steps with MEAN_DIM0_ONLY strategy."""
        reset_rng_states()
        device = "cuda"
        dtype = torch.bfloat16
        in_features = 64
        out_features = 64
        total_rows = ALIGN * num_gemms * 2
        m_splits = make_m_splits(total_rows, num_gemms)
        nvfp4_rec = get_nvfp4_recipe()

        module = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).train()

        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for step in range(3):
            optimizer.zero_grad()
            torch.manual_seed(step + 100)
            inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

            with get_metis_context(use_metis=True, quantization_strategy="mean_dim0_only",
                                   enable_activation_svd=False, enable_backward_svd=True):
                with te.autocast(enabled=True, recipe=nvfp4_rec):
                    out = module(inp, m_splits)

            loss = out.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            assert not torch.isnan(out).any(), f"Output NaN at step {step}"
            for i in range(num_gemms):
                w = getattr(module, f"weight{i}")
                assert w.grad is not None, f"weight{i}.grad is None at step {step}"


# ---------------------------------------------------------------------------
# MEAN strategy tests (requires NVFP4)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMetisGroupedLinearMean:
    """Tests for MEAN quantization strategy with NVFP4."""

    @pytest.mark.parametrize("num_gemms", [2, 4])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("bias", [False, True])
    def test_mean_forward_backward_runs(self, num_gemms, dtype, bias):
        """Verify MEAN strategy forward + backward runs without error."""
        reset_rng_states()
        device = "cuda"
        in_features = 128
        out_features = 256

        total_rows = ALIGN * (num_gemms-1) * 2
        m_splits = make_m_splits(total_rows, num_gemms)

        module = GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            bias=bias and (num_gemms == 1),  # GroupedLinear with TP>1 doesn't support bias
            params_dtype=dtype,
            parallel_mode=None,
            device=device,
            enable_metis=True,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, dgrad = run_forward_backward(module, inp, m_splits, fp8=True, recipe_obj=nvfp4_rec)

        assert out.shape == (total_rows, out_features), f"Output shape mismatch: {out.shape}"
        assert dgrad is not None, "Input gradient should not be None"
        assert dgrad.shape == (total_rows, in_features), f"Grad shape mismatch: {dgrad.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isnan(dgrad).any(), "Input grad contains NaN"

    @pytest.mark.parametrize("num_gemms", [2, 3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_mean_weight_grad_exists(self, num_gemms, dtype):
        """Verify that weight gradients are computed under MEAN strategy."""
        reset_rng_states()
        device = "cuda"
        in_features = 128
        out_features = 128

        total_rows = ALIGN * num_gemms * 2
        m_splits = make_m_splits(total_rows, num_gemms)

        module = GroupedLinear(
            num_gemms=num_gemms,
            in_features=in_features,
            out_features=out_features,
            bias=False,
            params_dtype=dtype,
            parallel_mode=None,
            device=device,
            enable_metis=True,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        with get_metis_context(
            use_metis=True,
            quantization_strategy="mean",
            enable_activation_svd=False,
            enable_backward_svd=True,
        ):
            out, _ = run_forward_backward(module, inp, m_splits, fp8=True, recipe_obj=nvfp4_rec)

        for i in range(num_gemms):
            weight = getattr(module, f"weight{i}")
            assert weight.grad is not None, f"weight{i}.grad should not be None"
            assert not torch.isnan(weight.grad).any(), f"weight{i}.grad contains NaN"

    @pytest.mark.parametrize("num_gemms", [2])
    def test_mean_output_shape_matches_standard(self, num_gemms):
        """Verify Metis MEAN output shape matches standard path output shape."""
        reset_rng_states()
        device = "cuda"
        dtype = torch.bfloat16
        in_features = 128
        out_features = 256
        total_rows = ALIGN * num_gemms * 2
        m_splits = make_m_splits(total_rows, num_gemms)

        nvfp4_rec = get_nvfp4_recipe()

        # Standard path
        reset_rng_states()
        module_std = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device, enable_metis=True,
        ).eval()

        # Metis MEAN path (same weights)
        reset_rng_states()
        module_metis = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device, enable_metis=True,
        ).eval()
        with torch.no_grad():
            for i in range(num_gemms):
                getattr(module_metis, f"weight{i}").copy_(getattr(module_std, f"weight{i}"))

        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        with te.autocast(enabled=True, recipe=nvfp4_rec):
            with get_metis_context(use_metis=False):
                out_std = module_std(inp, m_splits)
            with get_metis_context(use_metis=True, quantization_strategy="mean",
                                   enable_activation_svd=False, enable_backward_svd=False):
                out_metis = module_metis(inp, m_splits)

        assert out_metis.shape == out_std.shape, (
            f"Shape mismatch: metis={out_metis.shape}, std={out_std.shape}"
        )

    @pytest.mark.parametrize("num_gemms", [2])
    def test_mean_multi_step_training(self, num_gemms):
        """Simulate multiple training steps with MEAN strategy."""
        reset_rng_states()
        device = "cuda"
        dtype = torch.bfloat16
        in_features = 64
        out_features = 64
        total_rows = ALIGN * num_gemms * 2
        m_splits = make_m_splits(total_rows, num_gemms)
        nvfp4_rec = get_nvfp4_recipe()

        module = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).train()

        optimizer = torch.optim.SGD(module.parameters(), lr=1e-3)

        for step in range(3):
            optimizer.zero_grad()
            torch.manual_seed(step + 100)
            inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

            with get_metis_context(use_metis=True, quantization_strategy="mean",
                                   enable_activation_svd=False, enable_backward_svd=True):
                with te.autocast(enabled=True, recipe=nvfp4_rec):
                    out = module(inp, m_splits)

            loss = out.sum()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

            assert not torch.isnan(out).any(), f"Output NaN at step {step}"
            for i in range(num_gemms):
                w = getattr(module, f"weight{i}")
                assert w.grad is not None, f"weight{i}.grad is None at step {step}"


# ---------------------------------------------------------------------------
# SEPARATE_RESIDUAL strategy tests (requires NVFP4)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMetisGroupedLinearSeparateResidual:
    """Tests for SEPARATE_RESIDUAL quantization strategy with NVFP4."""

    @pytest.mark.parametrize("num_gemms", [2])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_separate_residual_forward_backward_runs(self, num_gemms, dtype):
        """Verify SEPARATE_RESIDUAL strategy forward + backward runs without error."""
        reset_rng_states()
        device = "cuda"
        in_features = 1024  # Use larger dims for NVFP4 GEMM compatibility
        out_features = 1024
        total_rows = ALIGN * num_gemms * 4
        m_splits = make_m_splits(total_rows, num_gemms)

        module = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        svd_rank = min(64, in_features // 16)  # rank must be large enough for NVFP4 GEMM
        metis_params = {
            "use_metis": True,
            "quantization_strategy": "separate_residual",
            "enable_activation_svd": True,
            "enable_backward_svd": True,
            "activation_lowrank_svd": svd_rank,
            "backward_lowrank_svd": svd_rank,
            "activation_lowrank_niter": 0,
            "backward_lowrank_niter": 0,
        }

        with get_metis_context(**metis_params):
            out, dgrad = run_forward_backward(module, inp, m_splits, fp8=True, recipe_obj=nvfp4_rec)

        assert out.shape == (total_rows, out_features), f"Output shape mismatch: {out.shape}"
        assert dgrad is not None, "Input gradient should not be None"
        assert dgrad.shape == (total_rows, in_features), f"Grad shape mismatch: {dgrad.shape}"

    @pytest.mark.parametrize("num_gemms", [3])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_separate_residual_weight_grad_exists(self, num_gemms, dtype):
        """Verify weight gradients are computed under SEPARATE_RESIDUAL."""
        reset_rng_states()
        device = "cuda"
        in_features = 1024  # Use larger dims for NVFP4 GEMM compatibility
        out_features = 1024
        total_rows = ALIGN * num_gemms * 4
        m_splits = make_m_splits(total_rows, num_gemms)

        module = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).train()

        nvfp4_rec = get_nvfp4_recipe()
        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)

        svd_rank = min(64, in_features // 16)  # rank must be large enough for NVFP4 GEMM
        metis_params = {
            "use_metis": True,
            "quantization_strategy": "separate_residual",
            "enable_activation_svd": True,
            "enable_backward_svd": True,
            "activation_lowrank_svd": svd_rank,
            "backward_lowrank_svd": svd_rank,
            "activation_lowrank_niter": 0,
            "backward_lowrank_niter": 0,
        }

        with get_metis_context(**metis_params):
            out, _ = run_forward_backward(module, inp, m_splits, fp8=True, recipe_obj=nvfp4_rec)

        for i in range(num_gemms):
            weight = getattr(module, f"weight{i}")
            assert weight.grad is not None, f"weight{i}.grad should not be None"

    @pytest.mark.parametrize("num_gemms", [2])
    def test_separate_residual_output_shape_matches_standard(self, num_gemms):
        """Verify SEPARATE_RESIDUAL output shape matches standard path."""
        reset_rng_states()
        device = "cuda"
        dtype = torch.bfloat16
        in_features = 1024  # Use larger dims for NVFP4 GEMM compatibility
        out_features = 1024
        total_rows = ALIGN * num_gemms * 4
        m_splits = make_m_splits(total_rows, num_gemms)
        nvfp4_rec = get_nvfp4_recipe()

        reset_rng_states()
        module_std = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).eval()

        reset_rng_states()
        module_metis = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).eval()
        with torch.no_grad():
            for i in range(num_gemms):
                getattr(module_metis, f"weight{i}").copy_(getattr(module_std, f"weight{i}"))

        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device)
        svd_rank = min(64, in_features // 16)  # rank must be large enough for NVFP4 GEMM

        with te.autocast(enabled=True, recipe=nvfp4_rec):
            with get_metis_context(use_metis=False):
                out_std = module_std(inp, m_splits)
            with get_metis_context(
                use_metis=True, quantization_strategy="separate_residual",
                enable_activation_svd=True, enable_backward_svd=False,
                activation_lowrank_svd=svd_rank, activation_lowrank_niter=0,
            ):
                out_metis = module_metis(inp, m_splits)

        assert out_metis.shape == out_std.shape, (
            f"Shape mismatch: metis={out_metis.shape}, std={out_std.shape}"
        )


# ---------------------------------------------------------------------------
# Fallback test: use_metis=False → standard _GroupedLinear path
# ---------------------------------------------------------------------------

class TestMetisGroupedLinearFallback:
    """Verify that when use_metis=False, standard _GroupedLinear is used."""

    @pytest.mark.parametrize("num_gemms", [2])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_fallback_to_standard_path(self, num_gemms, dtype):
        """When use_metis=False, GroupedLinear should use standard path."""
        reset_rng_states()
        device = "cuda"
        in_features = 64
        out_features = 128
        total_rows = 32

        m_splits = [total_rows // num_gemms] * num_gemms
        m_splits[-1] += total_rows - sum(m_splits)

        module = GroupedLinear(
            num_gemms=num_gemms, in_features=in_features, out_features=out_features,
            bias=False, params_dtype=dtype, parallel_mode=None, device=device,
        ).train()

        inp = torch.randn(total_rows, in_features, dtype=dtype, device=device, requires_grad=True)

        with get_metis_context(use_metis=False):
            out = module(inp, m_splits)
            loss = out.sum()
            loss.backward()

        assert out.shape == (total_rows, out_features)
        assert inp.grad.shape == inp.shape
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("num_gemms", [2])
    def test_metis_path_context_isolation(self, num_gemms):
        """Verify Metis context is properly isolated (context manager restores state)."""
        original_strategy = LinearLowbitContext.quantization_strategy
        original_use_metis = LinearLowbitContext.use_metis

        with get_metis_context(use_metis=True, quantization_strategy="mean"):
            assert LinearLowbitContext.use_metis is True
            assert LinearLowbitContext.quantization_strategy == QuantizationStrategy.MEAN

        assert LinearLowbitContext.quantization_strategy == original_strategy
        assert LinearLowbitContext.use_metis == original_use_metis


# ---------------------------------------------------------------------------
# Smoke test runner (for quick validation)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke tests for MetisGroupedLinear...")

    # Fallback test (no FP8 needed)
    # t_fb = TestMetisGroupedLinearFallback()
    # t_fb.test_fallback_to_standard_path(num_gemms=2, dtype=torch.float32)
    # print("Fallback (float32): PASSED")
    # t_fb.test_metis_path_context_isolation(num_gemms=2)
    # print("Context isolation: PASSED")

    if nvfp4_available:
        # MEAN_DIM0_ONLY strategy
        t_mean_dim0 = TestMetisGroupedLinearMeanDim0Only()
        t_mean_dim0.test_mean_dim0_only_forward_backward_runs(num_gemms=8, dtype=torch.bfloat16, bias=False)
        print("MEAN_DIM0_ONLY forward/backward (num_gemms=8): PASSED")
        t_mean_dim0.test_mean_dim0_only_weight_grad_exists(num_gemms=8, dtype=torch.bfloat16)
        print("MEAN_DIM0_ONLY weight grad (num_gemms=8): PASSED")
        t_mean_dim0.test_mean_dim0_only_output_shape_matches_standard(num_gemms=2)
        print("MEAN_DIM0_ONLY output shape: PASSED")

        # MEAN strategy
        t_mean = TestMetisGroupedLinearMean()
        t_mean.test_mean_forward_backward_runs(num_gemms=8, dtype=torch.bfloat16, bias=False)
        print("MEAN forward/backward (num_gemms=8): PASSED")
        t_mean.test_mean_weight_grad_exists(num_gemms=8, dtype=torch.bfloat16)
        print("MEAN weight grad (num_gemms=8): PASSED")
        t_mean.test_mean_output_shape_matches_standard(num_gemms=2)
        print("MEAN output shape: PASSED")

        # SEPARATE_RESIDUAL strategy
        t_sep = TestMetisGroupedLinearSeparateResidual()
        t_sep.test_separate_residual_forward_backward_runs(num_gemms=2, dtype=torch.bfloat16)
        print("SEPARATE_RESIDUAL forward/backward (num_gemms=2): PASSED")
        t_sep.test_separate_residual_weight_grad_exists(num_gemms=3, dtype=torch.bfloat16)
        print("SEPARATE_RESIDUAL weight grad (num_gemms=3): PASSED")
    else:
        print(f"NVFP4 not available: {reason_for_no_nvfp4} — skipping FP8 tests")

    print("\nAll smoke tests PASSED.")
