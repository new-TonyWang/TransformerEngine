# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for MeanDim0QuantResult and MetisMeanFunction.mean_dim0_only_quant.

Verifies the full pipeline for quantization_strategy = mean_dim0_only:
  - MeanDim0QuantResult dataclass and its methods
  - mean_dim0_only_quant quantization method
  - gemm_operation_with_mean_quant (with MeanDim0QuantResult)
  - compute_input_gradient_mean (with MeanDim0QuantResult)
  - compute_weight_gradient_mean (with MeanDim0QuantResult)
  - grouped_gemm_operation_with_mean_quant (with MeanDim0QuantResult)
  - compute_input_gradient_mean_grouped_gemm (with MeanDim0QuantResult)
  - compute_weight_gradient_mean_grouped_gemm (with MeanDim0QuantResult)
  - QuantizationStrategy.MEAN_DIM0_ONLY enum value
"""

import pytest
import torch
import torch.nn.functional as F

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.module.metis.quant import (
    MetisMeanFunction,
    MeanDim0QuantResult,
)
from transformer_engine.pytorch.module.metis.metis_context import QuantizationStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# NVFP4 GEMM 要求 M 维度对齐到 64
ALIGN = 64


def reset_rng(seed: int = 1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_nvfp4_quantizer():
    """创建 NVFP4 量化器（rowwise + columnwise）。"""
    return NVFP4Quantizer(rowwise=True, columnwise=True)


def make_aligned_input(batch, hidden, align=ALIGN, dtype=torch.bfloat16, device="cuda"):
    """创建第 0 维对齐到 align 的 2D 随机张量。"""
    rows = ((batch + align - 1) // align) * align
    return torch.randn(rows, hidden, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# 1. QuantizationStrategy 枚举值测试
# ---------------------------------------------------------------------------

class TestQuantizationStrategyEnum:
    """验证 MEAN_DIM0_ONLY 枚举值正确注册。"""

    def test_enum_value_exists(self):
        assert hasattr(QuantizationStrategy, "MEAN_DIM0_ONLY")

    def test_enum_value_string(self):
        assert QuantizationStrategy.MEAN_DIM0_ONLY.value == "mean_dim0_only"

    def test_from_string(self):
        strategy = QuantizationStrategy.from_string("mean_dim0_only")
        assert strategy == QuantizationStrategy.MEAN_DIM0_ONLY

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            QuantizationStrategy.from_string("non_existent_strategy")


# ---------------------------------------------------------------------------
# 2. MeanDim0QuantResult 数据类测试（不依赖 NVFP4 GEMM）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMeanDim0QuantResult:
    """验证 MeanDim0QuantResult 数据类的方法。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.B, self.S, self.H = 2, 64, 128  # 总行数 B*S = 128 = 2 * ALIGN
        self.quantizer = make_nvfp4_quantizer()

    def _make_result(self):
        inp = torch.randn(self.B * self.S, self.H, dtype=self.dtype, device=self.device)
        return MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)

    def test_result_type(self):
        result = self._make_result()
        assert isinstance(result, MeanDim0QuantResult)

    def test_quant_input_shape(self):
        result = self._make_result()
        assert result.quant_input.size(0) == self.B * self.S
        assert result.quant_input.size(1) == self.H

    def test_mean_dim0_shape(self):
        result = self._make_result()
        assert result.input_tensor_mean_dim0.shape == (1, self.H)

    def test_shape_field(self):
        inp = torch.randn(self.B, self.S, self.H, dtype=self.dtype, device=self.device)
        result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        assert result.shape == (self.B, self.S, self.H)

    def test_get_quant_mean_tensor_shape(self):
        """get_quant_mean_tensor 应返回 [64, H] 的量化张量（padding 到 64）。"""
        result = self._make_result()
        qm = result.get_quant_mean_tensor()
        # mean 原始为 [1, H]，padding 到 64 的倍数 = 64
        assert qm.size(0) == 64
        assert qm.size(1) == self.H

    def test_get_quant_mean_expanded_shape(self):
        """get_quant_mean_expanded 应返回 [B*S, H] 的量化张量。"""
        result = self._make_result()
        qm_exp = result.get_quant_mean_expanded()
        assert qm_exp.size(0) == self.B * self.S
        assert qm_exp.size(1) == self.H

    def test_mean_dim0_correctness(self):
        """验证 dim=0 均值的数值正确性。"""
        inp = torch.randn(self.B * self.S, self.H, dtype=torch.float32, device=self.device)
        result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        expected_mean = inp.mean(dim=0, keepdim=True)  # [1, H]
        torch.testing.assert_close(
            result.input_tensor_mean_dim0.float(),
            expected_mean,
            rtol=1e-5, atol=1e-5,
        )

    def test_padding_to_64(self):
        """验证 get_quant_mean_tensor 中 padding 为 64 行。"""
        result = self._make_result()
        qm = result.get_quant_mean_tensor()
        assert qm.size(0) % 64 == 0

    def test_clear(self):
        """clear() 后 shape 应为 None，张量数据应释放。"""
        result = self._make_result()
        result.clear()
        assert result.shape is None

    def test_prepare_and_restore_for_saving(self):
        """prepare_for_saving / restore_from_saved 往返一致性。"""
        inp = torch.randn(self.B * self.S, self.H, dtype=self.dtype, device=self.device)
        result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)

        original_mean = result.input_tensor_mean_dim0.clone()
        tensors, meta = result.prepare_for_saving()

        # prepare 后 mean_dim0 应被置为 None
        assert meta.input_tensor_mean_dim0 is None

        # 恢复
        meta.restore_from_saved(tensors)
        assert meta.input_tensor_mean_dim0 is not None
        torch.testing.assert_close(meta.input_tensor_mean_dim0, original_mean)


# ---------------------------------------------------------------------------
# 3. mean_dim0_only_quant 功能测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMeanDim0OnlyQuant:
    """验证 mean_dim0_only_quant 方法。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.quantizer = make_nvfp4_quantizer()

    @pytest.mark.parametrize("shape", [
        (128, 256),        # 2D 输入
        (2, 64, 256),      # 3D 输入，b=2, s=64, h=256
        (4, 32, 128),      # 3D 输入，b=4, s=32, h=128
    ])
    def test_various_input_shapes(self, shape):
        inp = torch.randn(*shape, dtype=torch.bfloat16, device=self.device)
        result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        assert isinstance(result, MeanDim0QuantResult)
        assert result.shape == shape
        hidden = shape[-1]
        total_rows = 1
        for d in shape[:-1]:
            total_rows *= d
        assert result.quant_input.size(0) == total_rows
        assert result.quant_input.size(1) == hidden

    def test_only_dim0_mean_computed(self):
        """验证仅计算 dim=0 均值（形状为 [1, H]，不含 dim=1 均值）。"""
        inp = torch.randn(128, 256, dtype=torch.bfloat16, device=self.device)
        result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        # dim=0 均值形状应为 [1, H]
        assert result.input_tensor_mean_dim0.shape == (1, 256)


# ---------------------------------------------------------------------------
# 4. gemm_operation_with_mean_dim0_quant 测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestGemmWithMeanDim0Quant:
    """验证 gemm_operation_with_mean_quant (with MeanDim0QuantResult) 的输出形状和数值合理性。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.quantizer = make_nvfp4_quantizer()

    def _run_gemm(self, b, s, h, out_features):
        inp = make_aligned_input(b * s, h, device=self.device, dtype=self.dtype)
        weight = make_aligned_input(out_features, h, align=ALIGN, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)

        quant_result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        out = MetisMeanFunction.gemm_operation_with_mean_quant(
            quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        return inp, weight, out

    @pytest.mark.parametrize("b,s,h,out_features", [
        (2, 64, 128, 256),
        (4, 64, 256, 128),
    ])
    def test_output_shape_2d_input(self, b, s, h, out_features):
        """2D 输入时输出形状应为 [b*s, out_features]。"""
        rows = ((b * s + ALIGN - 1) // ALIGN) * ALIGN
        inp = make_aligned_input(rows, h, device=self.device, dtype=self.dtype)
        weight = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)

        quant_result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        out = MetisMeanFunction.gemm_operation_with_mean_quant(
            quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert out.shape == (rows, out_features), f"Expected ({rows}, {out_features}), got {out.shape}"

    @pytest.mark.parametrize("b,s,h,out_features", [
        (2, 64, 128, 256),
    ])
    def test_output_shape_3d_input(self, b, s, h, out_features):
        """3D 输入时输出形状应保持 3D：[b, s, out_features]。"""
        inp = torch.randn(b, s, h, dtype=self.dtype, device=self.device)
        weight = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)

        quant_result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        out = MetisMeanFunction.gemm_operation_with_mean_quant(
            quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert out.shape == (b, s, out_features), f"Expected ({b}, {s}, {out_features}), got {out.shape}"

    def test_output_dtype(self):
        inp = make_aligned_input(128, 128, device=self.device, dtype=self.dtype)
        weight = make_aligned_input(64, 128, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)
        quant_result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        out = MetisMeanFunction.gemm_operation_with_mean_quant(
            quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert out.dtype == self.dtype

    def test_mean_subtraction_effect(self):
        """验证均值减法确实改变了输出（不等于直接量化输入的 GEMM 结果）。"""
        inp = make_aligned_input(128, 128, device=self.device, dtype=self.dtype)
        weight = make_aligned_input(64, 128, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)
        quant_result = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)

        out_mean = MetisMeanFunction.gemm_operation_with_mean_quant(
            quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        # 不带均值减法的普通 GEMM（仅 quant_input @ weight.T）
        from transformer_engine.pytorch.module.metis.quant import MetisSvdFunction
        out_plain = MetisSvdFunction.svd_quant_gemm(
            quant_weight, quant_result.quant_input, self.dtype, None, layout="TN"
        )
        # 两者应不相等（均值减法有实际效果）
        assert not torch.allclose(out_mean, out_plain, atol=1e-3), \
            "Mean subtraction should produce different output from plain GEMM"


# ---------------------------------------------------------------------------
# 5. compute_input_gradient_mean_dim0 测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestComputeInputGradientMeanDim0:
    """验证 compute_input_gradient_mean (with MeanDim0QuantResult) 的输出形状和数值合理性。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.quantizer = make_nvfp4_quantizer()

    @pytest.mark.parametrize("rows,out_features,h", [
        (128, 64, 256),
        (256, 128, 128),
    ])
    def test_output_shape_2d(self, rows, out_features, h):
        """2D 情形：dx 形状应为 [rows, h]。"""
        grad_out = make_aligned_input(rows, out_features, device=self.device, dtype=self.dtype)
        weight = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)

        grad_quant_result = MetisMeanFunction.mean_dim0_only_quant(grad_out, self.quantizer)
        dx = MetisMeanFunction.compute_input_gradient_mean(
            grad_quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert dx.shape == (rows, h), f"Expected ({rows}, {h}), got {dx.shape}"
        assert dx.dtype == self.dtype

    def test_output_shape_3d(self):
        """3D 情形：dx 形状应与原始输入 shape 对应。"""
        b, s, out_features, h = 2, 64, 128, 256
        grad_out = torch.randn(b, s, out_features, dtype=self.dtype, device=self.device)
        weight = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
        quant_weight = self.quantizer(weight)

        grad_quant_result = MetisMeanFunction.mean_dim0_only_quant(grad_out, self.quantizer)
        dx = MetisMeanFunction.compute_input_gradient_mean(
            grad_quant_result, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert dx.shape == (b, s, h), f"Expected ({b}, {s}, {h}), got {dx.shape}"


# ---------------------------------------------------------------------------
# 6. compute_weight_gradient_mean_dim0 测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestComputeWeightGradientMeanDim0:
    """验证 compute_weight_gradient_mean (with MeanDim0QuantResult) 的输出形状和数值合理性。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.quantizer = make_nvfp4_quantizer()

    @pytest.mark.parametrize("rows,h,out_features", [
        (128, 256, 64),
        (256, 128, 128),
    ])
    def test_output_shape(self, rows, h, out_features):
        """dw 形状应为 [out_features, h]。"""
        inp = make_aligned_input(rows, h, device=self.device, dtype=self.dtype)
        grad_out = make_aligned_input(rows, out_features, device=self.device, dtype=self.dtype)

        x_quant = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        dy_quant = MetisMeanFunction.mean_dim0_only_quant(grad_out, self.quantizer)

        dw = MetisMeanFunction.compute_weight_gradient_mean(
            x_quant, dy_quant, activation_dtype=torch.float32, quantizer=None
        )
        assert dw.shape == (out_features, h), f"Expected ({out_features}, {h}), got {dw.shape}"

    def test_output_not_all_zeros(self):
        """权重梯度不应全为零（随机输入下）。"""
        rows, h, out_features = 128, 128, 64
        inp = make_aligned_input(rows, h, device=self.device, dtype=self.dtype)
        grad_out = make_aligned_input(rows, out_features, device=self.device, dtype=self.dtype)

        x_quant = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        dy_quant = MetisMeanFunction.mean_dim0_only_quant(grad_out, self.quantizer)

        dw = MetisMeanFunction.compute_weight_gradient_mean(
            x_quant, dy_quant, activation_dtype=torch.float32, quantizer=None
        )
        assert not torch.all(dw == 0), "Weight gradient should not be all zeros"


# ---------------------------------------------------------------------------
# 7. Grouped GEMM 测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestGroupedGemmMeanDim0:
    """验证 Grouped GEMM 系列方法（forward + dgrad + wgrad）with MeanDim0QuantResult。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.quantizer = make_nvfp4_quantizer()

    def _make_splits(self, num_experts, rows_each, h, out_features):
        """构造 num_experts 个量化结果。"""
        quant_results = []
        weight_list = []
        m_splits = []
        for _ in range(num_experts):
            inp = make_aligned_input(rows_each, h, device=self.device, dtype=self.dtype)
            quant_results.append(
                MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
            )
            w = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
            weight_list.append(self.quantizer(w))
            m_splits.append(rows_each)
        return quant_results, weight_list, m_splits

    @pytest.mark.parametrize("num_experts,rows_each,h,out_features", [
        (2, 128, 128, 64),
        (4, 64, 256, 128),
    ])
    def test_grouped_gemm_forward_shape(self, num_experts, rows_each, h, out_features):
        """grouped_gemm_operation_with_mean_dim0_quant 输出形状验证。"""
        quant_results, weight_list, m_splits = self._make_splits(
            num_experts, rows_each, h, out_features
        )
        outputs = MetisMeanFunction.grouped_gemm_operation_with_mean_quant(
            quant_results, weight_list, self.dtype, m_splits
        )
        assert len(outputs) == num_experts
        for i, out in enumerate(outputs):
            assert out.shape == (rows_each, out_features), \
                f"Expert {i}: expected ({rows_each}, {out_features}), got {out.shape}"

    @pytest.mark.parametrize("num_experts", [2, 3])
    def test_grouped_gemm_forward_with_zero_split(self, num_experts):
        """m_splits 中有 0 时，对应输出应为空张量。"""
        h, out_features, rows_each = 128, 64, 128
        quant_results, weight_list, m_splits = self._make_splits(
            num_experts, rows_each, h, out_features
        )
        # 将第 0 个 split 置为 0
        m_splits[0] = 0
        inp_zero = torch.empty(0, h, dtype=self.dtype, device=self.device)
        quant_results[0] = MetisMeanFunction.mean_dim0_only_quant(inp_zero, self.quantizer) \
            if inp_zero.numel() > 0 else quant_results[0]

        outputs = MetisMeanFunction.grouped_gemm_operation_with_mean_quant(
            quant_results, weight_list, self.dtype, m_splits
        )
        assert len(outputs) == num_experts
        # split=0 的输出行数为 0
        assert outputs[0].shape[0] == 0

    @pytest.mark.parametrize("num_experts,rows_each,h,out_features", [
        (2, 128, 256, 64),
    ])
    def test_grouped_gemm_dgrad_shape(self, num_experts, rows_each, h, out_features):
        """compute_input_gradient_mean_dim0_grouped_gemm 输出形状验证。"""
        grad_results = []
        weight_list = []
        m_splits = []
        for _ in range(num_experts):
            g = make_aligned_input(rows_each, out_features, device=self.device, dtype=self.dtype)
            grad_results.append(
                MetisMeanFunction.mean_dim0_only_quant(g, self.quantizer)
            )
            w = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
            weight_list.append(self.quantizer(w))
            m_splits.append(rows_each)

        dx_list = MetisMeanFunction.compute_input_gradient_mean_grouped_gemm(
            grad_results, weight_list, self.dtype, m_splits
        )
        assert len(dx_list) == num_experts
        for i, dx in enumerate(dx_list):
            assert dx.shape == (rows_each, h), \
                f"Expert {i}: expected ({rows_each}, {h}), got {dx.shape}"

    @pytest.mark.parametrize("num_experts,rows_each,h,out_features", [
        (2, 128, 256, 64),
    ])
    def test_grouped_gemm_wgrad_shape(self, num_experts, rows_each, h, out_features):
        """compute_weight_gradient_mean_dim0_grouped_gemm 输出形状验证。"""
        x_list, dy_list, m_splits = [], [], []
        for _ in range(num_experts):
            inp = make_aligned_input(rows_each, h, device=self.device, dtype=self.dtype)
            x_list.append(MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer))
            g = make_aligned_input(rows_each, out_features, device=self.device, dtype=self.dtype)
            dy_list.append(MetisMeanFunction.mean_dim0_only_quant(g, self.quantizer))
            m_splits.append(rows_each)

        dw_list = MetisMeanFunction.compute_weight_gradient_mean_grouped_gemm(
            x_list, dy_list, torch.float32, m_splits
        )
        assert len(dw_list) == num_experts
        for i, dw in enumerate(dw_list):
            assert dw.shape == (out_features, h), \
                f"Expert {i}: expected ({out_features}, {h}), got {dw.shape}"

    def test_all_zero_splits_returns_empty(self):
        """m_splits 全为 0 时所有输出应为空张量。"""
        num_experts, h, out_features = 3, 128, 64
        quant_results, weight_list, m_splits = self._make_splits(
            num_experts, 128, h, out_features
        )
        m_splits_zero = [0] * num_experts

        out_fwd = MetisMeanFunction.grouped_gemm_operation_with_mean_quant(
            quant_results, weight_list, self.dtype, m_splits_zero
        )
        out_dgrad = MetisMeanFunction.compute_input_gradient_mean_grouped_gemm(
            quant_results, weight_list, self.dtype, m_splits_zero
        )

        for i in range(num_experts):
            assert out_fwd[i].shape[0] == 0
            assert out_dgrad[i].shape[0] == 0


# ---------------------------------------------------------------------------
# 8. 完整前向 + 反向流程集成测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestMeanDim0FullPipeline:
    """端到端流程：量化 → GEMM → 梯度计算。"""

    @pytest.fixture(autouse=True)
    def setup(self):
        reset_rng()
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.quantizer = make_nvfp4_quantizer()

    def test_full_forward_backward_pipeline(self):
        """验证前向 GEMM 和两个反向梯度（dgrad + wgrad）流程正常运行。"""
        rows, h, out_features = 128, 256, 128

        inp = make_aligned_input(rows, h, device=self.device, dtype=self.dtype)
        weight = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
        grad_out = make_aligned_input(rows, out_features, device=self.device, dtype=self.dtype)

        quant_weight = self.quantizer(weight)

        # 前向量化
        x_quant = MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer)
        # 前向 GEMM
        out = MetisMeanFunction.gemm_operation_with_mean_quant(
            x_quant, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert out.shape == (rows, out_features)

        # 反向：量化梯度输出
        dy_quant = MetisMeanFunction.mean_dim0_only_quant(grad_out, self.quantizer)

        # 反向 dgrad：dx = (dy - dy_mean) @ weight
        dx = MetisMeanFunction.compute_input_gradient_mean(
            dy_quant, quant_weight, activation_dtype=self.dtype, quantizer=None
        )
        assert dx.shape == (rows, h)

        # 反向 wgrad：dw = (dy - dy_mean).T @ (x - x_mean)
        dw = MetisMeanFunction.compute_weight_gradient_mean(
            x_quant, dy_quant, activation_dtype=torch.float32, quantizer=None
        )
        assert dw.shape == (out_features, h)

        # 确保所有结果不含 NaN 或 Inf
        assert torch.isfinite(out).all(), "Forward output contains non-finite values"
        assert torch.isfinite(dx).all(), "Input gradient contains non-finite values"
        assert torch.isfinite(dw).all(), "Weight gradient contains non-finite values"

    def test_grouped_full_pipeline(self):
        """验证 Grouped GEMM 前向 + dgrad + wgrad 完整流程。"""
        num_experts, rows_each, h, out_features = 2, 128, 256, 64

        x_list, dy_list, weight_list, m_splits = [], [], [], []
        for _ in range(num_experts):
            inp = make_aligned_input(rows_each, h, device=self.device, dtype=self.dtype)
            g = make_aligned_input(rows_each, out_features, device=self.device, dtype=self.dtype)
            w = make_aligned_input(out_features, h, device=self.device, dtype=self.dtype)
            x_list.append(MetisMeanFunction.mean_dim0_only_quant(inp, self.quantizer))
            dy_list.append(MetisMeanFunction.mean_dim0_only_quant(g, self.quantizer))
            weight_list.append(self.quantizer(w))
            m_splits.append(rows_each)

        # 前向 grouped GEMM
        fwd_out = MetisMeanFunction.grouped_gemm_operation_with_mean_quant(
            x_list, weight_list, self.dtype, m_splits
        )
        # 反向 dgrad grouped GEMM
        dx_list = MetisMeanFunction.compute_input_gradient_mean_grouped_gemm(
            dy_list, weight_list, self.dtype, m_splits
        )
        # 反向 wgrad grouped GEMM
        dw_list = MetisMeanFunction.compute_weight_gradient_mean_grouped_gemm(
            x_list, dy_list, torch.float32, m_splits
        )

        for i in range(num_experts):
            assert fwd_out[i].shape == (rows_each, out_features)
            assert dx_list[i].shape == (rows_each, h)
            assert dw_list[i].shape == (out_features, h)
            assert torch.isfinite(fwd_out[i]).all()
            assert torch.isfinite(dx_list[i]).all()
            assert torch.isfinite(dw_list[i]).all()
