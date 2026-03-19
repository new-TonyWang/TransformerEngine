# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API"""
from typing import Union, Optional, Callable, Tuple, List
from itertools import chain
import warnings

import functools
import torch

import transformer_engine_torch as tex

from transformer_engine.common.recipe import Recipe
from ..base import (
    get_dummy_wgrad,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from .._common import WeightGradStore
from ...quantization import FP8GlobalStateManager
from ...utils import (
    divide,
    cast_if_needed,
    clear_tensor_data,
    init_method_constant,
    requires_grad,
    get_nvtx_range_context,
    nvtx_range_push,
    nvtx_range_pop,
)
from ...distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ...cpp_extensions import (
    general_grouped_gemm,
)
from ...constants import GemmParallelModes, dist_group_type
from ...jit import no_torch_dynamo
from ...cpu_offload import is_cpu_offload_enabled, mark_not_offload, start_offload

from ...tensor.float8_tensor import Float8CurrentScalingQuantizer, Float8Quantizer
from ...quantized_tensor import (
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_saved,
)
from ....debug.pytorch.debug_quantization import DebugQuantizer
from ....debug.pytorch.debug_state import TEDebugState

# Metis imports
from .quant import MetisSvdFunction, MetisMeanFunction
from .metis_context import LinearLowbitContext, QuantizationStrategy

# Re-export standard GroupedLinear; its forward() dispatches to _MetisGroupedLinear
# when LinearLowbitContext.use_metis is True.
from ..grouped_linear import GroupedLinear  # noqa: F401

__all__ = ["GroupedLinear"]

class _MetisGroupedLinear(torch.autograd.Function):
    """Metis GroupedLinear semi-top level module.
    Applies Metis quantization (SEPARATE_RESIDUAL or MEAN strategy) per m_split.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        non_tensor_args: Tuple,
        *weights_and_biases,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        (
            m_splits,
            use_bias,
            is_first_microbatch,
            fp8,
            fp8_calibration,
            wgrad_store,
            input_quantizers,
            weight_quantizers,
            output_quantizers,
            grad_input_quantizers,
            grad_weight_quantizers,
            grad_output_quantizers,
            fuse_wgrad_accumulation,
            cpu_offloading,
            sequence_parallel,
            activation_dtype,
            is_grad_enabled,
            module,
            skip_fp8_weight_update,
            save_original_input,
            debug,
            enable_metis,
            svd_grad_output_histories,
        ) = non_tensor_args

        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]
        device = inp.device
        weight_requires_grad = weights[0].requires_grad

        in_features = weights[0].size(-1)
        if inp.size(-1) != in_features:
            raise ValueError(
                f"Input tensor (shape={tuple(inp.size())}) is not compatible with "
                f"weight tensor (shape={tuple(weights[0].size())})"
            )
        inp_view = inp.reshape(-1, in_features)

        # Capture current Metis context
        metis_ctx = LinearLowbitContext().clone()
        current_forward_use_metis = (
            enable_metis and metis_ctx.use_metis and metis_ctx.enable_activation_svd
        )

        # Split input by m_splits for per-split quantization
        inp_splits = list(torch.split(inp_view, list(m_splits)))

        # ------------------------------------------------------
        # Per-split input quantization
        # ------------------------------------------------------
        nvtx_range_push("_MetisGroupedLinear.forward.input_quant")
        if (
            current_forward_use_metis
            and fp8
            and metis_ctx.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL
        ):
            ug_sg_list, vg_list, res_list, restore_info_list = [], [], [], []
            for i in range(num_gemms):
                split = inp_splits[i]
                quantizer = input_quantizers[i]
                quantizer.set_usage(rowwise=True, columnwise=True)
                ug_sg, vg, res, restore_info = MetisSvdFunction.svd_lowrank_quant_separate_residual(
                    split,
                    quantizer,
                    rank=metis_ctx.activation_lowrank_svd,
                    niter=metis_ctx.activation_lowrank_niter,
                    token_drop_rate=metis_ctx.activation_token_drop_rate,
                    broadcast_dim=metis_ctx.activation_broadcast_dim,
                    restore_strategy=metis_ctx.activation_restore_strategy,
                    load_history=metis_ctx.load_history,
                    is_backward=False,
                    use_power_iteration_svd=metis_ctx.use_power_iteration_svd,
                    power_iteration_time=metis_ctx.power_iteration_time,
                    enable_history_optimization=metis_ctx.enable_history_optimization,
                    tp_size=1,
                    tp_group=None,
                    tp_strategy=metis_ctx.tp_strategy,
                    parallel_mode=None,
                )
                ug_sg_list.append(ug_sg)
                vg_list.append(vg)
                res_list.append(res)
                restore_info_list.append(restore_info)
        elif metis_ctx.quantization_strategy == QuantizationStrategy.MEAN:
            mean_quant_results = []
            for i in range(num_gemms):
                split = inp_splits[i]
                quantizer = input_quantizers[i]
                result = MetisMeanFunction.mean_split_dim_quant(split, quantizer)
                mean_quant_results.append(result)
        elif metis_ctx.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
            mean_dim0_only_quant_results = []
            for i in range(num_gemms):
                split = inp_splits[i]
                quantizer = input_quantizers[i]
                result = MetisMeanFunction.mean_dim0_only_quant(split, quantizer)
                mean_dim0_only_quant_results.append(result)
        else:
            raise ValueError(
                f"Unsupported quantization strategy: {metis_ctx.quantization_strategy}"
            )
        nvtx_range_pop("_MetisGroupedLinear.forward.input_quant")

        # ------------------------------------------------------
        # Prepare weights (same as _GroupedLinear)
        # ------------------------------------------------------
        if weight_quantizers[0] is not None:
            columnwise_usage = is_grad_enabled and inp.requires_grad
            if not columnwise_usage:
                columnwise_usage = (
                    is_fp8_activation_recompute_enabled()
                    and not in_fp8_activation_recompute_phase()
                )
            if not isinstance(weights[0], QuantizedTensorStorage):
                for wq in weight_quantizers:
                    wq.set_usage(rowwise=True, columnwise=columnwise_usage)
            else:
                weight_quantizers = [w._quantizer for w in weights]

        update_workspace = is_first_microbatch is None or is_first_microbatch
        weights_fp8 = []
        for i in range(num_gemms):
            weight_fp8 = module.get_weight_workspace(
                tensor=weights[i],
                quantizer=weight_quantizers[i],
                cache_name=(None if is_first_microbatch is None else f"weight{i}"),
                update_workspace=update_workspace,
                skip_update_flag=skip_fp8_weight_update,
                workspace_dtype=activation_dtype,
            )
            weight_fp8.update_usage(rowwise_usage=True)
            weights_fp8.append(weight_fp8)

        # Bias
        bias_dtype = activation_dtype
        if fp8 and activation_dtype == torch.float32:
            bias_dtype = torch.bfloat16
        biases = [cast_if_needed(b, bias_dtype) for b in biases] if use_bias else list(biases)

        # ------------------------------------------------------
        # Per-split GEMM (batched via grouped_gemm)
        # ------------------------------------------------------
        nvtx_range_push("_MetisGroupedLinear.forward.gemm")
        if metis_ctx.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL:
            out_parts = MetisSvdFunction.grouped_gemm_with_separate_residual(
                ug_sg_list,
                vg_list,
                res_list,
                weights_fp8,
                activation_dtype,
                input_quantizers,
                list(m_splits),
                is_grad=False,
                restore_info_list=restore_info_list,
                restore_strategy=metis_ctx.activation_restore_strategy,
            )
            if use_bias:
                for i in range(num_gemms):
                    if m_splits[i] > 0:
                        out_parts[i] = out_parts[i] + biases[i]
        elif metis_ctx.quantization_strategy == QuantizationStrategy.MEAN:
            out_parts = MetisMeanFunction.grouped_gemm_operation_with_mean_quant(
                mean_quant_results,
                weights_fp8,
                activation_dtype,
                list(m_splits),
            )
            if use_bias:
                for i in range(num_gemms):
                    if m_splits[i] > 0:
                        out_parts[i] = out_parts[i] + biases[i]
        elif metis_ctx.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
            out_parts = MetisMeanFunction.grouped_gemm_operation_with_mean_quant(
                mean_dim0_only_quant_results,
                weights_fp8,
                activation_dtype,
                list(m_splits),
            )
            if use_bias:
                for i in range(num_gemms):
                    if m_splits[i] > 0:
                        out_parts[i] = out_parts[i] + biases[i]
        nvtx_range_pop("_MetisGroupedLinear.forward.gemm")

        out = torch.cat(out_parts, dim=0)

        # ------------------------------------------------------
        # Save context for backward
        # ------------------------------------------------------
        if is_grad_enabled:
            ctx.metis_context = metis_ctx
            ctx.enable_metis = enable_metis
            ctx.svd_grad_output_histories = svd_grad_output_histories
            ctx.weight_quantizers = weight_quantizers
            ctx.weights_shape_1 = weights[0].shape[1]

            if metis_ctx.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL:
                ctx.restore_info_list = restore_info_list
                tensors_to_save, tensor_objects = prepare_for_saving(
                    *ug_sg_list,
                    *vg_list,
                    *res_list,
                    *weights_fp8,
                    *weights,
                    *biases,
                )
            elif metis_ctx.quantization_strategy == QuantizationStrategy.MEAN:
                ctx.restore_info_list = None
                tensors_to_save, tensor_objects = prepare_for_saving(
                    *mean_quant_results,
                    *weights_fp8,
                    *weights,
                    *biases,
                )
            elif metis_ctx.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
                ctx.restore_info_list = None
                tensors_to_save, tensor_objects = prepare_for_saving(
                    *mean_dim0_only_quant_results,
                    *weights_fp8,
                    *weights,
                    *biases,
                )

            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.grad_input_quantizers = grad_input_quantizers
            ctx.grad_output_quantizers = grad_output_quantizers
            ctx.grad_weight_quantizers = grad_weight_quantizers

            ctx.weights_requires_grad = weight_requires_grad
            if fuse_wgrad_accumulation and ctx.weights_requires_grad:
                if hasattr(weights[0], "__fsdp_param__"):
                    ctx.main_grad_funcs = [weights[i].get_main_grad for i in range(num_gemms)]
                else:
                    ctx.main_grad_funcs = [
                        lambda j=i: weights[j].main_grad for i in range(num_gemms)
                    ]
            else:
                ctx.main_grad_funcs = [lambda: None for _ in range(num_gemms)]

            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.inp_shape = inp.shape
            ctx.requires_dgrad = inp.requires_grad
            ctx.wgrad_store = wgrad_store
            ctx.device = device

        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with get_nvtx_range_context("_MetisGroupedLinear_backward"):
            N = ctx.num_gemms
            saved_tensors = restore_from_saved(ctx.tensor_objects, ctx.saved_tensors)
            ctx.tensor_objects = None

            if ctx.metis_context.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL:
                ug_sg_list = list(saved_tensors[: N])
                vg_list = list(saved_tensors[N : 2 * N])
                res_list = list(saved_tensors[2 * N : 3 * N])
                weights_fp8 = list(saved_tensors[3 * N : 4 * N])
                weights = list(saved_tensors[4 * N : 5 * N])
                biases = list(saved_tensors[5 * N : 6 * N])
            elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN:
                mean_quant_results = list(saved_tensors[: N])
                weights_fp8 = list(saved_tensors[N : 2 * N])
                weights = list(saved_tensors[2 * N : 3 * N])
                biases = list(saved_tensors[3 * N : 4 * N])
                mean_dim0_only_quant_results = [None] * N
            elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
                mean_dim0_only_quant_results = list(saved_tensors[: N])
                weights_fp8 = list(saved_tensors[N : 2 * N])
                weights = list(saved_tensors[2 * N : 3 * N])
                biases = list(saved_tensors[3 * N : 4 * N])
                mean_quant_results = [None] * N
            else:
                raise ValueError(
                    f"Unsupported quantization strategy: {ctx.metis_context.quantization_strategy}"
                )

            main_grads = [fn() for fn in ctx.main_grad_funcs]
            if ctx.fuse_wgrad_accumulation:
                for i in range(N):
                    if weights[i] is not None:
                        weights[i].main_grad = main_grads[i]

            # Split grad_output by m_splits
            grad_output_view = grad_output.contiguous().view(-1, grad_output.shape[-1])
            grad_output_splits = list(torch.split(grad_output_view, list(ctx.m_splits)))

            # --------------------------------------------------
            # Quantize grad_output per split
            # --------------------------------------------------
            nvtx_range_push("_MetisGroupedLinear.backward.grad_output_quant")
            output_grad_ug_sg_list = [None] * N
            output_grad_vg_list = [None] * N
            output_grad_res_list = [None] * N
            output_grad_restore_info_list = [None] * N
            grad_output_mean_quant_results = [None] * N
            grad_output_mean_dim0_only_quant_results = [None] * N

            if ctx.enable_metis and ctx.metis_context.use_metis and ctx.metis_context.enable_backward_svd:
                if (
                    ctx.metis_context.backward_lowrank_svd > 0
                    and ctx.metis_context.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL
                ):
                    for i in range(N):
                        if ctx.m_splits[i] == 0:
                            continue
                        grad_split = grad_output_splits[i]
                        quantizer = ctx.grad_output_quantizers[i]
                        if quantizer is not None:
                            quantizer.set_usage(rowwise=True, columnwise=True)
                        ug_sg, vg, res, restore_info = (
                            MetisSvdFunction.svd_lowrank_quant_separate_residual_backward(
                                grad_split,
                                quantizer,
                                rank=ctx.metis_context.backward_lowrank_svd,
                                niter=ctx.metis_context.backward_lowrank_niter,
                                token_drop_rate=ctx.metis_context.backward_token_drop_rate,
                                broadcast_dim=ctx.metis_context.backward_broadcast_dim,
                                enable_history_optimization=ctx.metis_context.enable_gradient_accumulation_optimization,
                                use_power_iteration_svd=ctx.metis_context.use_grad_power_iteration_svd,
                                power_iteration_time=ctx.metis_context.grad_power_iteration_time,
                                load_history=ctx.metis_context.load_history,
                                history_list=ctx.svd_grad_output_histories[i],
                                tp_size=1,
                                tp_group=None,
                                tp_strategy=ctx.metis_context.tp_strategy,
                                parallel_mode="",
                            )
                        )
                        output_grad_ug_sg_list[i] = ug_sg
                        output_grad_vg_list[i] = vg
                        output_grad_res_list[i] = res
                        output_grad_restore_info_list[i] = restore_info
                elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN:
                    for i in range(N):
                        if ctx.m_splits[i] == 0:
                            continue
                        grad_split = grad_output_splits[i]
                        quantizer = ctx.grad_output_quantizers[i]
                        result = MetisMeanFunction.mean_quant(grad_split, quantizer)
                        grad_output_mean_quant_results[i] = result
                elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
                    for i in range(N):
                        if ctx.m_splits[i] == 0:
                            continue
                        grad_split = grad_output_splits[i]
                        quantizer = ctx.grad_output_quantizers[i]
                        result = MetisMeanFunction.mean_dim0_only_quant(grad_split, quantizer)
                        grad_output_mean_dim0_only_quant_results[i] = result
            nvtx_range_pop("_MetisGroupedLinear.backward.grad_output_quant")

            # --------------------------------------------------
            # Compute dgrad per split (batched via grouped_gemm)
            # --------------------------------------------------
            dgrad_parts = []
            if ctx.requires_dgrad:
                nvtx_range_push("_MetisGroupedLinear.backward.dgrad_gemm")
                if ctx.metis_context.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL:
                    dgrad_parts = MetisSvdFunction.grouped_gemm_with_separate_residual(
                        output_grad_ug_sg_list,
                        output_grad_vg_list,
                        output_grad_res_list,
                        weights_fp8,
                        ctx.activation_dtype,
                        ctx.grad_output_quantizers,
                        list(ctx.m_splits),
                        is_grad=True,
                        restore_info_list=output_grad_restore_info_list,
                        restore_strategy=ctx.metis_context.backward_restore_strategy,
                    )
                elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN:
                    dgrad_parts = MetisMeanFunction.compute_input_gradient_mean_grouped_gemm(
                        grad_output_mean_quant_results,
                        weights_fp8,
                        ctx.activation_dtype,
                        list(ctx.m_splits),
                    )
                elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
                    dgrad_parts = MetisMeanFunction.compute_input_gradient_mean_grouped_gemm(
                        grad_output_mean_dim0_only_quant_results,
                        weights_fp8,
                        ctx.activation_dtype,
                        list(ctx.m_splits),
                    )
                dgrad = torch.cat(dgrad_parts, dim=0)
                nvtx_range_pop("_MetisGroupedLinear.backward.dgrad_gemm")

            # --------------------------------------------------
            # Compute wgrad per split (batched via grouped_gemm)
            # --------------------------------------------------
            wgrad_list = [None] * N
            grad_biases = [None] * N
            if ctx.weights_requires_grad:
                nvtx_range_push("_MetisGroupedLinear.backward.wgrad_gemm")
                if ctx.metis_context.quantization_strategy == QuantizationStrategy.SEPARATE_RESIDUAL:
                    wgrad_list = MetisSvdFunction.grouped_gemm_with_weight_grad_separate_residual(
                        ug_sg_list,
                        vg_list,
                        res_list,
                        output_grad_ug_sg_list,
                        output_grad_vg_list,
                        output_grad_res_list,
                        ctx.activation_dtype,
                        ctx.grad_output_quantizers,
                        list(ctx.m_splits),
                        input_restoreinfo_list=ctx.restore_info_list,
                        grad_restoreinfo_list=output_grad_restore_info_list,
                        restore_strategy=ctx.metis_context.backward_restore_strategy,
                    )
                elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN:
                    wgrad_list = MetisMeanFunction.compute_weight_gradient_mean_grouped_gemm(
                        mean_quant_results,
                        grad_output_mean_quant_results,
                        ctx.activation_dtype,
                        list(ctx.m_splits),
                        quantizer_list=ctx.grad_weight_quantizers,
                    )
                elif ctx.metis_context.quantization_strategy == QuantizationStrategy.MEAN_DIM0_ONLY:
                    wgrad_list = MetisMeanFunction.compute_weight_gradient_mean_grouped_gemm(
                        mean_dim0_only_quant_results,
                        grad_output_mean_dim0_only_quant_results,
                        ctx.activation_dtype,
                        list(ctx.m_splits),
                        quantizer_list=ctx.grad_weight_quantizers,
                    )

                if ctx.use_bias:
                    for i in range(N):
                        if ctx.m_splits[i] > 0:
                            grad_biases[i] = grad_output_splits[i].sum(dim=0)
                nvtx_range_pop("_MetisGroupedLinear.backward.wgrad_gemm")

                # Handle custom DDP from mcore
                def handle_custom_ddp_from_mcore(weight, wgrad):
                    if ctx.weights_requires_grad:
                        if ctx.fuse_wgrad_accumulation and hasattr(
                            weight, "grad_added_to_main_grad"
                        ):
                            weight.grad_added_to_main_grad = True
                            if getattr(weight, "zero_out_wgrad", False):
                                wgrad = get_dummy_wgrad(
                                    list(weight.main_grad.shape),
                                    weight.dtype,
                                    zero=True,
                                )
                            else:
                                wgrad = get_dummy_wgrad(
                                    list(weight.main_grad.shape),
                                    weight.dtype,
                                )
                        elif ctx.fuse_wgrad_accumulation:
                            wgrad = None
                    else:
                        wgrad = None
                    return wgrad

                wgrad_list = [
                    handle_custom_ddp_from_mcore(w, wg)
                    for w, wg in zip(weights, wgrad_list)
                ]
            else:
                wgrad_list = [None] * N

            if not ctx.use_bias:
                grad_biases = [None] * N

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            None,  # non_tensor_args
            *wgrad_list,
            *grad_biases,
        )
