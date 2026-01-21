# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Optional, Tuple, Union

import torch

import transformer_engine_torch as tex


from ..base import (
    get_dummy_wgrad,
    get_ub,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ...quantization import FP8GlobalStateManager
from ...utils import (
    cast_if_needed,
    requires_grad,
    needs_quantized_gemm,
    assert_dim_for_fp8_exec,
    assert_dim_for_all_gather,
    nvtx_range_pop,
    nvtx_range_push,
    get_nvtx_range_context,
)
from ...distributed import (
    get_distributed_world_size,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
)
from ...cpp_extensions import (
    general_gemm,
)
from ...quantized_tensor import (
    QuantizedTensor,
    QuantizedTensorStorage,
    prepare_for_saving,
    restore_from_saved,
)
from ...tensor.float8_tensor import Float8Quantizer
from ...tensor.utils import is_custom
from ...cpu_offload import (
    is_cpu_offload_enabled,
    start_offload,
    mark_not_offload,
    mark_activation_offload,
)
from .quant import MetisSvdFunction
from .metis_context import LinearLowbitContext

__all__ = ["_MetisLinear"]


class _MetisLinear(torch.autograd.Function):
    """Linear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        inp: torch.Tensor,
        bias: Optional[torch.Tensor],
        non_tensor_args: Tuple,
    ) -> torch.Tensor:
        # print("foward LinearLowbitContext=", LinearLowbitContext())
        # pylint: disable=missing-function-docstring

        (
            is_first_microbatch,
            fp8,
            fp8_calibration,
            wgrad_store,
            input_quantizer,
            weight_quantizer,
            output_quantizer,
            grad_input_quantizer,
            grad_weight_quantizer,
            grad_output_quantizer,
            fuse_wgrad_accumulation,
            cpu_offloading,
            tp_group,
            tp_size,
            sequence_parallel,
            tensor_parallel,
            activation_dtype,
            parallel_mode,
            is_grad_enabled,
            ub_overlap_rs_fprop,
            ub_overlap_ag_dgrad,
            ub_overlap_ag_fprop,
            ub_overlap_rs_dgrad,
            ub_bulk_dgrad,
            ub_bulk_wgrad,
            ub_name,
            fp8_output,  # pylint: disable=unused-variable
            fsdp_group,
            module,
            skip_fp8_weight_update,
            symmetric_ar_type,
            save_original_input,
            debug,
            enable_metis,
            svd_grad_output_history,
        ) = non_tensor_args

        current_forward_use_metis = (
            enable_metis and LinearLowbitContext.use_metis and LinearLowbitContext.enable_activation_svd
        )
        # print("forward _Linear current_forward_use_metis=", current_forward_use_metis)
        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.forward"
        if ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ub_name}"

        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        assert inp.shape[-1] == in_features, "GEMM not possible"

        # Configure tensor-parallel communication
        tp_world_size = get_distributed_world_size(tp_group)
        backward_needs_input = is_grad_enabled and weight.requires_grad
        with_input_all_gather_nccl = parallel_mode == "column" and sequence_parallel and not ub_overlap_ag_fprop

        # Configure Userbuffers communication (comm+GEMM overlap)
        if debug:  # turn off userbuffers in debug mode
            ub_overlap_rs_fprop = False
            ub_overlap_ag_fprop = False
            ub_overlap_rs_dgrad = False
            ub_bulk_wgrad = False
            ub_bulk_dgrad = False
        ub_obj = None
        ub_type = None
        if ub_overlap_rs_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.RS
        elif ub_overlap_ag_fprop:
            ub_obj = get_ub(ub_name + "_fprop", fp8)
            ub_type = tex.CommOverlapType.AG

        # custom recipe check
        custom = is_custom(input_quantizer) or is_custom(weight_quantizer)

        # ------------------------------------------------------
        # Prepare input tensor
        # Note: Cast to expected dtype and perform tensor-parallel communication
        # ------------------------------------------------------
        nvtx_range_push(f"{nvtx_label}.input_cast_comm")
        inputmat = inp  # Input tensor to save for backward (maybe sharded)
        inputmat_total = None  # Input tensor to pass to GEMM (gathered)
        own_quantized_input = False
        if fp8:
            assert_dim_for_fp8_exec(inputmat, weight)
            assert_dim_for_all_gather(inputmat, with_input_all_gather_nccl, input_quantizer)
            if save_original_input:
                assert not isinstance(
                    input_quantizer, Float8Quantizer
                ), "DelayedScaling recipe is not supported with save_original_input"
        # print(f"use_metis=={use_metis},LinearLowbitContext=", LinearLowbitContext())
        input_ug_sg, input_vg, input_res = None, None, None
        if isinstance(inputmat, QuantizedTensorStorage):
            inputmat.update_usage(rowwise_usage=True)
        elif current_forward_use_metis and fp8:
            # ------------------------------------------------------
            # Forward x SVD
            # ------------------------------------------------------
            # print("foward enable_activation_svd LinearLowbitContext=", LinearLowbitContext())
            input_quantizer.set_usage(rowwise=True, columnwise=True)
            input_ug_sg, input_vg, input_res,input_restore_info = MetisSvdFunction.svd_lowrank_quant_separate_residual(
                inputmat,
                input_quantizer,
                rank=LinearLowbitContext.activation_lowrank_svd,
                niter=LinearLowbitContext.activation_lowrank_niter,
                token_drop_rate=LinearLowbitContext.activation_token_drop_rate,
                broadcast_dim=LinearLowbitContext.activation_broadcast_dim,
                restore_strategy=LinearLowbitContext.activation_restore_strategy,
                load_history=LinearLowbitContext.load_history, # to store noise
                is_backward=False,

            )
            own_quantized_input = True
        if fp8:
            if save_original_input:
                # No need for column-wise data since this
                # tensor will not be cached for backward pass
                input_quantizer.set_usage(columnwise=False)
                own_quantized_input = False
            inputmat = input_quantizer(inputmat)
        else:
            inputmat_total = inputmat

        if is_cpu_offload_enabled():
            start_offload(input_ug_sg, input_vg, input_res)
        nvtx_range_pop(f"{nvtx_label}.input_cast_comm")
        # ------------------------------------------------------
        # Input tensor is ready for GEMM...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Prepare weight tensor
        # ------------------------------------------------------
        weightmat = weight
        # Configure quantizer
        # No need to set the quantizer states if weight is already quantized
        if weight_quantizer is not None and not isinstance(weight, QuantizedTensor):
            columnwise_usage = is_grad_enabled and inp.requires_grad
            if not columnwise_usage:
                columnwise_usage = is_fp8_activation_recompute_enabled() and not in_fp8_activation_recompute_phase()
            weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)
        elif isinstance(weight, QuantizedTensor):
            # If weight is already quantized, no need to set quantizer states
            weight_quantizer = weight._quantizer
        # Get quantized weight
        update_workspace = is_first_microbatch is None or is_first_microbatch
        weightmat = module.get_weight_workspace(
            tensor=weight,
            quantizer=weight_quantizer,
            cache_name=(None if is_first_microbatch is None else "weight"),
            update_workspace=update_workspace,
            skip_update_flag=skip_fp8_weight_update,
            fsdp_group=fsdp_group,
            workspace_dtype=activation_dtype,
        )
        weightmat.update_usage(rowwise_usage=True)
        # ------------------------------------------------------
        # Weight tensor is ready for GEMM...
        # ------------------------------------------------------

        # Cast bias to expected dtype
        bias_dtype = activation_dtype
        if needs_quantized_gemm(inputmat_total) and activation_dtype == torch.float32:
            # cuBLAS does not support FP8 GEMM with FP32 bias, so we cast to BF16
            bias_dtype = torch.bfloat16
        bias = cast_if_needed(bias, bias_dtype) if bias is not None else bias

        # Calibrate quantizers if needed
        # if not fp8 and fp8_calibration:
        #     if input_quantizer is not None:
        #         input_quantizer.calibrate(inputmat_total)
        #     if weight_quantizer is not None:
        #         weight_quantizer.calibrate(weight)

        # Choose whether to use GEMM kernel with split accumulator
        use_split_accumulator = _2X_ACC_FPROP
        if fp8:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if hasattr(recipe, "fp8_gemm_fprop"):
                use_split_accumulator = recipe.fp8_gemm_fprop.use_split_accumulator

        # Configure output quantizer
        if output_quantizer is not None:
            output_quantizer.set_usage(rowwise=True, columnwise=False)

        # Output buffer for Userbuffers reduce-scatter
        # reduce_scatter_out = None
        # if ub_overlap_rs_fprop:
        #     out_shape = list(inp.shape)
        #     out_shape[0] //= tp_world_size
        #     out_shape[-1] = out_features
        #     reduce_scatter_out = torch.empty(out_shape, dtype=activation_dtype, device=inp.device)

        # ------------------------------------------------------
        # Forward GEMM
        # Note: y = x * w^T
        # ------------------------------------------------------
        # print(f"forward weightmat dtype= {type(weightmat)}, inputmat_total dtype= {type(inputmat_total)}")
        nvtx_range_push(f"{nvtx_label}.gemm")
        if not current_forward_use_metis:
            gemm_out, *_, reduce_scatter_out = general_gemm(
                weightmat,
                inputmat_total,
                quantization_params=output_quantizer,
                out_dtype=activation_dtype,
                bias=bias,
                use_split_accumulator=use_split_accumulator,
                ub=ub_obj,
                ub_type=ub_type,
                extra_output=reduce_scatter_out,
            )
        else:
            output_shape = list(inp.shape)
            output_shape[-1] = weightmat.shape[0]
            gemm_out, _ = MetisSvdFunction.gemm_with_separate_residual(
                input_ug_sg,
                input_vg,
                input_res,
                weightmat,
                activation_dtype,
                input_quantizer,
                is_grad=False,
                output_shape=output_shape,
                restore_info = input_restore_info,
                restore_strategy=LinearLowbitContext.activation_restore_strategy,
            )
        nvtx_range_pop(f"{nvtx_label}.gemm")
        # ------------------------------------------------------
        # Finished forward GEMM...
        # ------------------------------------------------------

        # Deallocate GEMM input tensor if no longer needed
        # TODO(yuzhongw, tmoon): Figure out why inputmat_total is not automatically
        # deallocated by GC. Manually deallocating is a temporary hack.
        # if with_input_all_gather_nccl:
        #     clear_tensor_data(inputmat_total)
        #     inputmat_total = None

        # ------------------------------------------------------
        # Prepare output tensor
        # Note: Perform tensor-parallel communication
        # ------------------------------------------------------
        # out = None
        # if ub_overlap_rs_fprop:
        #     out = reduce_scatter_out
        # elif parallel_mode == "row" and tp_size > 1:
        #     nvtx_range_push(f"{nvtx_label}.row_parallel_comm")
        #     out = gemm_out
        #     if sequence_parallel:
        #         out, _ = reduce_scatter_along_first_dim(out, tp_group)
        #     elif tensor_parallel:
        #         if symmetric_ar_type is not None:
        #             out, _ = symmetric_all_reduce(out, tp_group, all_reduce_type=symmetric_ar_type)
        #         else:
        #             out, _ = allreduce(out, tp_group)
        #     nvtx_range_pop(f"{nvtx_label}.row_parallel_comm")
        # else:
        out = gemm_out
        # ------------------------------------------------------
        # Output tensor is ready to return...
        # ------------------------------------------------------

        # ------------------------------------------------------
        # Cache state for backward pass
        # ------------------------------------------------------

        if is_grad_enabled:
            if save_original_input:
                inputmat = inp

            ctx.weight_quantizer = weight_quantizer

            ctx.backward_input_needs_gather = weight.requires_grad and parallel_mode == "column" and sequence_parallel

            # Discard unneeded data in input tensor
            if backward_needs_input and own_quantized_input and isinstance(inputmat, QuantizedTensorStorage):
                if ctx.backward_input_needs_gather and weight_quantizer.supports_only_rowwise_all_gather():
                    # All-gather is not supported with FP8 column-wise data
                    inputmat.update_usage(rowwise_usage=True, columnwise_usage=False)
                else:
                    # Discard row-wise data since it is not needed in backward pass
                    inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)

            # Cached input tensor
            # saved_inputmat = None
            # if backward_needs_input:
            #     saved_inputmat = inputmat

            if cpu_offloading and input_ug_sg and input_vg and input_res:
                mark_activation_offload(input_ug_sg, input_vg, input_res)

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: FSDP sharding is not valid for models initialized with primary Fp8 weights
            # nvtx_range_push(f"{nvtx_label}.fsdp_scatter")
            # ctx.fsdp_group = fsdp_group
            # ctx.fsdp_shapes = _fsdp_scatter_tensors(
            #     fsdp_group,
            #     saved_inputmat,
            #     weightmat if fp8 and not isinstance(weight, QuantizedTensorStorage) else None,
            # )
            # nvtx_range_pop(f"{nvtx_label}.fsdp_scatter")

            if cpu_offloading:
                ctx.grad_added_to_main_grad = hasattr(weight, "grad_added_to_main_grad")

                if ctx.grad_added_to_main_grad:
                    # If you are passing torch.nn.Parameter through the Torch hooks, you will
                    # get back torch.Tensor. Torch rips off the Parameter wrapper.
                    # You need to preserve the weight object to have all the attributes user
                    # sets for the weights. Because of this, it is not recommended to offload
                    # weights if weights are externally touched outside this module
                    ctx.weight_object = weight

            mark_not_offload(weight, weightmat, bias)
            # TODO(ksivamani): Check memory usage
            tensors_to_save, tensor_objects = prepare_for_saving(
                input_ug_sg,
                input_vg,
                input_res,
                weightmat,
                weight,
                bias,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects

            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_recipe = FP8GlobalStateManager.get_fp8_recipe() if fp8 else None
            ctx.input_quantizer = input_quantizer
            ctx.grad_input_quantizer = grad_input_quantizer
            ctx.grad_weight_quantizer = grad_weight_quantizer
            ctx.grad_output_quantizer = grad_output_quantizer
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            if fuse_wgrad_accumulation and weight.requires_grad:
                # This check is needed to ensure that main_grad is not created
                # during the forward pass when using MCore FSDP as it creates
                # the main_grad buffer lazily before backprop
                if hasattr(weight, "__fsdp_param__"):
                    # MCore FSDP creates main_grad lazily before backward
                    ctx.main_grad_func = weight.get_main_grad
                else:
                    ctx.main_grad_func = lambda: weight.main_grad

            ctx.debug = debug
            ctx.custom = custom
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = bias is not None
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp.shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.ub_overlap_ag = ub_overlap_ag_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_name = ub_name
            ctx.tp_size = tp_size
            ctx.requires_dgrad = inp.requires_grad
            ctx.requires_wgrad = weight.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            # ctx.owns_input = saved_inputmat is not inp
            if ctx.fp8 and requires_grad(inp, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module
            ctx.wgrad_store = wgrad_store

            # Load metis lowbit context
            ctx.enable_metis = enable_metis
            ctx.metis_context = LinearLowbitContext().clone()
            ctx.svd_grad_output_history = svd_grad_output_history
            ctx.input_restore_info = input_restore_info
        # ------------------------------------------------------
        # Cached state for backward pass is ready...
        # ------------------------------------------------------

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # print("backward LinearLowbitContext=", ctx.metis_context)
        # pylint: disable=missing-function-docstring

        # NVTX label for profiling
        nvtx_label = "transformer_engine._Linear.backward"
        if ctx.ub_name is not None:
            nvtx_label = f"{nvtx_label}.{ctx.ub_name}"

        with get_nvtx_range_context("_Linear_backward"):
            saved_tensors = ctx.saved_tensors
            (
                input_ug_sg,
                input_vg,
                input_res,
                weight_fp8,
                weight,
                bias,
            ) = restore_from_saved(  # pylint: disable=unbalanced-tuple-unpacking
                ctx.tensor_objects, saved_tensors
            )

            # Delete the references to tensor objects once they've been consumed
            # by the `restore_from_saved` method to construct back the actual tensors.
            ctx.tensor_objects = None

            # Since main_grad can be modified inplace, it should not be a part of saved_tensors
            main_grad = (
                ctx.main_grad_func()
                if weight is not None and ctx.fuse_wgrad_accumulation and ctx.requires_wgrad
                else None
            )

            if ctx.cpu_offloading:
                if ctx.grad_added_to_main_grad:
                    weight = ctx.weight_object
            if ctx.requires_wgrad and ctx.fuse_wgrad_accumulation:
                weight.main_grad = main_grad

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            # nvtx_range_push(f"{nvtx_label}.fsdp_gather")
            # _fsdp_gather_tensors(
            #     ctx.fsdp_group,
            #     ctx.fsdp_shapes,
            #     inputmat,
            #     weight_fp8,
            # )
            # nvtx_range_pop(f"{nvtx_label}.fsdp_gather")

            # Configure Userbuffers communication (comm+GEMM overlap)
            # ctx.ub_obj_gradout = None
            # ub_obj_dgrad = None
            # ub_obj_wgrad = None
            # ub_type_dgrad = None
            # ub_type_wgrad = None
            # dgrad_shape = [reduce(multiply_op, ctx.inp_shape[:-1]), ctx.inp_shape[-1]]
            # if ctx.ub_overlap_ag:
            #     # Overlap grad_output all-gather with dgrad compute
            #     ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
            #     ub_obj_dgrad = ctx.ub_obj_gradout
            #     ub_type_dgrad = tex.CommOverlapType.AG
            # elif ctx.ub_overlap_rs_dgrad:
            #     # Overlap dgrad reduce-scatter with dgrad compute
            #     ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
            #     ub_obj_dgrad = ctx.ub_obj_gradout
            #     ub_type_dgrad = tex.CommOverlapType.RS
            # else:
            #     if ctx.ub_bulk_dgrad:
            #         # Overlap inputmat all-gather with dgrad compute
            #         ctx.ub_obj_gradout = get_ub(ctx.ub_name + "_dgrad", ctx.fp8)
            #         ub_obj_dgrad = ctx.ub_obj_gradout
            #         ub_type_dgrad = tex.CommOverlapType.AG
            #     if ctx.ub_bulk_wgrad:
            #         # Overlap dgrad reduce-scatter with wgrad compute
            #         ub_obj_wgrad = get_ub(ctx.ub_name + "_wgrad", ctx.fp8)
            #         ub_type_wgrad = tex.CommOverlapType.RS

            # --------------------------------------------------
            # Prepare grad output tensor
            # Note: Cast to expected dtype and perform tensor-parallel communication
            # --------------------------------------------------

            # Unmodified grad output tensor
            grad_output_arg = grad_output

            # Configure quantizer for grad output tensor
            # Note: dgrad GEMM requires row-wise usage, wgrad GEMM
            # requires column-wise usage
            if ctx.grad_output_quantizer is not None:
                quantizer = ctx.grad_output_quantizer
                quantizer.set_usage(rowwise=True, columnwise=True)
                if ctx.ub_overlap_ag:
                    # Userbuffers only supports communication for one
                    # tensor usage at a time. Configure quantizer with
                    # usage for only dgrad GEMM.
                    quantizer.set_usage(columnwise=False)

            # Adjust the quantization direction approach depending
            # on whether wgrad calculations will be performed.
            # NOTE: If requires_dgrad is False, disabling `rowwise` quantization and keeping `columnwise` quantization
            #       results in `Assertion failed: output_tensor->has_data(). Quantizing in only the columnwise direction not supported yet!`
            # NOTE: For `ctx.bias is True`, selected quantize kernel errors with
            #       `cast_kernels.cuh:1322 in function fp8_quantize_arch_l_100: Not implemented scaling mode or fusion: NVTE_DELAYED_TENSOR_SCALING or IS_DBIAS=true on GPU with compute capability < 10.0.`
            if not ctx.use_bias and not ctx.requires_wgrad and ctx.grad_output_quantizer is not None:
                ctx.grad_output_quantizer.set_usage(columnwise=False)
            # print("backward  ctx.use_metis=", ctx.use_metis)
            # Prepare grad output tensor
            nvtx_range_push(f"{nvtx_label}.grad_output_preprocess")
            # (
            #     grad_output,
            #     grad_bias,
            # ) = TransformerEngineBaseModule.grad_output_preprocess(
            #     ctx,
            #     grad_output,
            #     ctx.parallel_mode == "row",
            #     ctx.grad_output_quantizer,
            # )
            if ctx.enable_metis and ctx.metis_context.use_metis and ctx.metis_context.enable_backward_svd:
                if ctx.metis_context.backward_lowrank_svd > 0:
                    grad_output_shape = grad_output.shape
                    output_grad_ug_sg, output_grad_vg, output_grad_res,output_grad_restore_info = (
                        MetisSvdFunction.svd_lowrank_quant_separate_residual(
                            grad_output,
                            input_quantizer=ctx.grad_output_quantizer,
                            rank=ctx.metis_context.backward_lowrank_svd,
                            niter=ctx.metis_context.backward_lowrank_niter,
                            token_drop_rate=ctx.metis_context.backward_token_drop_rate,
                            broadcast_dim=ctx.metis_context.backward_broadcast_dim,
                            enable_gradient_accumulation_optimization=ctx.metis_context.enable_gradient_accumulation_optimization,
                            use_grad_power_iteration_svd=ctx.metis_context.use_grad_power_iteration_svd,
                            load_history = ctx.metis_context.load_history,
                            history_list = ctx.svd_grad_output_history,
                            is_backward=True,

                        )
                    )
            nvtx_range_pop(f"{nvtx_label}.grad_output_preprocess")

            # --------------------------------------------------
            # Grad output tensor is ready for computing grad input...
            # --------------------------------------------------

            # --------------------------------------------------
            # Prepare input tensor
            # Note: Input tensor is needed for wgrad GEMM.
            # Tensor-parallel communication is overlapped with dgrad
            # GEMM.
            # --------------------------------------------------
            inputmat_total = None
            inputmat_total_work = None
            # if ctx.requires_wgrad:
            #     assert isinstance(input_ug_sg, QuantizedTensorStorage)
            # --------------------------------------------------
            # Input tensor is ready for computing grad weight...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad input tensor
            # --------------------------------------------------

            dgrad = None
            dgrad_work = None
            if ctx.requires_dgrad:

                # Make sure required data is available
                if isinstance(grad_output, QuantizedTensorStorage):
                    grad_output.update_usage(rowwise_usage=True)
                if ctx.weight_quantizer is not None and isinstance(weight_fp8, QuantizedTensorStorage):
                    weight_fp8.update_usage(columnwise_usage=True)

                # Choose whether to use GEMM kernel with split accumulator
                use_split_accumulator = _2X_ACC_DGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_dgrad"):
                        use_split_accumulator = recipe.fp8_gemm_dgrad.use_split_accumulator

                # Update grad input quantizer
                if ctx.grad_input_quantizer is not None:
                    ctx.grad_input_quantizer.set_usage(rowwise=True, columnwise=False)

                # # Output buffers for Userbuffers reduce-scatter
                gemm_out = None
                # reduce_scatter_out = None
                # if ctx.ub_overlap_rs_dgrad:
                #     reduce_scatter_out = torch.empty(
                #         dgrad_shape, dtype=ctx.activation_dtype, device=grad_output_arg.device
                #     )
                # elif ctx.ub_bulk_wgrad:
                #     gemm_out = ub_obj_wgrad.get_buffer(local_chunk=False)

                # dgrad GEMM
                # Note: dx = dy * w
                # print(f"backward weight_fp8 dtype= {type(weight_fp8)}, grad_output dtype= {type(grad_output)}")
                nvtx_range_push(f"{nvtx_label}.dgrad_gemm")
                if ctx.enable_metis and ctx.metis_context.use_metis and ctx.metis_context.enable_backward_svd:
                    gemm_out, _ = MetisSvdFunction.gemm_with_separate_residual(
                        output_grad_ug_sg,
                        output_grad_vg,
                        output_grad_res,
                        weight_fp8,
                        ctx.activation_dtype,
                        ctx.grad_output_quantizer,
                        True,
                        input_res.size(),
                        restore_info=output_grad_restore_info,
                        restore_strategy=ctx.metis_context.backward_restore_strategy
                    )
                else:
                    gemm_out, *_, reduce_scatter_out = general_gemm(
                        weight_fp8,
                        grad_output,
                        layout="NN",
                        grad=True,
                        quantization_params=ctx.grad_input_quantizer,
                        out=gemm_out,
                        out_dtype=ctx.activation_dtype,
                        use_split_accumulator=use_split_accumulator,
                        bulk_overlap=ctx.ub_bulk_dgrad,
                    )
                nvtx_range_pop(f"{nvtx_label}.dgrad_gemm")

                # Prepare grad input tensor
                # Note: Perform tensor-parallel communication
                # if ctx.ub_overlap_rs_dgrad:
                #     dgrad = reduce_scatter_out
                # elif ctx.ub_bulk_wgrad:
                #     dgrad = ub_obj_wgrad.get_buffer(local_chunk=True)
                # elif ctx.parallel_mode == "column" and ctx.tp_size > 1:
                #     nvtx_range_push(f"{nvtx_label}.column_parallel_comm_dgrad")
                #     dgrad = gemm_out
                #     if ctx.sequence_parallel:
                #         dgrad, dgrad_work = reduce_scatter_along_first_dim(
                #             dgrad,
                #             ctx.tp_group,
                #             async_op=True,
                #         )
                #     else:
                #         dgrad, dgrad_work = allreduce(dgrad, ctx.tp_group, async_op=True)
                #     nvtx_range_pop(f"{nvtx_label}.column_parallel_comm_dgrad")
                # else:
                dgrad = gemm_out

            # --------------------------------------------------
            # Grad input tensor has been computed...
            # --------------------------------------------------

            # --------------------------------------------------
            # Compute grad weight
            # --------------------------------------------------

            wgrad = None
            if ctx.requires_wgrad:

                # Prepare input tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                if inputmat_total_work is not None:
                    inputmat_total_work.wait()
                    inputmat_total_work = None
                # if ctx.fp8 or ctx.debug:
                #     assert isinstance(inputmat_total, QuantizedTensorStorage)
                    # if isinstance(inputmat_total, QuantizedTensorStorage):
                    #     inputmat_total.update_usage(columnwise_usage=True)
                    # else:
                    #     ctx.input_quantizer.set_usage(rowwise=False, columnwise=True)
                    #     inputmat_total = ctx.input_quantizer(inputmat_total)

                # Prepare grad output tensor
                # Note: Synchronize tensor-parallel communication and
                # make sure required data is available
                # if ctx.ub_overlap_ag and isinstance(ctx.grad_output_quantizer, MXFP8Quantizer):
                # UB does not support pipelined overlapping grad output
                # all-gather with wgrad GEMM. Also, we can't
                # convert row-scaled MXFP8 to column-scaled, so we
                # can't reuse the grad output that was gathered
                # for the dgrad GEMM. We work around by explicitly
                # overlapping the AG operation with the dgrad GEMM.

                # Get the communication stream from the dgrad GEMM to use for the AG
                # dgrad_send_stream, dgrad_recv_stream = ub_obj_dgrad.get_communication_stream()

                # # This object is separate from the ub_obj_wgrad object which is passed to the GEMM
                # ub_obj_overlap_wgrad = get_ub(ctx.ub_name + "_wgrad", ctx.fp8)

                # ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)

                # # We use the send stream to copy into the userbuffers.
                # # This is the same stream that we will use to access the data in the AG,
                # # so we dont need to add any syncs yet.
                # with torch.cuda.stream(dgrad_send_stream):
                #     grad_output, _ = fill_userbuffers_buffer_for_all_gather(
                #         ub_obj_overlap_wgrad,
                #         grad_output_arg,
                #         ctx.grad_output_quantizer,
                #         ctx.tp_group,
                #     )

                # Allgather grad_outputs[0] using the dgrad streams so we can overlap with the fc2_dgrad gemm
                # tex.bulk_overlap_ag_with_external_gemm(
                #     ub_obj_overlap_wgrad, dgrad_send_stream, dgrad_recv_stream
                # )

                # if ctx.fp8 or ctx.debug:
                #     assert isinstance(grad_output, QuantizedTensorStorage)
                    # if isinstance(grad_output, QuantizedTensorStorage):
                    #     grad_output.update_usage(columnwise_usage=True)
                    # else:
                    #     ctx.grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                    #     grad_output = ctx.grad_output_quantizer(grad_output)

                # Figure out whether to use split accumulator
                use_split_accumulator = _2X_ACC_WGRAD
                if ctx.fp8:
                    recipe = ctx.fp8_recipe
                    if hasattr(recipe, "fp8_gemm_wgrad"):
                        use_split_accumulator = recipe.fp8_gemm_wgrad.use_split_accumulator

                # Figure out whether to output wgrad GEMM directly into main grad
                if ctx.is_first_microbatch is not None:
                    accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                else:
                    accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

                # Output buffer for overlapping FP8 grad input
                # reduce-scatter with wgrad GEMM
                # reduce_scatter_out = None
                # if ctx.ub_bulk_wgrad and ub_obj_wgrad.is_fp8_ubuf():
                #     reduce_scatter_out = torch.empty(
                #         dgrad_shape, dtype=ctx.activation_dtype, device=grad_output_arg.device
                #     )

                # Arguments to include in wgrad GEMM closure
                # wgrad_gemm_kwargs = {
                #     "out_dtype": (
                #         main_grad.dtype if ctx.fuse_wgrad_accumulation else ctx.activation_dtype
                #     ),
                #     "quantization_params": ctx.grad_weight_quantizer,
                #     "accumulate": (
                #         accumulate_wgrad_into_param_main_grad
                #         if not getattr(weight, "overwrite_main_grad", False)
                #         else False
                #     ),
                #     "layout": "NT",
                #     "out": main_grad if ctx.fuse_wgrad_accumulation else None,
                #     "bias": (bias if (grad_bias is None and not ctx.fp8) else None),
                #     "use_split_accumulator": use_split_accumulator,
                #     "grad": True,
                #     "ub": ub_obj_wgrad,
                #     "ub_type": ub_type_wgrad,
                #     "extra_output": reduce_scatter_out,
                #     "bulk_overlap": ctx.ub_bulk_wgrad,
                # }

                # def wgrad_gemm(
                #     x: torch.Tensor,
                #     dy: torch.Tensor,
                # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                #     """Perform wgrad GEMM: dw = dy^T * x

                #     May be fused with bgrad computation.

                #     May be called outside of this function to enable
                #     some advanced communication/compute overlapping.

                #     """
                #     nvtx_range_push(f"{nvtx_label}.wgrad_gemm")
                #     dw, db, *_ = general_gemm(x, dy, **wgrad_gemm_kwargs)
                #     nvtx_range_pop(f"{nvtx_label}.wgrad_gemm")
                #     return dw, db

                # Choose whether to call wgrad GEMM now or delay
                # if ctx.wgrad_store is not None and ctx.wgrad_store.delay_wgrad_compute():
                #     if (
                #         wgrad_gemm_kwargs["ub"] is not None
                #         or wgrad_gemm_kwargs["ub_type"] is not None
                #         or wgrad_gemm_kwargs["extra_output"] is not None
                #         or wgrad_gemm_kwargs["bulk_overlap"]
                #     ):
                #         raise NotImplementedError(
                #             "Delayed weight grad computation is not supported "
                #             "with Userbuffers (tensor-parallel communication overlapping)"
                #         )
                #     ctx.wgrad_store.put([inputmat_total, grad_output], wgrad_gemm)
                # else:

                #     # Call wgrad GEMM now
                #     wgrad, grad_bias_ = wgrad_gemm(inputmat_total, grad_output)

                #     # Update grad bias if needed
                #     if grad_bias is None:
                #         grad_bias = grad_bias_
                #     del grad_bias_

                #     # Deallocate tensors if permitted
                #     if ctx.owns_input:
                #         # Input tensor is internal
                #         clear_tensor_data(inputmat_total)
                #     elif ctx.backward_input_needs_gather:
                #         # Gathered input tensor is internal
                #         clear_tensor_data(inputmat_total)
                #     if ctx.parallel_mode == "row" and ctx.sequence_parallel:
                #         # Gathered grad output tensor is internal
                #         clear_tensor_data(grad_output)

                # # Update grad input if overlapping reduce-scatter with wgrad GEMM
                # if ctx.ub_bulk_wgrad:
                #     if ub_obj_wgrad.is_fp8_ubuf():
                #         dgrad = reduce_scatter_out
                #     else:
                #         dgrad = ub_obj_wgrad.get_buffer(local_chunk=True).clone()
                if ctx.enable_metis and ctx.metis_context.use_metis and ctx.metis_context.enable_backward_svd:
                    wgrad = MetisSvdFunction.gemm_with_weight_grad_separate_residual(
                        input_ug_sg,
                        input_vg,
                        input_res,
                        output_grad_ug_sg,
                        output_grad_vg,
                        output_grad_res,
                        ctx.activation_dtype,
                        ctx.grad_output_quantizer,
                        False,
                        False,
                        input_restoreinfo=ctx.input_restore_info,
                        grad_restoreinfo=output_grad_restore_info,
                        restore_strategy=ctx.metis_context.backward_restore_strategy,
                    )
                    # wgrad = ctx.grad_weight_quantizer(wgrad)
            # --------------------------------------------------
            # Grad weight has been computed...
            # --------------------------------------------------

            # Don't return grad bias if not needed
            if not ctx.use_bias:
                grad_bias = None

            # Make sure all tensor-parallel communication is finished
            if inputmat_total_work is not None:
                inputmat_total_work.wait()
                inputmat_total_work = None
            if dgrad_work is not None:
                dgrad_work.wait()
                dgrad_work = None

        if ctx.requires_wgrad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and weight is not None and hasattr(weight, "grad_added_to_main_grad"):
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

        # Update FP8 scaling factors if needed
        # if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
        #     nvtx_range_push(f"{nvtx_label}.reduce_and_update_fp8_tensors")
        #     FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)
        #     nvtx_range_pop(f"{nvtx_label}.reduce_and_update_fp8_tensors")

        # Scatter fp8 weight buffers
        # if ctx.fp8 and not isinstance(weight, QuantizedTensorStorage):
        #     _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)
        return (
            wgrad,
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            grad_bias,
            None,
        )
