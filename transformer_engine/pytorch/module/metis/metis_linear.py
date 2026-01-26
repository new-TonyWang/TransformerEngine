# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear API"""
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
from functools import reduce, partial
from operator import mul as multiply_op
from enum import Enum
import warnings

import torch
import torch.nn as nn

from transformer_engine.pytorch.distributed import dist_group_type
from transformer_engine.pytorch.transformer import TransformerEngineBaseModule
import transformer_engine_torch as tex
import debugpy

# from ....debug.pytorch.debug_state import TEDebugState
from .metis_context import LinearLowbitContext

__all__ = ["MetisLinear"]

@torch.no_grad()
def init_tensor_with_data(source_tensor: torch.Tensor, weight_tensor: torch.Tensor):
    weight_tensor.copy_(source_tensor.detach())

class ZeroLayer(nn.Module):
    def forward(self, x, **kwargs):
        return torch.zeros_like(x)

def _linear_custom_repr(module) -> str:
    """
    专门用于格式化包含 enable_metis 和 params_dtype 的 Linear 层
    """
    # 1. 获取基础维度信息
    in_features = getattr(module, "in_features", "N/A")
    out_features = getattr(module, "out_features", "N/A")

    # 3. 获取特定属性
    enable_metis = getattr(module, "enable_metis", "N/A")

    # 4. 格式化输出
    return (
        f"{module.__class__.__name__}("
        f"in_features={in_features}, "
        f"out_features={out_features}, "
        f"bias={module.use_bias}, "
        f"enable_metis={enable_metis}, "
        f"device={module.weight.device},"
        f")"
    )

class DecomposedLinear(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 *args, 
                bias: bool = False,
                te_linear_args = {},
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.params_dtype = torch.get_default_dtype() if te_linear_args.get("params_dtype") is None else te_linear_args.get("params_dtype")
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.device = te_linear_args.get("device","cpu")
        self.forward_svd_rank = LinearLowbitContext.forward_svd_rank
        self.enable_lowbit = LinearLowbitContext.enable_lowbit
        self.weight_svd_has_initialized = False

        self.te_linear_args = te_linear_args
        self.vlinear = nn.Identity()
        self.ulinear = nn.Identity()
        # self.s = None
        self.linear_residual = ZeroLayer()

        self.decompose_weight()

    def decompose_weight(self):
        from transformer_engine.pytorch.module.linear import Linear  # avoid circular import
        rank = self.forward_svd_rank
        s_tensor = torch.empty(
            rank if rank > 0 else self.out_features,
            device=self.device,
            dtype=self.params_dtype,
        )
        if rank > 0:
            # only low rank need linear_residual
            self.linear_residual = Linear(
                self.in_features,
                self.out_features,
                enable_metis = True,
                bias=self.use_bias,
                **self.te_linear_args,
            )

            # decompose weight with low rank
            v_weight_shape = (self.in_features, rank)
            u_weight_shape = (rank, self.out_features)
        else:
            # decompose weight with full rank
            v_weight_shape = (self.in_features,self.out_features)
            u_weight_shape = (self.out_features, self.out_features)

        self.vlinear = Linear(
            v_weight_shape[0],
            v_weight_shape[1],
            bias=False,
            enable_metis=self.enable_lowbit,
            **self.te_linear_args,
        )
        self.ulinear = Linear(
            u_weight_shape[0],
            u_weight_shape[1],
            bias=False,
            **self.te_linear_args,
        )
        self.register_parameter(
            "s",
            torch.nn.Parameter(s_tensor),
        )

    def initialize_weight_svd_decomposition_from_tensor(self, weight_param: torch.Tensor, bias_param: Optional[torch.Tensor]):

        weight = weight_param.detach()
        device = self.device
        u,s,v = self._svd_decompose_weight_fp32(weight.float())
        if bias_param is not None:
            bias = bias_param.detach()
            bias = bias.to(device=device)
        else:
            bias = None
        w = weight.to(device=device)
        # forward svd low rank
        rank = self.forward_svd_rank
        if rank > 0:

            linear_residual_weight = (
                w - u[:, :rank] 
                @ torch.diag(s[:rank]) 
                @ v[:rank]
                )
            # only low rank need linear_residual
            self.linear_residual.weight.data.copy_(linear_residual_weight)
            # device=device
            if bias is not None:
                self.linear_residual.bias.data.copy_(bias)

            # decompose weight with low rank
            v_weight_data = v[: rank, :]
            u_weight_data = u[:, : rank]
            s_data = s[: rank]
        else:
            # decompose weight with full rank
            v_weight_data = v
            u_weight_data = u

        self.ulinear.weight.data.copy_(u_weight_data)
        self.vlinear.weight.data.copy_(v_weight_data)
        self.s.data.copy_(s_data)

        self.weight_svd_has_initialized = True

    def _svd_decompose_weight_fp32(self, weight_fp32:torch.Tensor):
        u, s, v = torch.linalg.svd(weight_fp32, full_matrices=False)
        u = u.to(
            device=self.device, dtype=self.params_dtype
        )
        s = s.to(
            device=self.device, dtype=self.params_dtype
        )
        v = v.to(
            device=self.device, dtype=self.params_dtype
        )
        return u,s,v

    @torch.no_grad()
    def update_weight_svd_decomposition(self):
        assert self.weight_svd_has_initialized
        print("updating weight svd decomposition ")
        weight_fp32 = (self.ulinear.weight @ torch.diag(self.s) @ self.vlinear.weight).float()
        u,s,v = self._svd_decompose_weight_fp32(weight_fp32)
        rank = self.forward_svd_rank
        if rank > 0:
            # no need to update linear_residual
            v_weight_data = v[: rank, :]
            u_weight_data = u[:, : rank]
            s_data = s[: rank]
        else:
            # decompose weight with full rank
            v_weight_data = v
            u_weight_data = u
            s_data = s

        self.ulinear.weight.data.copy_(u_weight_data)
        self.vlinear.weight.copy_(v_weight_data)
        self.s.data.copy_(s_data)

    def forward(self, inp: torch.Tensor, **kvargs) -> torch.Tensor:
        # TODO optimize vlinear inp quant and linear_residual inp quant, they share the same input, or merge them into one linear
        y = self.vlinear(inp, **kvargs)
        y = torch.mul(self.s, y)
        y = self.ulinear(y, **kvargs)
        y_0 = self.linear_residual(inp, **kvargs)
        y = y + y_0
        return y

    def __repr__(self):
            """
            自定义打印格式，整合了主类配置与子模块的详细状态
            """
            # 1. 主类配置信息
            main_str = (
                f"{self._get_name()}("
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.use_bias}, "
                f"rank={self.forward_svd_rank}, "
                f"lowbit={self.enable_lowbit}, "
                f"initialized={self.weight_svd_has_initialized}, "
                f"device={self.device},"
            )

            # 3. 如果已初始化，详细打印 SVD 组件
            # 使用 _linear_custom_repr 格式化 Linear 层
            child_lines = []
            
            # 2. 如果未初始化，进行提示
            if not self.weight_svd_has_initialized:
                child_lines.append(f"{main_str}\n  (status): Weights not decomposed.")

            # vlinear
            child_lines.append(f"(vlinear): {_linear_custom_repr(self.vlinear)}")
            
            # ulinear
            child_lines.append(f"(ulinear): {_linear_custom_repr(self.ulinear)}")
            
            # s (Singular Values)
            # s 是 Parameter，直接打印数据太长，我们打印形状和类型
            if isinstance(self.s, torch.nn.Parameter):
                s_info = f"Parameter(shape={tuple(self.s.shape)}, dtype={self.s.dtype})"
            else:
                s_info = str(self.s) # Fallback if it's Identity
            child_lines.append(f"(s): {s_info}")

            # linear_residual (仅在 rank > 0 时存在有效值)
            if self.forward_svd_rank > 0:
                child_lines.append(f"(linear_residual): {_linear_custom_repr(self.linear_residual)}")
            else:
                # 如果是全秩分解，residual 通常是 ZeroLayer 或不使用
                child_lines.append(f"(linear_residual): {self.linear_residual}")

            # 4. 拼接最终字符串 (模拟 PyTorch 的缩进格式)
            main_str += "\n  " + "\n  ".join(child_lines) + "\n)"
            
            return main_str

class MetisLinearState(Enum):
    STATE_ORIGINAL_LINEAR = "STATE_ORIGINAL_LINEAR"
    STATE_WEIGHT_DECOMPOSED_LINEAR = "STATE_WEIGHT_DECOMPOSED_LINEAR"


class MetisLinear(torch.nn.Module):
    """Applies a linear transformation to the incoming data :math:`y = xA^T + b`

    On NVIDIA GPUs it is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in. This argument along with
                             weight tensor having attribute 'overwrite_main_grad' set to True
                             will overwrite `main_grad` instead of accumulating.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    delay_wgrad_compute : bool, default = `False`
                         Whether or not to delay weight gradient computation. If set to `True`,
                         it's the user's responsibility to call `module.backward_dw` to compute
                         weight gradients.
    symmetric_ar_type : {None, 'multimem_all_reduce', 'two_shot', 'one_shot'}, default = None
                   Type of symmetric memory all-reduce to use during the forward pass.
                   This can help in latency bound communication situations.
                   Requires PyTorch version 2.7.0 or higher. When set to None, standard all-reduce
                   is used.
    save_original_input : bool, default = `False`
                       If set to `True`, always saves the original input tensor rather than the
                       cast tensor. In some scenarios, the input tensor is used by multiple modules,
                       and saving the original input tensor may reduce the memory usage.
                       Cannot work with FP8 DelayedScaling recipe.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_ag: bool = False,
        ub_overlap_rs: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_bulk_wgrad: bool = False,
        ub_name: Optional[str] = None,
        delay_wgrad_compute: bool = False,
        symmetric_ar_type: Optional[str] = None,
        save_original_input: bool = False,
        name: Optional[str] = None,
    ) -> None:
        # print("current LinearLowbitContext=", LinearLowbitContext())
        from transformer_engine.pytorch.module.linear import Linear  # avoid circular import

        super().__init__()
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name
        self.symmetric_ar_type = symmetric_ar_type
        self.save_original_input = save_original_input
        self.name = name
        self.device = device
        self.commonMetisSvdFunction_args = {
            "return_bias": return_bias,
            "get_rng_state_tracker": get_rng_state_tracker,
            "rng_tracker_name": rng_tracker_name,
            "parameters_split": parameters_split,
            "device": device,
            "params_dtype": params_dtype,
            "sequence_parallel": sequence_parallel,
            "tp_group": tp_group,
            "tp_size": tp_size,
            "parallel_mode": parallel_mode,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "ub_overlap_ag": ub_overlap_ag,
            "ub_overlap_rs": ub_overlap_rs,
            "ub_overlap_rs_dgrad": ub_overlap_rs_dgrad,
            "ub_bulk_dgrad": ub_bulk_dgrad,
            "ub_bulk_wgrad": ub_bulk_wgrad,
            "ub_name": ub_name,
            "delay_wgrad_compute": delay_wgrad_compute,
            "symmetric_ar_type": symmetric_ar_type,
            "save_original_input": save_original_input,
            "name": name,
        }

        # print("Metis linear==",LinearLowbitContext())
        self.original_linear = Linear(
            in_features,
            out_features,
            bias=bias,
            enable_metis=LinearLowbitContext.enable_lowbit,
            init_method=init_method,
            **self.commonMetisSvdFunction_args,
        )

        self.current_state = MetisLinearState.STATE_ORIGINAL_LINEAR

        self.weight_svd_decomposition_model = nn.Identity()
        self.enable_weight_svd = LinearLowbitContext.enable_weight_svd

        self.current_forward_model = self.original_linear

        if self.enable_weight_svd:
            #  using svd to decompose linear_residual later
            self.weight_svd_decomposition_model = DecomposedLinear(in_features,out_features,te_linear_args = self.commonMetisSvdFunction_args)

        # debugpy.breakpoint()
        if LinearLowbitContext.enable_weight_svd:
            if LinearLowbitContext.forward_svd_warmup_steps <= 0:
            # no need to warmup original_linear, directly decomposition linear_residual
                self.weight_svd_decomposition()
            else:
            # need to warmup original_linear, off load to cpu when warming up original_linear
                self.weight_svd_decomposition_model.to("cpu")

    @staticmethod
    @torch.no_grad()
    def init_tensor_with_data(source_tensor: torch.Tensor, weight_tensor: torch.Tensor):
        weight_tensor.copy_(source_tensor.detach())

    def weight_svd_decomposition(self):
        from transformer_engine.pytorch.module.linear import Linear  # avoid circular import
        assert self.enable_weight_svd
        if self.current_state == MetisLinearState.STATE_ORIGINAL_LINEAR:
            assert isinstance(self.weight_svd_decomposition_model, DecomposedLinear)
            print("init_weight_svd_decomposition")
            assert isinstance(self.current_forward_model, Linear)
            self.weight_svd_decomposition_model.to(self.device) # load to GPU 
            self.weight_svd_decomposition_model.initialize_weight_svd_decomposition_from_tensor(
                weight_param = self.original_linear.weight.detach(),
                bias_param =  self.original_linear.weight.bias.detach() if self.use_bias else None
            )
            self.original_linear.to("cpu") # offload old linear
            self.current_forward_model = self.weight_svd_decomposition_model
            self.current_state = MetisLinearState.STATE_WEIGHT_DECOMPOSED_LINEAR

        elif self.current_state == MetisLinearState.STATE_WEIGHT_DECOMPOSED_LINEAR:
            print("update_weight_svd_decomposition")
            self.weight_svd_decomposition_model.to(self.device) # ensure model has loaded to GPU 
            self.weight_svd_decomposition_model.update_weight_svd_decomposition()
        
    def forward(self, inp: torch.Tensor, **kvargs) -> torch.Tensor:
        return self.current_forward_model(inp,**kvargs)

    def __repr__(self):
            """
            Custom repr to display MetisLinear internal state and active path.
            """
            # 1. 顶部属性列表
            info_lines = [
                f"in_features={self.in_features},",
                f"out_features={self.out_features},",
                f"bias={self.use_bias},",
                f"enable_weight_svd={self.enable_weight_svd},",
                f"state={self.current_state}",
                f"name={self.name}"
            ]
            
            # 2. 格式化子模块
            # (original_linear): 使用自定义的 _linear_custom_repr
            orig_linear_str = _linear_custom_repr(self.original_linear)
            
            # (weight_svd_decomposition_model): 获取 DecomposedLinear 的 repr 并处理缩进
            svd_model_str = repr(self.weight_svd_decomposition_model)
            svd_lines = svd_model_str.split('\n')
            # 第一行保持原样，后续行增加两个空格的缩进
            svd_model_indented = svd_lines[0]
            if len(svd_lines) > 1:
                svd_model_indented += '\n' + '\n'.join(['  ' + line for line in svd_lines[1:]])

            # 3. 确定当前 forward 指针的文本描述
            if self.current_forward_model is self.original_linear:
                current_ptr_str = "original_linear"
            elif self.current_forward_model is self.weight_svd_decomposition_model:
                current_ptr_str = "weight_svd_decomposition_model"
            else:
                current_ptr_str = str(self.current_forward_model)

            # 4. 组装最终字符串
            # 格式:
            # MetisLinear(
            #   in_features=...,
            #   ...
            #   (original_linear): ...
            #   (weight_svd_decomposition_model): ...
            #   (current_forward_model) = ...
            # )
            
            main_str = f"{self._get_name()}(\n"
            
            # 添加属性
            for line in info_lines:
                main_str += f"  {line}\n"
                
            # 添加子模块和状态
            main_str += f"  (original_linear): {orig_linear_str}\n"
            main_str += f"  (weight_svd_decomposition_model): {svd_model_indented}\n"
            main_str += f"  (current_forward_model) = {current_ptr_str}\n"
            
            main_str += ")"
            
            return main_str

    def fetch_weight_gradients(self) -> Dict[str, Optional[torch.Tensor]]:
        """
        根据当前模型的运行状态(Linear 或 DecomposedLinear)，
        动态获取对应的权重梯度。
        
        Returns:
            Dict[str, Tensor]: 包含梯度名称和对应 Tensor 的字典。
                            如果在第一次 backward 之前调用，Tensor 可能为 None。
        """
        grads = {}

        # 1. 如果当前是原始 Linear 状态
        if self.current_state == MetisLinearState.STATE_ORIGINAL_LINEAR:
            # 获取 Weight 梯度
            grads['weight_grad'] = self.original_linear.weight.grad
            
            # 获取 Bias 梯度 (如果存在)
            if self.use_bias and self.original_linear.bias is not None:
                grads['bias_grad'] = self.original_linear.bias.grad

        # 2. 如果当前是 SVD 分解状态
        elif self.current_state == MetisLinearState.STATE_WEIGHT_DECOMPOSED_LINEAR:
            model = self.weight_svd_decomposition_model
            
            # --- 核心组件: U, V, S ---
            # ulinear.weight
            if hasattr(model.ulinear, 'weight'):
                grads['ulinear_grad'] = model.ulinear.weight.grad
            
            # vlinear.weight
            if hasattr(model.vlinear, 'weight'):
                grads['vlinear_grad'] = model.vlinear.weight.grad
            
            # s (奇异值向量)
            # 注意：s 是 Parameter，直接取 .grad
            if isinstance(model.s, torch.nn.Parameter):
                grads['s_grad'] = model.s.grad
            
            # --- 可选组件: Residual & Bias ---
            # 如果 rank > 0，linear_residual 会参与计算
            if model.forward_svd_rank > 0 and isinstance(model.linear_residual, nn.Module):
                if hasattr(model.linear_residual, 'weight'):
                    grads['residual_weight_grad'] = model.linear_residual.weight.grad
                
                # Bias 通常挂载在 residual 上 (参考 DecomposedLinear 的 decompose_weight 逻辑)
                if model.use_bias and hasattr(model.linear_residual, 'bias') and model.linear_residual.bias is not None:
                    grads['bias_grad'] = model.linear_residual.bias.grad

        return grads