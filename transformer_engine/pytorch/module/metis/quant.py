from dataclasses import dataclass
import enum
from typing import Any, Optional, List, Union
import torch

from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop
from transformer_engine.pytorch.constants import TE_DType
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

from .utils import TensorOffloadManager
from ...distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    symmetric_all_reduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    is_fp8_activation_recompute_enabled,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
import torch.nn.functional as F
def schedule_none(input_: torch.Tensor):
    return input_, 1.0


def schedule_l1_m1p5_s2(input_: torch.Tensor):
    input_[5:] *= 1.5
    return input_, 2.0

@dataclass
class RestoreInfo:
    shape:tuple = None # 数值是[M+N,1]
    keep_indices: torch.Tensor = None # shape [M]
    drop_indices: torch.Tensor = None # shape [N]
    drop_count:int = -1 # N

def cuda_time_call(fn, *args, **kwargs):
    """Run a callable on CUDA and measure elapsed time in milliseconds.

    Args:
        fn: callable to run.
        *args, **kwargs: forwarded to fn.

    Returns:
        A tuple (result, elapsed_ms). If fn returns multiple values, result
        is whatever fn returned.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Record start, run, record end, synchronize, compute elapsed
    start.record()
    result = fn(*args, **kwargs)
    # Ensure any CUDA kernels launched by fn are recorded before ending
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    return result, elapsed

def process_and_fill_matrix(x, old_noise = None, drop_rate = 0.8):
    """
    输入:
        x: 原始矩阵 (B*S, H)
        old_noise: 历史drop out噪声矩阵
        drop_rate : 需要删除的 token 比例
    输出:
        x_compact: RestoreInfo
    """
    
    B_S, _ = x.shape
    drop_count = int(B_S * drop_rate)
    drop_count = (drop_count // 8 ) * 8 # 保证 matmul k 轴能被 8 整除
    keep_count = B_S - drop_count
    
    if keep_count <= 0:
        raise ValueError("删除数量过多，保留数量必须大于0")

    # ==========================================
    # 获取 保留下标 (Keep) 和 删除下标 (Drop)
    # ==========================================
    if old_noise is None:
        # 使用噪声矩阵选取token
        noise = torch.rand(B_S, device=x.device)
    else: 
        noise = old_noise

    # A. 获取保留的下标 (噪声值最大的前 keep_count 个)
    _, keep_indices = torch.topk(noise, k=keep_count, dim=0, largest=True)
    keep_indices, _ = torch.sort(keep_indices, dim=0) # 保持时序
    
    # B. 获取被删除的下标 (噪声值最小的前 drop_count 个) -> 补集
    _, drop_indices = torch.topk(noise, k=drop_count, dim=0, largest=False)
    drop_indices, _ = torch.sort(drop_indices, dim=0) # 保持时序
    
    # 提取保留的数据
    x_compact = x.index_select(0,keep_indices)
    return x_compact, RestoreInfo(x.shape, keep_indices, drop_indices ,drop_count), noise

def get_fill_values(x: torch.Tensor, restore_info: RestoreInfo, strategy) -> torch.Tensor:
    """
    基于 x 的数值和 keep_indices，生成用于填补 drop_indices 位置的数据。
    """

    drop_count = restore_info.drop_count
    
    # 获取 x 的特征维度 K
    if x.dim() > 1:
        feature_dim = x.shape[-1]
    else:
        # 处理 x 是一维向量的情况
        feature_dim = 1
        x = x.unsqueeze(-1) 

    # 校验 drop_count
    if drop_count == -1 and restore_info.drop_indices is not None:
        drop_count = len(restore_info.drop_indices)
    
    # ------------------------------------------------------
    # 策略 1: Mean (均值填充)
    # 计算 x 的全局均值，填入所有缺失位置
    # ------------------------------------------------------
    if strategy == "mean":
        # 计算 x 的均值 [1, K]
        mean_val = x.mean(dim=0, keepdim=True)
        # 扩展成 [N, K]
        return mean_val.expand(drop_count, feature_dim)

    # ------------------------------------------------------
    # 策略 2: Nearest (最近邻填充) - 强依赖 x 的数值
    # 寻找物理位置(index)最近的 x，复用其数值
    # ------------------------------------------------------
    elif strategy == "nearest":
        if restore_info.keep_indices is None or restore_info.drop_indices is None:
            raise ValueError("Nearest strategy need keep_indices and drop_indices")
            
        # 1. 计算距离矩阵: |drop_index - keep_index|
        # drop: [N, 1], keep: [1, M] -> dist: [N, M]
        d_idx = restore_info.drop_indices.unsqueeze(1).float()
        k_idx = restore_info.keep_indices.unsqueeze(0).float()
        dist = torch.abs(d_idx - k_idx)
        
        # 2. 找到每个 drop 位置距离最近的 keep 位置的“下标” (范围 0 到 M-1)
        nearest_indices_in_x = torch.argmin(dist, dim=1) 
        
        # 3. 直接从 x 中取出对应的值作为填充值
        # fill_values[i] = x[ nearest_indices_in_x[i] ]
        return x[nearest_indices_in_x]

    # ------------------------------------------------------
    # 策略 3: Tile (平铺/复制)
    # 循环利用 x 的数值来填补
    # ------------------------------------------------------
    elif strategy == "tile":
        if x.shape[0] == 0:
            return torch.zeros((drop_count, feature_dim), device=x.device, dtype=x.dtype)
        
        # 计算需要重复多少次才能覆盖 drop_count
        num_repeats = (drop_count + x.shape[0] - 1) // x.shape[0]
        # 复制 x
        tiled = x.repeat(num_repeats, 1)
        # 截取前 drop_count 个
        return tiled[:drop_count]

    # ------------------------------------------------------
    # 策略 4: Random (随机) / Zeros
    # ------------------------------------------------------
    elif strategy == "random":
        return torch.randn((drop_count, feature_dim), device=x.device, dtype=x.dtype)
    
    elif strategy == "zeros":
        return torch.zeros((drop_count, feature_dim), device=x.device, dtype=x.dtype)

    else:
        raise NotImplementedError(f"Strategy {strategy} is not supported.")

def restore_matrix(x: torch.Tensor, restore_info: RestoreInfo,restore_strategy, reinfer_shape = False) -> torch.Tensor:
    """
    主函数：复原矩阵。
    步骤：
    1. 创建空矩阵
    2. 将 x 填入 keep_indices (保证原数据不丢失)
    3. 将生成的 fill_values 填入 drop_indices
    """
    # 1. 初始化画布
    target_shape = restore_info.shape
    # 如果没有传入shape，尝试自动推断
    if reinfer_shape or target_shape is None:
        total_len = x.shape[0] + restore_info.drop_count
        target_shape = (total_len, x.shape[-1])

        
    out = torch.zeros(target_shape, device=x.device, dtype=x.dtype)
    
    # 确保索引是 Long 类型以支持 scatter/index_copy
    keep_indices = restore_info.keep_indices.long()
    drop_indices = restore_info.drop_indices.long()
    
    # 2. 核心步骤：回填 x
    # 这一步保证了 keep_indices 位置上的数值绝对是 x 的原始数值
    # out[keep_indices] = x
    out.index_copy_(0, keep_indices, x)
    # 3. 核心步骤：填补 drop
    if restore_info.drop_count > 0:
        # 获取基于 x 生成的填充数据
        fill_values = get_fill_values(x, restore_info, restore_strategy)
        # 填入 drop 位置
        out.index_copy_(0,drop_indices,fill_values)

    return out

def grad_power_iteration_svd(dy, v_prev:torch.Tensor, rank = 64, t_iter = 1,niter = 2,eps:float = 1e-8)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    if v_prev is None:
        u, s, v = torch.svd_lowrank(dy.to(torch.float32), q=rank, niter=niter)
        u = u.to(dy.dtype)
        s = s.to(dy.dtype)
        v = v.to(dy.dtype)
        return u,s,v

    def iter_func(dy,v):
        p = dy @ v
        u, _ = torch.linalg.qr(p.to(torch.float32), mode = "reduced")
        u = u.to(dy.dtype)
        w = dy.T @ u
        return u,w

    v = v_prev
    u = None
    w = None

    for i in range(t_iter - 1):
        u,w = iter_func(dy,v)
        col_norm = torch.linalg.norm(w, dim = 0, keepdim = True)
        v = w / (col_norm + eps)
    # last iter
    u, w = iter_func(dy,v)
    s:torch.Tensor = torch.linalg.norm(w, dim=0,keepdim = False)
    v_new = w / (s.unsqueeze(0) + eps)

    return u,s,v_new

class MetisSvdFunction:

    @staticmethod
    @torch.no_grad()
    def svd_quant_gemm(x, y, output_dtype, output_quantizer=None, layout="TN", grad=False,nvtx_label="",**kargs):
        kargs.update(
            {
                "A":x,
                "B":y,
                "accumulate":False,
                "layout":layout,
                "quantization_params":output_quantizer,
                "out_dtype":output_dtype,
                "use_split_accumulator":False,
                "grad":grad,
            }
        )
        nvtx_range_push(f"transformer_engine.MetisSvdFunction.svd_quant_gemm_{nvtx_label}.gemm")
        gemm_out, *_ = general_gemm(
            **kargs
        )
        nvtx_range_pop(f"transformer_engine.MetisSvdFunction.svd_quant_gemm_{nvtx_label}.gemm")
        return gemm_out

    @staticmethod
    @torch.no_grad()
    @torch.compile
    def svd_lowrank_quant_grad_output(grad_output: torch.Tensor, grad_output_shape, **kargs):
        assert grad_output_shape is not None
        grad_output = grad_output.view(grad_output_shape)
        return MetisSvdFunction.svd_lowrank_quant(grad_output, **kargs)

    @staticmethod
    @torch.no_grad()
    # @torch.compile
    def svd_lowrank_quant(
        input_: torch.Tensor,
        input_quantizer: "Quantizer" = None,
        rank=60,
        niter=2,
        broadcast_dim=-1,
        is_backward=False,
        enable_gradient_accumulation_optimization=False,
        load_history=False,
        history_list: Union[dict, TensorOffloadManager] = {},
    ):

        # for backward, input_ has already shaped into 2d tensor.
        # input_ shape [b,s,h]
        input_quantizer.set_usage(rowwise=True, columnwise=True)
        input_shape = input_.shape
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)  # [s,h]
        else:
            cinput = input_
        original_shape = cinput.shape  # [s,h]
        if load_history and enable_gradient_accumulation_optimization and is_backward:
            ker, de_svd_gemm_out = history_list["svd_history"]
            # print("load")
        else:
            cinput = cinput.view(-1, original_shape[-1])  # [s,h] or [b*s,h]
            # print(f"cinput shape==",cinput.shape)
            # ug, sg, vg = torch.svd(cinput.to(torch.float32))
            ug, sg, vg = torch.svd_lowrank(cinput.to(torch.float32), q=rank, niter=niter)
            # print("running svd")
            ug = ug.to(input_.dtype)
            sg = sg.to(input_.dtype)
            sg = torch.diag(sg)
            vg = vg.to(input_.dtype)
            # print(f"cinput.shape={cinput.shape},input_.shape={input_.shape},sg.size={sg.size()}, ug.size={ug.size()}, vg.size={vg.size()}")
            ker = ug @ sg @ vg.T  # [s,h] or [b*s,h]
            if broadcast_dim >= 0:
                ker = ker.unsqueeze(broadcast_dim)  # [1,s,h]
            else:
                ker = ker.view(input_shape)  # [b,s,h]

            if input_quantizer is None:
                gemm_out = MetisSvdFunction.svd_quant_gemm(
                    sg, ug.T, input_.dtype, input_quantizer, layout="NT", nvtx_label="U@S"
                )
                de_svd_gemm_out = MetisSvdFunction.svd_quant_gemm(
                    vg, gemm_out, input_.dtype, None, layout="TN", nvtx_label="U@S@V"
                )

            if input_quantizer is not None:
                ug = input_quantizer(ug)
                vg = input_quantizer(vg)
                sg = input_quantizer(sg)
                gemm_out = MetisSvdFunction.svd_quant_gemm(
                    sg, ug, input_.dtype, input_quantizer, layout="NN", nvtx_label="U@S"
                )
                de_svd_gemm_out = MetisSvdFunction.svd_quant_gemm(
                    vg, gemm_out, input_.dtype, None, layout="TN", nvtx_label="U@S@V"
                )

            # [s,h] or [b*s,h]
            if broadcast_dim >= 0:
                de_svd_gemm_out = de_svd_gemm_out.unsqueeze(broadcast_dim)  # [1,s,h]
            else:
                de_svd_gemm_out = de_svd_gemm_out.view(input_shape)  # [b,s,h]
            if enable_gradient_accumulation_optimization and is_backward:
                # print("storing history_list----")
                # history_list.clear()
                history_list["svd_history"] = [ker, de_svd_gemm_out]

        # de_svd_gemm_out
        # ker
        def fused_add_sub(input_, ker, de_svd_gemm_out):
            # input_: [b, s, h]
            # ker:    [b, s, h]
            # de_svd_gemm_out: [b, s, h]

            # 原逻辑：input_res = input_ - ker
            #         out_tensor = de_svd_gemm_out + input_res
            # fuse 后就是:
            return de_svd_gemm_out + (input_ - ker)

        # compiled_fused_add_sub = torch.compile(fused_add_sub)
        compiled_fused_add_sub = fused_add_sub
        # input_res = input_ - ker #[b,s,h]
        # out_tensor = de_svd_gemm_out + input_res #[b,s,h]
        out_tensor = compiled_fused_add_sub(input_, ker, de_svd_gemm_out)
        # output_fp4 = input_quantizer(out_tensor)
        # return out_tensor
        return out_tensor

    @staticmethod
    @torch.no_grad()
    @torch.compile
    def svd_lowrank_quant_grad_output_separate_residual(grad_output: torch.Tensor, grad_output_shape, **kargs):
        assert grad_output_shape is not None
        grad_output = grad_output.view(grad_output_shape)
        return MetisSvdFunction.svd_lowrank_quant_separate_residual(grad_output, **kargs)

    @staticmethod
    def _validate_tp_config(tp_size: int, tp_strategy: str, token_drop_rate: float):
        """验证TP配置参数"""
        if tp_size > 1:
            if tp_strategy not in ["allgather", "fully_distributed"]:
                raise ValueError(
                    f"tp_strategy must be 'allgather' or 'fully_distributed', "
                    f"got '{tp_strategy}'"
                )
            
            if tp_strategy == "fully_distributed":
                raise NotImplementedError(
                    "fully_distributed TP strategy is not yet implemented. "
                    "Please use 'allgather' strategy instead."
                )
            
            if token_drop_rate >= 0 and tp_strategy != "allgather":
                raise ValueError(
                    f"When token_drop_rate is enabled with TP (tp_size={tp_size}), "
                    f"tp_strategy must be 'allgather' to ensure noise consistency. "
                    f"Got tp_strategy='{tp_strategy}'"
                )

    @staticmethod
    def _prepare_input_with_token_drop(
        input_: torch.Tensor,
        token_drop_rate: float,
        broadcast_dim: int,
        is_backward: bool,
        load_history: bool,
        history_list: Union[dict, TensorOffloadManager],
    ):
        """处理token drop和输入准备"""
        restore_info = None
        
        if token_drop_rate >= 0:
            # token drop处理
            if not is_backward:
                if not load_history:
                    cinput, restore_info, noise = process_and_fill_matrix(
                        input_, None, token_drop_rate)
                    history_list["token_drop_noise"] = noise
                else:
                    old_noise = history_list["token_drop_noise"]
                    cinput, restore_info, noise = process_and_fill_matrix(
                        input_, old_noise, token_drop_rate)
            else:
                forward_noise = history_list.get("token_drop_noise", None)
                cinput, restore_info, noise = process_and_fill_matrix(
                    input_, forward_noise, token_drop_rate)
        elif broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)
        else:
            cinput = input_
            
        return cinput, restore_info

    @staticmethod
    def _allgather_for_svd(
        cinput: torch.Tensor,
        tp_size: int,
        tp_strategy: str,
        tp_group,
        parallel_mode: str,
        is_backward: bool,
    ):
        """TP allgather操作"""
        should_allgather = (
            tp_size > 1 and 
            tp_strategy == "allgather" and 
            ((parallel_mode == "row" and not is_backward) or 
             (parallel_mode == "column" and is_backward))
        )
        
        if should_allgather:
            gathered_list = [torch.zeros_like(cinput) for _ in range(tp_size)]
            torch.distributed.all_gather(gathered_list, cinput, group=tp_group)
            cinput_for_svd = torch.cat(gathered_list, dim=-1)
            return cinput_for_svd, True
        else:
            return cinput, False

    @staticmethod
    def _perform_svd_and_quantize(
        cinput: torch.Tensor,
        input_quantizer: "Quantizer",
        rank: int,
        niter: int,
        broadcast_dim: int,
        input_dtype: torch.dtype,
    ):
        """执行SVD并量化"""
        ug, sg, vg = torch.svd_lowrank(cinput.to(torch.float32), q=rank, niter=niter)
        
        ug = ug.to(input_dtype)
        sg = sg.to(input_dtype)
        if len(sg.shape) == 1:
            sg = torch.diag(sg)
        vg = vg.to(input_dtype)
        
        ug_sg = ug @ sg
        ker = ug_sg @ vg.T
        
        if broadcast_dim >= 0:
            ker = ker.unsqueeze(broadcast_dim)
        
        return ug_sg, vg, ker

    @staticmethod
    def _split_allgathered_results(
        ug_sg: torch.Tensor,
        vg: torch.Tensor,
        ker: torch.Tensor,
        cinput_for_svd: torch.Tensor,
        tp_size: int,
        tp_group,
    ):
        """分割allgather后的SVD结果"""
        from ...distributed import get_distributed_rank
        
        tp_rank = get_distributed_rank(tp_group)
        hidden_size = cinput_for_svd.shape[-1]
        local_hidden_size = hidden_size // tp_size
        start_idx = tp_rank * local_hidden_size
        end_idx = start_idx + local_hidden_size
        
        # vg: [hidden, rank] -> 在第0维分割
        vg = vg[start_idx:end_idx, :]
        # ker: [seq_len, hidden] -> 在第1维分割
        ker = ker[:, start_idx:end_idx]
        # ug_sg: [seq_len, rank] -> 不需要分割
        
        return ug_sg, vg, ker

    @staticmethod
    def _compute_residual(
        input_: torch.Tensor,
        ker: torch.Tensor,
        input_quantizer: "Quantizer",
        keep_dim: bool,
        should_reshape: bool,
        input_shape: tuple,
    ):
        """计算残差"""
        if keep_dim and should_reshape:
            res = input_ - ker
            res = res.view(input_shape)
        else:
            res = input_ - ker
        
        if input_quantizer is not None:
            res = input_quantizer(res)
        
        return res

    @staticmethod
    @torch.no_grad()
    def svd_lowrank_quant_separate_residual_forward(
        input_: torch.Tensor,
        input_quantizer: "Quantizer",
        rank=64,
        niter=2,
        token_drop_rate: float = -1.0,
        broadcast_dim=-1,
        keep_dim=False,
        history_list: Union[dict, TensorOffloadManager] = {},
        restore_strategy = "tile",
        tp_size: int = 1,
        tp_group = None,
        tp_strategy: str = "allgather",
        parallel_mode: str = "",
        use_power_iteration_svd = False,
        power_iteration_time = 1,
        enable_history_optimization = False,
        load_history = False,
    ):
        """
        SVD低秩量化前向传播
        
        Args:
            input_: 输入张量
            input_quantizer: 量化器
            rank: SVD秩
            niter: SVD迭代次数
            token_drop_rate: token丢弃率（-1表示不丢弃）
            broadcast_dim: 广播维度
            keep_dim: 是否保持维度
            history_list: 历史数据存储
            restore_strategy: 恢复策略
            tp_size: TP并行大小
            tp_group: TP通信组
            tp_strategy: TP策略
            parallel_mode: 并行模式
            use_power_iteration_svd: 是否使用幂迭代SVD
            power_iteration_time: 幂迭代次数
            enable_history_optimization: 是否启用历史优化
            load_history: 是否加载历史记录
        
        Returns:
            ug_sg: U @ S矩阵
            vg: V矩阵（量化后）
            res: 残差（量化后）
            restore_info: token恢复信息
        """
        # 验证TP配置
        MetisSvdFunction._validate_tp_config(tp_size, tp_strategy, token_drop_rate)
        
        # 准备输入形状
        input_shape = input_.shape
        should_reshape = False
        if broadcast_dim < 0 and len(input_shape) > 2:
            should_reshape = True
            input_ = input_.view(-1, input_.shape[-1])
        
        # 处理token drop
        cinput, restore_info = MetisSvdFunction._prepare_input_with_token_drop(
            input_, token_drop_rate, broadcast_dim, 
            is_backward=False, load_history=False, history_list=history_list
        )
        
        # TP allgather
        cinput_for_svd, did_allgather = MetisSvdFunction._allgather_for_svd(
            cinput, tp_size, tp_strategy, tp_group, parallel_mode, is_backward=False
        )
        
        # 根据不同策略执行SVD
        if use_power_iteration_svd and enable_history_optimization:
            # 使用幂迭代SVD并启用历史优化
            if load_history and "forward_svd_history" in history_list:
                ker, ug_sg, vg = history_list["forward_svd_history"]
            else:
                ker, ug_sg, vg = None, None, None
            
            u, s, v = grad_power_iteration_svd(
                cinput_for_svd, vg, rank, power_iteration_time, niter
            )
            ug_sg = u @ torch.diag(s)
            ker = ug_sg @ v.T
            vg = input_quantizer(v)
            
            # 只有在不加载历史或历史不存在时才存储新结果
            if not load_history or "forward_svd_history" not in history_list:
                # 存储bf16数据，不是量化后的数据
                history_list["forward_svd_history"] = [ker, ug_sg, v]
            
        elif use_power_iteration_svd:
            # 仅使用幂迭代SVD，不使用历史优化
            u, s, v = grad_power_iteration_svd(
                cinput_for_svd, None, rank, power_iteration_time, niter
            )
            ug_sg = u @ torch.diag(s)
            ker = ug_sg @ v.T
            vg = input_quantizer(v)
            
        else:
            # 标准SVD
            ug_sg, vg, ker = MetisSvdFunction._perform_svd_and_quantize(
                cinput_for_svd, input_quantizer, rank, niter, broadcast_dim, input_.dtype
            )
        
        # 恢复token
        if restore_info is not None:
            ker = restore_matrix(ker, restore_info, restore_strategy)
        
        # 分割allgather的结果
        if did_allgather:
            ug_sg, vg, ker = MetisSvdFunction._split_allgathered_results(
                ug_sg, vg, ker, cinput_for_svd, tp_size, tp_group
            )

        # # 量化vg
        vg = input_quantizer(vg)

        # 计算残差
        res = MetisSvdFunction._compute_residual(
            input_, ker, input_quantizer, keep_dim, should_reshape, input_shape
        )
        
        return ug_sg, vg, res, restore_info

    @staticmethod
    @torch.no_grad()
    def svd_lowrank_quant_separate_residual_backward(
        input_: torch.Tensor,
        input_quantizer: "Quantizer",
        rank=64,
        niter=2,
        token_drop_rate: float = -1.0,
        broadcast_dim=-1,
        enable_gradient_accumulation_optimization=False,
        load_history=False,
        use_power_iteration_svd = False,
        power_iteration_time = 1,
        keep_dim=False,
        history_list: Union[dict, TensorOffloadManager] = {},
        restore_strategy = "tile",
        tp_size: int = 1,
        tp_group = None,
        tp_strategy: str = "allgather",
        parallel_mode: str = "",
    ):
        """
        SVD低秩量化反向传播
        
        Args:
            input_: 输入梯度张量
            input_quantizer: 量化器
            rank: SVD秩
            niter: SVD迭代次数
            token_drop_rate: token丢弃率（-1表示不丢弃）
            broadcast_dim: 广播维度
            enable_gradient_accumulation_optimization: 是否启用梯度累积优化
            load_history: 是否加载历史数据
            use_grad_power_iteration_svd: 是否使用梯度幂迭代SVD
            grad_power_iteration_time: 梯度幂迭代次数
            keep_dim: 是否保持维度
            history_list: 历史数据存储
            restore_strategy: 恢复策略
            tp_size: TP并行大小
            tp_group: TP通信组
            tp_strategy: TP策略
            parallel_mode: 并行模式
        
        Returns:
            ug_sg: U @ S矩阵
            vg: V矩阵（量化后）
            res: 残差（量化后）
            restore_info: token恢复信息
        """
        # 验证TP配置
        MetisSvdFunction._validate_tp_config(tp_size, tp_strategy, token_drop_rate)
        
        # 准备输入形状
        input_shape = input_.shape
        should_reshape = False
        if broadcast_dim < 0 and len(input_shape) > 2:
            should_reshape = True
            input_ = input_.view(-1, input_.shape[-1])
        
        # 处理token drop
        cinput, restore_info = MetisSvdFunction._prepare_input_with_token_drop(
            input_, token_drop_rate, broadcast_dim,
            is_backward=True, load_history=load_history, history_list=history_list
        )
        
        if not (enable_gradient_accumulation_optimization and load_history and not use_power_iteration_svd):
            # TP allgather, 直接读取历史记录的时候，不需要重新计算svd也不需要通信，所以可以跳过allgather这一步
            cinput_for_svd, did_allgather = MetisSvdFunction._allgather_for_svd(
                cinput.contiguous(), tp_size, tp_strategy, tp_group, parallel_mode, is_backward=True
            )
        else:
            cinput_for_svd = cinput
            did_allgather = True
        
        # 根据不同策略执行SVD
        if use_power_iteration_svd:
            # 使用梯度幂迭代SVD
            if load_history:
                ker, ug_sg, vg = history_list["backward_svd_history"]
            else:
                ker, ug_sg, vg = None, None, None
            
            u, s, v = grad_power_iteration_svd(
                cinput_for_svd, vg, rank, power_iteration_time, niter
            )
            ug_sg = u @ torch.diag(s)
            ker = ug_sg @ v.T
            vg = input_quantizer(v)
            
            if restore_info is not None:
                ker = restore_matrix(ker, restore_info, restore_strategy)
            
            # 存储bf16数据，不是量化后的数据
            history_list["backward_svd_history"] = [ker, ug_sg, v]
            
        elif enable_gradient_accumulation_optimization and load_history:
            # 直接加载历史数据
            ker, ug_sg, vg = history_list["backward_svd_history"]
        else:
            # 标准SVD
            ug_sg, vg, ker = MetisSvdFunction._perform_svd_and_quantize(
                cinput_for_svd, input_quantizer, rank, niter, broadcast_dim, input_.dtype
            )
            
            if restore_info is not None:
                ker = restore_matrix(ker, restore_info, restore_strategy)

            if enable_gradient_accumulation_optimization:
                # 这里存储的是完整的矩阵，所以拿取历史数据的时候需要切分数据
                history_list["svd_history"] = [ker, ug_sg, vg]
        
        # 分割allgather的结果
        if did_allgather:
            ug_sg, vg, ker = MetisSvdFunction._split_allgathered_results(
                ug_sg, vg, ker, cinput_for_svd, tp_size, tp_group
            )

        # # 量化vg
        vg = input_quantizer(vg)

        # 计算残差
        res = MetisSvdFunction._compute_residual(
            input_, ker, input_quantizer, keep_dim, should_reshape, input_shape
        )
        
        return ug_sg, vg, res, restore_info

    @staticmethod
    @torch.no_grad()
    def svd_lowrank_quant_separate_residual(
        input_: torch.Tensor,
        input_quantizer: "Quantizer",
        rank=64,
        niter=2,
        token_drop_rate: float = -1.0,
        broadcast_dim=-1,
        is_backward=False,
        keep_dim=False,
        history_list: Union[dict, TensorOffloadManager] = {},
        restore_strategy = "tile",
        tp_size: int = 1,
        tp_group = None,
        tp_strategy: str = "allgather",
        parallel_mode: str = "",
        use_power_iteration_svd = False,
        power_iteration_time = 1,
        enable_history_optimization = False,
        load_history = False,
    ):
        """
        SVD低秩量化（分离残差版本）- 兼容接口
        
        SVD低秩量化（分离残差版本）- 兼容接口
        
        根据is_backward参数自动路由到前向或反向函数
        
        统一参数说明:
            use_power_iteration_svd: 是否使用幂迭代SVD（前向/反向通用）
            power_iteration_time: 幂迭代次数（前向/反向通用）
            enable_history_optimization: 是否启用历史优化（前向/反向通用）
            load_history: 是否加载历史记录（前向/反向通用）
        """
        if is_backward:
            return MetisSvdFunction.svd_lowrank_quant_separate_residual_backward(
                input_=input_,
                input_quantizer=input_quantizer,
                rank=rank,
                niter=niter,
                token_drop_rate=token_drop_rate,
                broadcast_dim=broadcast_dim,
                enable_gradient_accumulation_optimization=enable_history_optimization,
                load_history=load_history,
                use_grad_power_iteration_svd=use_power_iteration_svd,
                grad_power_iteration_time=power_iteration_time,
                keep_dim=keep_dim,
                history_list=history_list,
                restore_strategy=restore_strategy,
                tp_size=tp_size,
                tp_group=tp_group,
                tp_strategy=tp_strategy,
                parallel_mode=parallel_mode,
            )
        else:
            return MetisSvdFunction.svd_lowrank_quant_separate_residual_forward(
                input_=input_,
                input_quantizer=input_quantizer,
                rank=rank,
                niter=niter,
                token_drop_rate=token_drop_rate,
                broadcast_dim=broadcast_dim,
                keep_dim=keep_dim,
                history_list=history_list,
                restore_strategy=restore_strategy,
                tp_size=tp_size,
                tp_group=tp_group,
                tp_strategy=tp_strategy,
                parallel_mode=parallel_mode,
                use_power_iteration_svd=use_power_iteration_svd,
                power_iteration_time=power_iteration_time,
                enable_history_optimization=enable_history_optimization,
                load_history=load_history,
            )

    @staticmethod
    @torch.no_grad()
    def svd_fullrank_quant(input_: torch.Tensor, quantizer: "Quantizer"):
        ### Full rank SVD quantization
        ug, sg, vg = torch.svd(input_.to(torch.float32), some=True)
        ug = ug.to(input_.dtype)
        sg = torch.diag(sg.to(input_.dtype))
        vg = vg.to(input_.dtype)
        ug_nvfp4 = quantizer.make_empty(
            ug.shape, dtype=ug.dtype, device=ug.device, requires_grad=False
        )
        vg_nvfp4 = quantizer.make_empty(
            vg.shape, dtype=vg.dtype, device=vg.device, requires_grad=False
        )
        sg_nvfp4 = quantizer.make_empty(
            sg.shape, dtype=sg.dtype, device=sg.device, requires_grad=False
        )
        ug_quant = quantizer.update_quantized(ug, ug_nvfp4)
        vg_quant = quantizer.update_quantized(vg, vg_nvfp4)
        sg_quant = quantizer.update_quantized(sg, sg_nvfp4)
        gemm_out = MetisSvdFunction.svd_quant_gemm(
            sg_quant, ug_quant, input_.dtype, quantizer, layout="NN", nvtx_label="U@S"
        )
        de_svd_gemm_out = MetisSvdFunction.svd_quant_gemm(
            vg_quant, gemm_out, input_.dtype, quantizer, layout="TN", nvtx_label="U@S@V"
        )
        return de_svd_gemm_out
    
    @staticmethod
    def gemm_with_separate_residual(
        u_s, v, res, weightmat, activation_dtype, input_quantizer, is_grad=False,output_shape=None, restore_info:RestoreInfo = None, restore_strategy = "tile"
    ):
            # ------------------------------------------------------
            # forward:
            # y = x @ w.T
            #   = (u_s @ v.T + res) @ w.T
            #   = u_s @ (v.T @ w.T) + res @ w.T
            #   = u_s @ (w @ v).T + res @ w.T
            # backward:
            # dx = dy @ w
            #    = (u_s @ v.T + res) @ w
            #    = u_s @ (v.T @ w) + res @ w 
            #    = u_s @ (w.T @ v).T + res @ w
            # ------------------------------------------------------
            if not isinstance(u_s,QuantizedTensorStorage):
                u_s = input_quantizer(u_s)

            if not isinstance(v,QuantizedTensorStorage):
                v = input_quantizer(v)

            if is_grad:
                layout_list = ["NT","TN","NN"]
            else:
                layout_list = ["NN","TN","TN"]

            v_weight_out = MetisSvdFunction.svd_quant_gemm(v,weightmat,activation_dtype,input_quantizer,layout_list[0],is_grad,"V@W")
            low_rank_output = MetisSvdFunction.svd_quant_gemm(v_weight_out,u_s,activation_dtype,None,layout_list[1],is_grad,"V@W")
            input_res_weight_out = MetisSvdFunction.svd_quant_gemm(weightmat,res,activation_dtype,None,layout_list[2],is_grad,"INPUT_RES@W")

            if restore_info is not None:
                low_rank_output_expand = restore_matrix(low_rank_output, restore_info, restore_strategy, reinfer_shape=True)
            else:
                low_rank_output_expand = low_rank_output
            gemm_out = low_rank_output_expand + input_res_weight_out
            if output_shape is not None:
                gemm_out = gemm_out.view(output_shape)
                low_rank_output_expand = low_rank_output_expand.view(output_shape)
            return gemm_out,low_rank_output_expand

    @staticmethod
    def gemm_with_weight_grad_separate_residual_full_rank_svd_bf16(
        input_ug_sg, input_vg, input_res, grad_ug_sg, grad_vg, grad_res, activation_dtype, input_quantizer, tensor_reshape=False,skip_residual=False
    ):
            # ------------------------------------------------------
            # 仅bf16格式使用
            # Compute according to:
            # where A=input_ug_sg, [b,s,low_rank_forward]
            #       B=input_vg, [b,low_rank_forward,h]
            #       C=input_res, [b,s,h]
            #       D=grad_ug_sg, [b,s,low_rank_backward]
            #       E=grad_vg, [b,low_rank_backward,h]
            #       F=grad_res [b,s,h]
            # caution: B and E are column major, others are row major
            # x = (A@B + C), dy = (D@E + F)
            #  dw = dy.T @ x
            #     = (D@E + F).T @ (A@B + C)
            #     = (E.T@D.T+F.T) (A@B + C)
            #     = (E.T @ (D.T @ A) @ B) + (F.T @ A @ B) + (E.T @ (D.T @ C)) + (F.T @ C)
            # Use `svd_quant_gemm` for every GEMM to preserve quantization behavior.
            if tensor_reshape:
                A = input_ug_sg.view(-1, input_ug_sg.shape[-1])
                B = input_vg.view(-1, input_vg.shape[-1])
                C = input_res.view(-1, input_res.shape[-1])
                D = grad_ug_sg.view(-1, grad_ug_sg.shape[-1])
                E = grad_vg.view(-1, grad_vg.shape[-1])
                F = grad_res.view(-1, grad_res.shape[-1])
            else:
                A = input_ug_sg
                B = input_vg
                C = input_res
                D = grad_ug_sg
                E = grad_vg
                F = grad_res

            # 1. DA_T = D.T @ A
            DA_T = MetisSvdFunction.svd_quant_gemm(A, D, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="D.T@A")
            
            # 2. EDA = E.T @ DA_T
            EDA = MetisSvdFunction.svd_quant_gemm(DA_T, E.T, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="E.T@DA_T")
            
            # 3. term1 = EDA @ B
            term1 = MetisSvdFunction.svd_quant_gemm(B.T, EDA, activation_dtype, None, layout="NN", grad=True, nvtx_label="EDA@B")
            dw = term1
            if not skip_residual:
                # 4. FA = F.T @ A
                FA = MetisSvdFunction.svd_quant_gemm(A,F, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="F.T@A")
                
                # 5. term2 = FA @ B
                term2 = MetisSvdFunction.svd_quant_gemm(B.T,FA, activation_dtype, None, layout="NN", grad=True, nvtx_label="FA@B")
                
                # 6. DC = D.T @ C
                DC = MetisSvdFunction.svd_quant_gemm(C,D, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="D.T@C")
                
                # 7. term3 = E.T @ DC
                term3 = MetisSvdFunction.svd_quant_gemm(DC,E.T, activation_dtype, None, layout="NT", grad=True, nvtx_label="E.T@DC")
                
                # 8. term4 = F.T @ C
                term4 = MetisSvdFunction.svd_quant_gemm(C,F, activation_dtype, None, layout="NT", grad=True, nvtx_label="F.T@C")
                dw = dw + term2 + term3 + term4
            return dw

    @staticmethod
    @torch.compile
    def gemm_with_weight_grad_separate_residual(
        input_u_s,
        input_v,
        input_res,
        grad_u_s,
        grad_v,
        grad_res, 
        activation_dtype,
        input_quantizer,
        tensor_reshape=False,
        skip_residual=False,
        input_restoreinfo = None,
        grad_restoreinfo = None,
        restore_strategy = "tile",
    ):
            # ------------------------------------------------------
            # Compute according to:
            # where A=input_ug_sg, [b,s,low_rank_forward]
            #       B=input_v, [low_rank_forward,h]
            #       C=input_res, [b,s,h]
            #       D=grad_ug_sg, [b,s,low_rank_backward]
            #       E=grad_vg, [low_rank_backward,h]
            #       F=grad_res [b,s,h]
            # x = (A@B.T + C), dy = (D@E.T + F)
            #  dw = dy.T @ x
            #     = (D@E.T + F).T @ (A@B.T + C)
            #     = (E@D.T+F.T) (A@B.T + C)
            #     = (E @ (D.T @ A) @ B) + (F.T @ A @ B.T) + ( @ (D.T @ C)) + (F.T @ C)
            # Use `svd_quant_gemm` for every GEMM to preserve quantization behavior.
            if tensor_reshape:
                A = input_u_s.view(-1, input_u_s.shape[-1])
                B = input_v.view(-1, input_v.shape[-1])
                C = input_res.view(-1, input_res.shape[-1])
                D = grad_u_s.view(-1, grad_u_s.shape[-1])
                E = grad_v.view(-1, grad_v.shape[-1])
                F = grad_res.view(-1, grad_res.shape[-1])
            else:
                A = input_u_s
                B = input_v
                C = input_res
                D = grad_u_s
                E = grad_v
                F = grad_res

            if not isinstance(A,QuantizedTensorStorage):
                if input_restoreinfo:
                    A = restore_matrix(A, input_restoreinfo, restore_strategy, True)
                A = input_quantizer(A)

            if not isinstance(D,QuantizedTensorStorage):
                if grad_restoreinfo:
                    D = restore_matrix(D, grad_restoreinfo, restore_strategy, True)
                D = input_quantizer(D)

            # 1. DA_T = D.T @ A
            DA_T = MetisSvdFunction.svd_quant_gemm(A, D, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="D.T@A")

            # 2. EDA = E @ DA_T
            EDA = MetisSvdFunction.svd_quant_gemm(DA_T, E, activation_dtype, input_quantizer, layout="NN", grad=True, nvtx_label="E.T@DA_T")
            # print("E.T @ DA_T",E @ DA_T)
            # 3. term1 = EDA @ B
            term1 = MetisSvdFunction.svd_quant_gemm(B, EDA, activation_dtype, None, layout="TN", grad=True, nvtx_label="EDA@B")
            dw = term1
            if not skip_residual:
                # 4. FA = F.T @ A
                FA = MetisSvdFunction.svd_quant_gemm(A,F, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="F.T@A")
                
                # 5. term2 = FA @ B
                term2 = MetisSvdFunction.svd_quant_gemm(B,FA, activation_dtype, None, layout="TN", grad=True, nvtx_label="FA@B")
                
                # 6. DC = D.T @ C
                DC = MetisSvdFunction.svd_quant_gemm(C,D, activation_dtype, input_quantizer, layout="NT", grad=True, nvtx_label="D.T@C")
                
                # 7. term3 = E @ DC
                term3 = MetisSvdFunction.svd_quant_gemm(DC,E, activation_dtype, None, layout="NN", grad=True, nvtx_label="E.T@DC")
                
                # 8. term4 = F.T @ C
                term4 = MetisSvdFunction.svd_quant_gemm(C,F, activation_dtype, None, layout="NT", grad=True, nvtx_label="F.T@C")
                dw = term1 + term2 + term3 + term4
            return dw

@dataclass
class MeanQuantResult:
    """封装mean_quant的输出结果"""
    quant_input: torch.Tensor  # 量化后的输入，shape [b*s, h]
    quant_mean: torch.Tensor   # 量化后的均值，shape [b, h]
    shape: tuple         # 原始输入形状，例如 (b, s, h)

@dataclass
class MeanConcatQuantResult:
    """封装mean_concat_quant的输出结果"""
    quant_concat: torch.Tensor  # 量化后的拼接张量，shape [b*(s+1), h]
    quant_concat_reverse: torch.Tensor = None   # 量化后的拼接张量，shape [b*(s+1), h]
    shape: tuple = tuple()         # 原始拼接形状，例如 (b, s, h)

class MetisMeanFunction:
    @staticmethod
    def mean_concat_quant(input_tensor: torch.Tensor, quantizer: "Quantizer"):
        """
        对输入tensor计算均值并拼接后进行量化，返回2D张量
        
        Args:
            input_tensor: 输入张量，shape为[b, s, h]
            quantizer: 量化器，必须是1D量化器
            
        Returns:
            MeanConcatQuantResult: 包含量化后的2D张量和原始形状信息
        """
        # 保存原始形状
        original_shape = input_tensor.shape  # (b, s, h)
        hidden_size = original_shape[-1]
        
        # 转换为2D张量
        input_concat_2d = input_tensor.view(-1, hidden_size)  # shape: [b*s, h]

        # 计算沿着序列维度的均值
        x = input_concat_2d.mean(dim=0, keepdim=True)  # shape: [1, h]
        
        # 拼接原始输入和均值
        input_concat = torch.cat([input_concat_2d, x], dim=0)  # shape: [b*s+1, h]

        # 使用量化器进行量化（要求是1D量化）
        quant_input_concat = quantizer(input_concat)
        
        # 返回包装结果
        return MeanConcatQuantResult(
            quant_concat=quant_input_concat,
            shape=original_shape
        )
    
    @staticmethod
    def mean_quant(input_tensor: torch.Tensor, quantizer: "Quantizer"):
        """
        分别对输入tensor和均值进行量化，返回2D张量
        
        Args:
            input_tensor: 输入张量，shape为[b, s, h]
            quantizer: 量化器
            
        Returns:
            MeanQuantResult: 包含量化后的2D张量和原始形状信息
        """
        # 保存原始形状
        input_shape = input_tensor.shape  # (b, s, h)
        hidden_size = input_shape[-1]
        
        # 转换为2D张量
        input_tensor_2d = input_tensor.view(-1, hidden_size)  # shape: [b*s, h]
        input_tensor_mean_dim0 = input_tensor_2d.mean(dim=0, keepdim=False).expand_as(input_tensor_2d)  # shape: [b, 1, h]        
        input_tensor_mean_dim1 = input_tensor_2d.mean(dim=1, keepdim=True).expand_as(input_tensor_2d)
        input_tensor_mean = input_tensor_mean_dim1 * input_tensor_mean_dim0
        # x_2d = input_tensor_2d.view(-1, hidden_size)  # shape: [b, h]
        
        # 分别进行量化
        # must pad to [8,hidden]
        # pad_amount=15
        # quant_mean = F.pad(input_tensor_mean, (0, 0, 0, pad_amount), mode='constant', value=0)
        quant_tensor_mean = quantizer(input_tensor_mean)
        quant_x = quantizer(input_tensor_2d)
        
        # 返回包装结果
        return MeanQuantResult(
            quant_input=quant_x,
            quant_mean=quant_tensor_mean,
            shape=input_shape,
        )
    
    @staticmethod
    def gemm_operation_with_mean_quant(quant_result: MeanQuantResult, weight: torch.Tensor, 
                                       activation_dtype: torch.dtype = None, 
                                       quantizer: "Quantizer" = None):
        """
        基于mean_quant输出的GEMM操作：out_final = out_0 - out_1
        其中 out_0 = quant_input @ weight.T
              out_1 = quant_mean @ weight.T
        
        Args:
            quant_result: MeanQuantResult对象，包含量化张量和形状信息
            weight: 权重矩阵，shape为[out_features, h]
            activation_dtype: 激活数据类型
            quantizer: 量化器
            
        Returns:
            out_final: 最终输出，根据input_shape和weight shape推断输出形状
        """
        print("running gemm_operation_with_mean_quant ")
        # 从quant_result中提取数据
        quant_input = quant_result.quant_input  # [b*s, h]
        quant_mean = quant_result.quant_mean    # [b, h]
        input_shape = quant_result.shape  # (b, s, h) or (b,h)

        output_shape = list(input_shape)
        output_shape[-1] = weight.shape[0]

        # 从形状信息推断维度
        # batch_size = input_shape[0]
        # seq_len = input_shape[1]
        # out_features = weight.shape[0]
        
        # 使用TE的GEMM接口计算out_0: [b*s, h] @ [h, out_features]^T -> [b*s, out_features]
        out_0 = MetisSvdFunction.svd_quant_gemm(
            weight, quant_input, activation_dtype, 
            quantizer, layout="TN", nvtx_label="input@weight"
        )  # [b*s, out_features]
        
        # 使用TE的GEMM接口计算out_1: [b, h] @ [h, out_features]^T -> [b, out_features]
        out_1 = MetisSvdFunction.svd_quant_gemm(
            weight, quant_mean, activation_dtype, 
            quantizer, layout="TN", nvtx_label="mean@weight"
        )  # [1, out_features]
        # unpad out_1
        # out_1 = out_1[0,:]
        # out_1 = out_1.unsqueeze(0)
        # 根据input_shape推断输出形状并重塑
        # output_shape = (batch_size, seq_len, out_features)
        
        # 计算最终输出：out_final = out_0 - out_1
        out_final = out_0 - out_1  # [b * s, out_features]

        out_final = out_final.view(output_shape)  # [b, s, out_features]
        
        return out_final

    @staticmethod
    def gemm_operation_with_mean_concat_quant(quant_result: MeanQuantResult, weight: torch.Tensor, 
                                       activation_dtype: torch.dtype = None, 
                                       quantizer: "Quantizer" = None):
        """
        基于mean_quant输出的GEMM操作：out_final = out_0 - out_1
        其中 out_0 = quant_input @ weight.T
              out_1 = quant_mean @ weight.T
        
        Args:
            quant_result: MeanConcatQuantResult对象，包含量化张量和形状信息
            weight: 权重矩阵，shape为[out_features, h]
            activation_dtype: 激活数据类型
            quantizer: 量化器
            
        Returns:
            out_final: 最终输出，根据input_shape和weight shape推断输出形状
        """
        # 从quant_result中提取数据
        quant_input = quant_result.quant_concat  # [b*s+1, h]
        input_shape = quant_result.shape  # (b, s, h)

        output_shape = list(input_shape)
        output_shape[-1] = weight.shape[0]

        # 从形状信息推断维度
        # batch_size = input_shape[0]
        # seq_len = input_shape[1]
        # out_features = weight.shape[0]
        
        # 使用TE的GEMM接口计算out_0: [b*s+1, h] @ [h, out_features]^T -> [b*s+1, out_features]
        output_all = MetisSvdFunction.svd_quant_gemm(
            weight, quant_input, activation_dtype, 
            quantizer, layout="TN", nvtx_label="input@weight"
        )  # [b*s+1, out_features]
        out_0 = output_all[:-1, :] # 除了最后一个 [b*s, out_features]
        out_1 = output_all[-1, :] # 最后一个 [1, out_features]
        # 根据input_shape推断输出形状并重塑
        # output_shape = (batch_size, seq_len, out_features)
        
        # 计算最终输出：out_final = out_0 - out_1
        out_final = out_0 - out_1  # [b, s, out_features]

        out_final = out_final.view(output_shape)  # [b, s, out_features]
        
        return out_final
    
    @staticmethod
    def compute_input_gradient_mean(grad_output_result: Union[MeanQuantResult, MeanConcatQuantResult], weight: torch.Tensor, 
                              activation_dtype: torch.dtype = None, 
                              quantizer: "Quantizer" = None):
        """
        计算输入梯度：dx = dout @ weight
        
        Args:
            dout: 输出梯度，shape为[b, s, out_features]
            weight: 权重矩阵，shape为[out_features, h]
            activation_dtype: 激活数据类型
            quantizer: 量化器
            
        Returns:
            dx: 输入梯度，shape为[b, s, h]
        """
        if isinstance(grad_output_result, MeanConcatQuantResult):
            quant_input = grad_output_result.quant_concat  # [b*s+1, out_features]
            # Split into input and mean parts
            input_shape = list(grad_output_result.shape)
            hidden_size = weight.shape[1]
            input_shape[-1] = hidden_size
            
            # 计算输入梯度：dx = dout @ weight
            dx_all = MetisSvdFunction.svd_quant_gemm(
                quant_input, weight, activation_dtype, 
                quantizer, layout="NN", grad=True, nvtx_label="dout@weight"
            )  # [b*s+1, h]
            
            dx = dx_all[:-1, :]  # [b*s, h]
            dx_mean = dx_all[-1, :]  # [1, h]
            
            dx_final = dx - dx_mean  # [b*s, h]
            dx_final = dx_final.view(input_shape)  # [b, s, h]
            
        elif isinstance(grad_output_result, MeanQuantResult):
            quant_mean = grad_output_result.quant_mean
            quant_input = grad_output_result.quant_input

            input_shape = list(grad_output_result.shape)
            hidden_size = weight.shape[1]
            input_shape[-1] = hidden_size
            
            # 计算输入梯度：dx = dout @ weight
            dx = MetisSvdFunction.svd_quant_gemm(
                weight, quant_input, activation_dtype, 
                quantizer, layout="NN", grad=True, nvtx_label="dout@weight"
            )  # [b*s, h]

            dx_mean = MetisSvdFunction.svd_quant_gemm(
                weight,quant_mean, activation_dtype, 
                quantizer, layout="NN", grad=True, nvtx_label="dout@weight"
            )  # [1, h]
            # dx_mean = dx_mean[0,:]
            # dx_mean = dx_mean.unsqueeze(0)
            dx_final = dx - dx_mean  # [b, s, h]
            dx_final = dx_final.view(input_shape)

        return dx_final

    @staticmethod
    def compute_input_gradient_mean_concat(grad_output_mean_quant_result: torch.Tensor, weight: torch.Tensor, 
                              activation_dtype: torch.dtype = None, 
                              quantizer: "Quantizer" = None):
        """
        计算输入梯度：dx = dout @ weight
        
        Args:
            dout: 输出梯度，shape为[b, s, out_features]
            weight: 权重矩阵，shape为[out_features, h]
            activation_dtype: 激活数据类型
            quantizer: 量化器
            
        Returns:
            dx: 输入梯度，shape为[b, s, h]
        """
        quant_mean = grad_output_mean_quant_result.quant_mean
        quant_input = grad_output_mean_quant_result.quant_input
        input_shape = list(grad_output_mean_quant_result.shape)
        hidden_size = weight.shape[1]
        input_shape[-1] = hidden_size
        # 重塑为2D张量
        
        # 计算输入梯度：dx = dout @ weight
        dx = MetisSvdFunction.svd_quant_gemm(
            quant_input, weight, activation_dtype, 
            quantizer, layout="NN", grad=True, nvtx_label="dout@weight"
        )  # [b*s, h]

        dx_mean = MetisSvdFunction.svd_quant_gemm(
            quant_mean, weight, activation_dtype, 
            quantizer, layout="NN", grad=True, nvtx_label="dout@weight"
        )  # [b*s, h]
        
        dx_final = dx - dx_mean  # [b, s, h]
        
        dx_final = dx_final.view(input_shape)  # [b, s, h]

        return dx_final
    
    @staticmethod
    def compute_weight_gradient_mean(x_quant_input:torch.Tensor,x_quant_mean:torch.Tensor, dy_quant: MeanQuantResult,
                               activation_dtype: torch.dtype = None, 
                               quantizer: "Quantizer" = None):
        """
        计算权重梯度：
                    dw = dy.T @ x
                       = (dy - dy_mean).T @ (x - x_mean)
                       = (dy.T @ x) - (dy.T @ x_mean) - (dy_mean.T @ x) + (dy_mean.T @ x_mean)
        Args:           
            x_quant: 输入的MeanQuantResult对象，包含量化张量和形状信息
            dy_quant: 输出梯度的MeanQuantResult对象，包含量化张量和形状信息
            activation_dtype: 激活数据类型
            quantizer: 量化器
            
        Returns:
            dw: 权重梯度，shape为[out_features, h]
        """
        # 从x_quant中提取输入数据
        # x_quant_input = x_quant.quant_input  # [b*s, h]
        # x_quant_mean = x_quant.quant_mean    # [b, h]
        # x_shape = x_quant.shape  # (b, s, h)
        
        # 从dy_quant中提取梯度数据
        dy_quant_input = dy_quant.quant_input  # [b*s, out_features]
        dy_quant_mean = dy_quant.quant_mean    # [b, out_features]
        # dy_shape = dy_quant.shape  # (b, s, out_features)
        
        # 根据公式计算四项：
        # dw = (dy.T @ x) - (dy.T @ x_mean) - (dy_mean.T @ x) + (dy_mean.T @ x_mean)
        
        # 第一项：dy.T @ x
        # [b*s, out_features]^T @ [b*s, h] -> [out_features, h]
        dw_term1 = MetisSvdFunction.svd_quant_gemm(
            x_quant_input, dy_quant_input, activation_dtype or torch.float32, 
            quantizer, layout="NT", grad=True, nvtx_label="dy_T@x"
        )  # [out_features, h]
        
        # 第二项：dy.T @ x_mean

        # x_quant_mean = x_quant_mean._quantizer(x_quant_mean.dequantize()[0,:].unsqueeze(0).expand(x_quant_input.size()))
        # [b*s, out_features]^T @ [b, h] -> [out_features, h]
        dw_term2 = MetisSvdFunction.svd_quant_gemm(
            x_quant_mean, dy_quant_input, activation_dtype, 
            quantizer, layout="NT", grad=True, nvtx_label="dy_T@x_mean"
        )  # [out_features, h]

        # 第三项：dy_mean.T @ x
        # [b, out_features]^T @ [b*s, h] -> [out_features, h]
        # 需要扩展dy_mean到[b*s, out_features]
        # dy_mean_expanded = dy_quant_mean.repeat_interleave(seq_len, dim=0)  # [b*s, out_features]
        # dy_quant_mean = dy_quant_mean._quantizer(dy_quant_mean.dequantize()[0,:].unsqueeze(0).expand(dy_quant_input.size()))
        dw_term3 = MetisSvdFunction.svd_quant_gemm(
            x_quant_input, dy_quant_mean, activation_dtype, 
            quantizer, layout="NT", grad=True, nvtx_label="dy_mean_T@x"
        )  # [out_features, h]
        
        # 第四项：dy_mean.T @ x_mean
        # [b, out_features]^T @ [b, h] -> [out_features, h]
        dw_term4 = MetisSvdFunction.svd_quant_gemm(
            x_quant_mean, dy_quant_mean, activation_dtype, 
            quantizer, layout="NT", grad=True, nvtx_label="dy_mean_T@x_mean"
        )  # [out_features, h]        
        # 最终权重梯度：dw = term1 - term2 - term3 + term4
        dw = dw_term1 - dw_term2 - dw_term3 + dw_term4  # [out_features, h]

        return dw
    
    @staticmethod
    def compute_weight_gradient_with_mean_concat(x_quant: MeanConcatQuantResult, dy_quant: MeanConcatQuantResult,
                                                       activation_dtype: torch.dtype = None, 
                                                       quantizer: "Quantizer" = None):
        """
        基于mean_concat_quant计算权重梯度
        
            dw = dy.T @ x
                = ([dy,dy_mean]).T @ (x - x_mean)
                = (dy.T @ x) - (dy.T @ x_mean) - (dy_mean.T @ x) + (dy_mean.T @ x_mean)

            let dy_expand_0 = [dy, dy_mean]
                dy_expand_1 = [dy, -dy_mean]
            let x_expand_0 = [x, dx_mean]
                x_expand_1 = [-x_mean, x]
            
            dw_0 = dy_expand_0.T @ x_expand_0
                 = dy.T @ dx + dy_mean.T @ x_mean
            dw_1 = dy_expand_1.T @ x_expand_1
                 = -dy_mean.T @ x - dy.T @ x_mean

            dw = dw_0 + dw_1
        Returns:
            dw: 权重梯度，shape为[out_features, h]
        """
        raise NotImplementedError
