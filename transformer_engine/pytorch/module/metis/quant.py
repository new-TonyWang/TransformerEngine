from dataclasses import dataclass
import enum
from typing import Optional, List
import torch

from transformer_engine.pytorch.cpp_extensions import (
    general_gemm,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop
from transformer_engine.pytorch.constants import TE_DType
import transformer_engine.pytorch as te
import transformer_engine_torch as tex

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

def process_and_fill_matrix(x, drop_rate = 0.8):
    """
    输入:
        x: 原始矩阵 (B*S, H)
        drop_rate : 需要删除的 token 比例
    输出:
        x_compact: RestoreInfo
    """
    
    B_S, _ = x.shape
    drop_count = int(B_S*drop_rate)
    drop_count = (drop_count // 8 ) * 8 # 保证 matmul k 轴能被 8 整除
    keep_count = B_S - drop_count
    
    if keep_count <= 0:
        raise ValueError("删除数量过多，保留数量必须大于0")

    # ==========================================
    # 获取 保留下标 (Keep) 和 删除下标 (Drop)
    # ==========================================
    
    # 使用噪声矩阵选取token
    noise = torch.rand(B_S, device=x.device)
    
    # A. 获取保留的下标 (噪声值最大的前 keep_count 个)
    _, keep_indices = torch.topk(noise, k=keep_count, dim=0, largest=True)
    keep_indices, _ = torch.sort(keep_indices, dim=0) # 保持时序
    
    # B. 获取被删除的下标 (噪声值最小的前 drop_count 个) -> 补集
    _, drop_indices = torch.topk(noise, k=drop_count, dim=0, largest=False)
    drop_indices, _ = torch.sort(drop_indices, dim=0) # 保持时序
    
    # 提取保留的数据
    x_compact = x.index_select(0,keep_indices)
    return x_compact, RestoreInfo(x.shape, keep_indices, drop_indices ,drop_count)

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
    @torch.compile
    def svd_lowrank_quant(
        input_: torch.Tensor,
        input_quantizer: "Quantizer" = None,
        rank=60,
        niter=2,
        broadcast_dim=-1,
        is_backward=False,
        gradacc_broadcast=False,
        load_history=False,
        history_list=List[Optional[torch.Tensor]],
    ):

        # for backward, input_ has already shaped into 2d tensor.
        # input_ shape [b,s,h]

        input_shape = input_.shape
        if broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)  # [s,h]
        else:
            cinput = input_
        original_shape = cinput.shape  # [s,h]
        if load_history and gradacc_broadcast and is_backward:
            ker, de_svd_gemm_out = history_list
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
            if gradacc_broadcast and is_backward:
                # print("storing history_list----")
                history_list.clear()
                history_list.extend([ker, de_svd_gemm_out])

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
    @torch.no_grad()
    @torch.compile
    def svd_lowrank_quant_separate_residual(
        input_: torch.Tensor,
        input_quantizer: "Quantizer",
        rank=64,
        niter=2,
        token_drop_rate: float = -1.0,
        broadcast_dim=-1,
        is_backward=False,
        gradacc_broadcast=False,
        load_history=False,
        keep_dim=False,
        history_list=List[Optional[torch.Tensor]],
        restore_strategy = "tile",
    ):

        # for backward, input_ has already shaped into 2d tensor.
        # input_ shape [b,s,h]

        input_shape = input_.shape
        should_reshape = False
        restore_info=None
        if len(input_shape) > 2:
            should_reshape = True
            input_ = input_.view(-1, input_.shape[-1])

        if token_drop_rate >=0:
            cinput,restore_info = process_and_fill_matrix(input_, token_drop_rate)
        elif broadcast_dim >= 0:
            cinput = input_.select(broadcast_dim, 0)  # [s,h]
                
        else:
            cinput = input_

        if load_history and gradacc_broadcast and is_backward:
            ker, ug_sg,vg = history_list
        else:
            # print(f"cinput shape==",cinput.shape)
            # ug, sg, vg = torch.svd(cinput.to(torch.float32)) # ug,vg are not contiguous, their transpose is contiguous.
            ug, sg, vg = torch.svd_lowrank(cinput.to(torch.float32), q=rank, niter=niter)
            # print("running svd")
            ug = ug.to(input_.dtype)
            sg = sg.to(input_.dtype)
            if len(sg.shape) == 1:
                sg = torch.diag(sg)
            vg = vg.to(input_.dtype)
            # print(f"cinput.shape={cinput.shape},input_.shape={input_.shape},sg.size={sg.size()}, ug.size={ug.size()}, vg.size={vg.size()}")
            ug_sg = ug @ sg
            ker = ug_sg @ vg.T  # [s,h] or [b*s,h]
            if broadcast_dim >= 0:
                ker = ker.unsqueeze(broadcast_dim)  # [1,s,h]
            else:
                pass
            if input_quantizer is None:
                ug_sg = MetisSvdFunction.svd_quant_gemm(
                sg, ug.T, input_.dtype, None, layout="NT", nvtx_label="U@S"
            )# we use layout = NT to tell te to transpose ug. ug.T cannot work because it does not change memory layout.
            else:
                # ugq = input_quantizer(ug)
                # sgq = input_quantizer(sg)
                # 仅 量化 vg，ug_sg 在 matmul 之前量化
                vgq = input_quantizer(vg)
                # ug_sgq = input_quantizer(ug_sg)   
                # ug_sg = ug_sgq
                vg = vgq

                if False:
                    ugq = input_quantizer(ug)
                    sgq = input_quantizer(sg)
                    ug_sgq = MetisSvdFunction.svd_quant_gemm(
                        sgq, ugq, input_.dtype, None, layout="NN", nvtx_label="U@S"
                    )
                    dequant_input = MetisSvdFunction.svd_quant_gemm(
                        vgq, ug_sgq, input_.dtype, None, layout="TN", nvtx_label="U@S"
                    )
            
            if keep_dim:
                ug_svd_shape = list(input_shape)
                ug_svd_shape[-1] = rank
                ug_sg = ug_sg.view(ug_svd_shape)

            if gradacc_broadcast and is_backward:
                # print("storing history_list----")
                history_list.clear()
                history_list.extend([ker, ug_sg, sg])

        if restore_info is not None:
            ker = restore_matrix(ker, restore_info, restore_strategy)
        if keep_dim and should_reshape:
            res = input_- ker
            res = res.view(input_shape)
        else:
            res = input_- ker
        
        if input_quantizer is not None:
            res = input_quantizer(res)

        return ug_sg,vg, res, restore_info

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
