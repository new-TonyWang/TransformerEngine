// CUTLASS GEMM实现
// 移植自 cutlass/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/tensor_view_io.h"


#include "../common.h"

using namespace cute;

// ---------------------------------------------------------------------------
// Scale factor reformat kernel: TE row-major -> CUTLASS interleaved layout
// ---------------------------------------------------------------------------
// TE stores scale factors in row-major (num_blocks_m, num_blocks_k) format
// CUTLASS SfAtom expects interleaved layout with Shape((32,4), (1,4)) Stride((16,4), (0,1))
// Mapping: CUTLASS_idx = (row % 32) * 16 + (row / 32) * 4 + col
__global__ void reformat_sf_te_to_cutlass_kernel(
    const uint8_t* __restrict__ te_sf,     // TE row-major scale factors
    uint8_t* __restrict__ cutlass_sf,      // CUTLASS interleaved scale factors  
    int num_rows,                          // Number of scale factor rows (M/16 or N/16)
    int num_cols                           // Number of scale factor columns (K/16)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_rows * num_cols;
    if (idx >= total) return;
    
    int row = idx / num_cols;
    int col = idx % num_cols;
    
    // TE layout: row-major
    int te_idx = row * num_cols + col;
    
    // CUTLASS interleaved layout: (row % 32) * 16 + (row / 32) * 4 + col
    // This matches SfAtom Shape((32,4), (1,4)) Stride((16,4), (0,1))
    int cutlass_idx = (row % 32) * 16 + (row / 32) * 4 + col;
    
    cutlass_sf[cutlass_idx] = te_sf[te_idx];
}

void reformat_sf_te_to_cutlass(
    const void* te_sf,
    void* cutlass_sf,
    int num_rows,
    int num_cols,
    cudaStream_t stream
) {
    int total = num_rows * num_cols;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;
    reformat_sf_te_to_cutlass_kernel<<<num_blocks, block_size, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(te_sf),
        reinterpret_cast<uint8_t*>(cutlass_sf),
        num_rows,
        num_cols
    );
    NVTE_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// NVFP4 BF16 GEMM 类型配置（对齐 79a 示例）
// ---------------------------------------------------------------------------

// A 矩阵：NVFP4，行主序，对齐32元素
using ElemA_fp4    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag   = cutlass::layout::RowMajor;
constexpr int AlignA = 32;

// B 矩阵：NVFP4，列主序，对齐32元素
using ElemB_fp4    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag   = cutlass::layout::ColumnMajor;
constexpr int AlignB = 32;

// C/D 矩阵：BF16，行主序
using ElemC_bf16   = cutlass::bfloat16_t;
using ElemD_bf16   = cutlass::bfloat16_t;
using LayoutCTag   = cutlass::layout::RowMajor;
using LayoutDTag   = cutlass::layout::RowMajor;
constexpr int AlignD = 128 / cutlass::sizeof_bits<ElemD_bf16>::value;
constexpr int AlignC = 128 / cutlass::sizeof_bits<ElemC_bf16>::value;

// 累加器类型、架构标签、算子类标签
using ElemAcc      = float;
using ArchTag      = cutlass::arch::Sm120;
using OpClass      = cutlass::arch::OpClassBlockScaledTensorOp;

// 线程块 Tile 形状与集群形状（GeForce RTX 50 系列不支持多播，ClusterShape = 1x1x1）
using TileShape    = Shape<_128, _128, _128>;
using CluShape     = Shape<_1, _1, _1>;

// Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass,
    TileShape, CluShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElemAcc, ElemAcc,
    ElemC_bf16, LayoutCTag, AlignC,
    ElemD_bf16, LayoutDTag, AlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

// Mainloop
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElemA_fp4, LayoutATag, AlignA,
    ElemB_fp4, LayoutBTag, AlignB,
    ElemAcc,
    TileShape, CluShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

// GemmKernel & Gemm 设备适配器
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Stride / Layout 辅助类型（从 GemmKernel 萃取）
using StrideA_t   = typename GemmKernel::StrideA;
using StrideB_t   = typename GemmKernel::StrideB;
using StrideC_t   = typename GemmKernel::StrideC;
using StrideD_t   = typename GemmKernel::StrideD;
using LayoutSFA_t = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB_t = typename GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ---------------------------------------------------------------------------
// nvfp4_bf16_cutlass_tensor_gemm
// ---------------------------------------------------------------------------
extern "C" void nvfp4_bf16_cutlass_tensor_gemm(const NVTETensor A, const NVTETensor B,
                            const NVTETensor C, NVTETensor D,
                            const NVTETensor bias, bool transa, bool transb, bool grad,
                            bool accumulate, bool use_split_accumulator, cudaStream_t stream) {
  NVTE_API_CALL(nvfp4_bf16_cutlass_tensor_gemm);

  // -----------------------------------------------------------------------
  // 1. 获取 TE Tensor 指针
  // -----------------------------------------------------------------------
  const transformer_engine::Tensor *A_tensor = transformer_engine::convertNVTETensorCheck(A);
  const transformer_engine::Tensor *B_tensor = transformer_engine::convertNVTETensorCheck(B);
  const transformer_engine::Tensor *C_tensor =
      (accumulate && C != nullptr) ? transformer_engine::convertNVTETensorCheck(C) : nullptr;
  transformer_engine::Tensor       *D_tensor = transformer_engine::convertNVTETensorCheck(D);

  // -----------------------------------------------------------------------
  // 2. Determine which data pointers to use based on transpose flags
  // -----------------------------------------------------------------------
  const bool is_A_transposed = transa;
  const bool is_B_transposed = transb;

  // Validate data availability
  // Note: gemm.cpp swaps A↔B before calling this function, so:
  //   CUTLASS A = original B, CUTLASS transa = original transb
  //   CUTLASS B = original A, CUTLASS transb = original transa
  // cuBLAS NVFP4 logic:
  //   Original A (weight): if transa, rowwise; else columnwise
  //   Original B (activation): if transb, columnwise; else rowwise
  // After swap:
  //   CUTLASS A (= orig B): if is_A_transposed (orig transb), columnwise; else rowwise
  //   CUTLASS B (= orig A): if is_B_transposed (orig transa), rowwise; else columnwise
  if (is_A_transposed) {
    NVTE_CHECK(A_tensor->has_columnwise_data(),
               "Input A is missing column-wise data for transposed access");
  } else {
    NVTE_CHECK(A_tensor->has_data(), "Input A is missing row-wise data for non-transposed access");
  }
  if (is_B_transposed) {
    // cuBLAS A uses rowwise when transposed
    NVTE_CHECK(B_tensor->has_data(),
               "Input B is missing row-wise data for transposed access");
  } else {
    // cuBLAS A uses columnwise when not transposed
    NVTE_CHECK(B_tensor->has_columnwise_data(),
               "Input B is missing column-wise data for non-transposed access");
  }

  // -----------------------------------------------------------------------
  // 3. Extract matrix dimensions M, N, K
  //    After the swap in gemm.cpp:
  //      CUTLASS A_tensor = original B, transa = original transb
  //      CUTLASS B_tensor = original A, transb = original transa
  //    cuBLAS (column-major) computes: (m_cub, n_cub) = (128, 64)
  //    Python expects row-major: (n_cub, m_cub) = (64, 128)
  //    CUTLASS row-major: M=n_cub, N=m_cub
  //    Derivation:
  //      n_cub = transb_orig ? B_orig1 : B_orig0 = transa ? A1 : A0
  //      m_cub = transa_orig ? A_orig0 : A_orig1 = transb ? B0 : B1
  //      k_cub = transa_orig ? A_orig1 : A_orig0 = transb ? B1 : B0
  // -----------------------------------------------------------------------
  const int A0 = A_tensor->flat_first_dim();
  const int A1 = A_tensor->flat_last_dim();
  const int B0 = B_tensor->flat_first_dim();
  const int B1 = B_tensor->flat_last_dim();

  // Dimensions for CUTLASS row-major output matching Python expectations
  const int M = transa ? A1 : A0;  // = n_cublas (Python rows)
  const int N = transb ? B0 : B1;  // = m_cublas (Python cols)
  const int K = transb ? B1 : B0;  // = k_cublas

  // -----------------------------------------------------------------------
  // 4. Compute alpha correction for NVFP4 scaling
  // -----------------------------------------------------------------------
  // NVFP4 scale factors are stored normalized to E4M3 max (448).
  // The actual dequantization is: dequant = e2m1_value × scale_inv_byte × (amax / (6 × 448))
  // CUTLASS computes: result = Σ (e2m1_A × sf_A × e2m1_B × sf_B)
  // To get correct result, we need: alpha = amax_A × amax_B / (6 × 6 × 448 × 448)
  // This matches the cuBLAS approach in nvte_nvfp4_compute_per_tensor_scale.
  constexpr float NVFP4_SCALE_FACTOR = 6.0f * 6.0f * 448.0f * 448.0f;  // = 7,257,600
  
  // Get amax pointers based on transpose flags (following cuBLAS convention)
  // After the swap in gemm.cpp:
  //   CUTLASS A = original B, CUTLASS transa = original transb
  //   CUTLASS B = original A, CUTLASS transb = original transa
  // cuBLAS uses: rowwise amax if transa (for A), rowwise amax if !transb (for B)
  // After swap:
  //   CUTLASS A (= orig B): rowwise if !is_A_transposed (i.e., !orig_transb)
  //   CUTLASS B (= orig A): rowwise if is_B_transposed (i.e., orig_transa)
  const void *amax_A_ptr = (!is_A_transposed) ? A_tensor->amax.dptr : A_tensor->columnwise_amax.dptr;
  const void *amax_B_ptr = is_B_transposed ? B_tensor->amax.dptr : B_tensor->columnwise_amax.dptr;
  
  // Fallback to available amax if preferred one is null
  if (amax_A_ptr == nullptr) amax_A_ptr = A_tensor->amax.dptr;
  if (amax_A_ptr == nullptr) amax_A_ptr = A_tensor->columnwise_amax.dptr;
  if (amax_B_ptr == nullptr) amax_B_ptr = B_tensor->amax.dptr;
  if (amax_B_ptr == nullptr) amax_B_ptr = B_tensor->columnwise_amax.dptr;
  
  NVTE_CHECK(amax_A_ptr != nullptr, "A tensor has no amax available");
  NVTE_CHECK(amax_B_ptr != nullptr, "B tensor has no amax available");
  
  // Read amax values from device to host
  float amax_A_host = 0.0f, amax_B_host = 0.0f;
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&amax_A_host, amax_A_ptr, sizeof(float), 
                                   cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(&amax_B_host, amax_B_ptr, sizeof(float), 
                                   cudaMemcpyDeviceToHost, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));  // Need sync to use amax on host
  
  // Compute corrected alpha
  const float alpha = (amax_A_host * amax_B_host) / NVFP4_SCALE_FACTOR;
  const float beta  = accumulate ? 1.0f : 0.0f;

  // -----------------------------------------------------------------------
  // 5. 构建 stride
  // -----------------------------------------------------------------------
  StrideA_t stride_A = cutlass::make_cute_packed_stride(StrideA_t{}, {M, K, 1});
  StrideB_t stride_B = cutlass::make_cute_packed_stride(StrideB_t{}, {N, K, 1});
  StrideC_t stride_C = cutlass::make_cute_packed_stride(StrideC_t{}, {M, N, 1});
  StrideD_t stride_D = cutlass::make_cute_packed_stride(StrideD_t{}, {M, N, 1});

  // -----------------------------------------------------------------------
  // 6. 构建块缩放因子的 interleaved Layout
  // -----------------------------------------------------------------------
  LayoutSFA_t layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
      cute::make_shape(M, N, K, 1));
  LayoutSFB_t layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      cute::make_shape(M, N, K, 1));

  // Compute scale factor dimensions
  // NVFP4 uses per-row, per-16-column block scaling
  // TE stores scale_inv as (padded_rows, K/16) - TE pads to 128-row blocks
  // We need to use the actual tensor scale_inv shapes, not GEMM dimensions
  constexpr int SF_BLOCK_SIZE = 16;  // 16 columns per scale factor
  
  // Get actual scale_inv dimensions from tensors
  // CUTLASS A (M×K) uses A_tensor's SF, CUTLASS B (N×K) uses B_tensor's SF
  const int sf_rows_A = is_A_transposed 
                        ? A_tensor->columnwise_scale_inv.shape[0]  // columnwise for transposed
                        : A_tensor->scale_inv.shape[0];            // rowwise for non-transposed
  const int sf_cols_A = (K + SF_BLOCK_SIZE - 1) / SF_BLOCK_SIZE;
  const int sf_rows_B = is_B_transposed
                        ? B_tensor->scale_inv.shape[0]             // rowwise for transposed
                        : B_tensor->columnwise_scale_inv.shape[0]; // columnwise for non-transposed
  const int sf_cols_B = (K + SF_BLOCK_SIZE - 1) / SF_BLOCK_SIZE;
  
  auto sfa_size = size(filter_zeros(layout_SFA));
  auto sfb_size = size(filter_zeros(layout_SFB));

  // -----------------------------------------------------------------------
  // 6b. Allocate workspace and reformat scale factors to CUTLASS layout
  // -----------------------------------------------------------------------
  // CUTLASS's interleaved layout requires more space due to striding
  // Allocate enough for the full interleaved layout
  size_t sf_workspace_A = (size_t)sfa_size * sizeof(uint8_t);
  size_t sf_workspace_B = (size_t)sfb_size * sizeof(uint8_t);
  
  cutlass::device_memory::allocation<uint8_t> cutlass_sf_A(sf_workspace_A);
  cutlass::device_memory::allocation<uint8_t> cutlass_sf_B(sf_workspace_B);
  
  // Initialize to zero (important for unused padding slots)
  NVTE_CHECK_CUDA(cudaMemsetAsync(cutlass_sf_A.get(), 0, sf_workspace_A, stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(cutlass_sf_B.get(), 0, sf_workspace_B, stream));

  // -----------------------------------------------------------------------
  // 7. Get data pointers
  //    Following cuBLAS convention for NVFP4:
  //    - A (M×K): use columnwise if transposed, rowwise if not
  //    - B (N×K): use rowwise if transposed, columnwise if not
  // -----------------------------------------------------------------------
  const void *A_data_ptr = is_A_transposed
                               ? A_tensor->columnwise_data.dptr
                               : A_tensor->data.dptr;
  const void *te_sf_A_ptr = is_A_transposed
                               ? A_tensor->columnwise_scale_inv.dptr
                               : A_tensor->scale_inv.dptr;
  const void *B_data_ptr = is_B_transposed
                               ? B_tensor->data.dptr
                               : B_tensor->columnwise_data.dptr;
  const void *te_sf_B_ptr = is_B_transposed
                               ? B_tensor->scale_inv.dptr
                               : B_tensor->columnwise_scale_inv.dptr;
  
  // Reformat scale factors from TE's row-major layout to CUTLASS's interleaved layout
  reformat_sf_te_to_cutlass(te_sf_A_ptr, cutlass_sf_A.get(), sf_rows_A, sf_cols_A, stream);
  reformat_sf_te_to_cutlass(te_sf_B_ptr, cutlass_sf_B.get(), sf_rows_B, sf_cols_B, stream);
  
  // Use reformatted scale factor pointers
  const void *A_sf_ptr = cutlass_sf_A.get();
  const void *B_sf_ptr = cutlass_sf_B.get();

  // C/D pointers
  const void *C_data_ptr = (accumulate && C_tensor != nullptr)
                               ? C_tensor->data.dptr
                               : nullptr;
  void       *D_data_ptr = D_tensor->data.dptr;

  NVTE_CHECK(A_data_ptr != nullptr, "A data pointer is null");
  NVTE_CHECK(te_sf_A_ptr != nullptr, "A scale_inv pointer is null");
  NVTE_CHECK(B_data_ptr != nullptr, "B data pointer is null");
  NVTE_CHECK(te_sf_B_ptr != nullptr, "B scale_inv pointer is null");
  NVTE_CHECK(D_data_ptr != nullptr, "D data pointer is null");
  if (accumulate) {
    NVTE_CHECK(C_data_ptr != nullptr, "C data pointer is null when accumulate=true");
  }

  // -----------------------------------------------------------------------
  // 8. Build Gemm::Arguments
  //    After the swap in gemm.cpp and with correct dimension calculation,
  //    the data pointers map directly without additional swapping:
  //      CUTLASS A (M×K) ← A_tensor data
  //      CUTLASS B (N×K) ← B_tensor data
  // -----------------------------------------------------------------------
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      reinterpret_cast<const ElemA_fp4::DataType *>(A_data_ptr),
      stride_A,
      reinterpret_cast<const ElemB_fp4::DataType *>(B_data_ptr),
      stride_B,
      reinterpret_cast<const ElemA_fp4::ScaleFactorType *>(A_sf_ptr),
      layout_SFA,
      reinterpret_cast<const ElemB_fp4::ScaleFactorType *>(B_sf_ptr),
      layout_SFB
    },
    {
      {alpha, beta},
      beta >=0 ? reinterpret_cast<const ElemC_bf16 *>(C_data_ptr): nullptr, stride_C,
      reinterpret_cast<ElemD_bf16 *>(D_data_ptr),       stride_D
    }
  };

  // -----------------------------------------------------------------------
  // 9. 创建 GEMM 实例并查询工作空间
  // -----------------------------------------------------------------------
  Gemm gemm_op;

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // -----------------------------------------------------------------------
  // 10. 检查 GEMM 可实现性
  // -----------------------------------------------------------------------
  cutlass::Status status = gemm_op.can_implement(arguments);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 GEMM cannot implement this problem: ",
             cutlassGetStatusString(status));

  // -----------------------------------------------------------------------
  // 11. 初始化 GEMM（传入 stream）
  // -----------------------------------------------------------------------
  status = gemm_op.initialize(arguments, workspace.get(), stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 GEMM initialize failed: ",
             cutlassGetStatusString(status));

  // -----------------------------------------------------------------------
  // 12. 执行 GEMM，结果写入 D
  // -----------------------------------------------------------------------
  status = gemm_op.run(stream);
  NVTE_CHECK(status == cutlass::Status::kSuccess,
             "CUTLASS NVFP4 GEMM run failed: ",
             cutlassGetStatusString(status));
}
