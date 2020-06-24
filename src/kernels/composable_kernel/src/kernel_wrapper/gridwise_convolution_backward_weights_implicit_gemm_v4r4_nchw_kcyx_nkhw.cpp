
#include "common_header.hpp"
#include "gridwise_convolution_backward_weight_implicit_gemm_v4r4_mlir.hpp"
#include "float_types.h"

extern "C" __global__

    __launch_bounds__(CK_PARAM_TUNABLE_BLOCK_SIZE, 2) void gridwise_convolution_backward_weights_implicit_gemm_v4r4_nchw_kcyx_nkhw 
        (const FLOAT* const __restrict__ p_in_global,
        const FLOAT* const __restrict__ p_out_global,
        FLOAT* const __restrict__ p_wei_global)
{
    using namespace ck;

    constexpr index_t ConvStrideH = CK_PARAM_PROBLEM_CONV_STRIDE_H;
    constexpr index_t ConvStrideW = CK_PARAM_PROBLEM_CONV_STRIDE_W;

    constexpr index_t ConvDilationH = CK_PARAM_PROBLEM_CONV_DILATION_H;
    constexpr index_t ConvDilationW = CK_PARAM_PROBLEM_CONV_DILATION_W;

    constexpr index_t InLeftPadH = CK_PARAM_PROBLEM_IN_LEFT_PAD_H;
    constexpr index_t InLeftPadW = CK_PARAM_PROBLEM_IN_LEFT_PAD_W;

    constexpr index_t InRightPadH = CK_PARAM_PROBLEM_IN_RIGHT_PAD_H;
    constexpr index_t InRightPadW = CK_PARAM_PROBLEM_IN_RIGHT_PAD_W;

    constexpr index_t BlockSize = CK_PARAM_TUNABLE_BLOCK_SIZE;
    constexpr index_t GridSize  = CK_PARAM_DEPENDENT_GRID_SIZE;

    constexpr index_t GemmMPerBlock = CK_PARAM_TUNABLE_GEMM_M_PER_BLOCK;
    constexpr index_t GemmNPerBlock = CK_PARAM_TUNABLE_GEMM_N_PER_BLOCK;
    constexpr index_t GemmKPerBlock = CK_PARAM_TUNABLE_GEMM_K_PER_BLOCK;

    // k,c,y,x
    constexpr index_t k = CK_PARAM_PROBLEM_K;
    constexpr index_t c = CK_PARAM_PROBLEM_C;
    constexpr index_t y = CK_PARAM_PROBLEM_Y;
    constexpr index_t x = CK_PARAM_PROBLEM_X;

    constexpr index_t stride_x = 1;
    constexpr index_t stride_y = x * stride_x;
    constexpr index_t stride_c = y * stride_y;
    constexpr index_t stride_k = c * stride_c;
    constexpr auto weight_k_c_y_x_desc = make_native_tensor_descriptor(Sequence<k, c, y, x>{}, Sequence<stride_k, stride_c, stride_y, stride_x>{});

    // ni,ci,hi,wi
    constexpr index_t ni = CK_PARAM_PROBLEM_N;
    constexpr index_t ci = CK_PARAM_PROBLEM_C;
    constexpr index_t hi = CK_PARAM_PROBLEM_HI;
    constexpr index_t wi = CK_PARAM_PROBLEM_WI;

    constexpr index_t stride_wi = 1;
    constexpr index_t stride_hi = wi * stride_wi;
    constexpr index_t stride_ci = hi * stride_hi;
    constexpr index_t stride_ni = ci * stride_ci;
    constexpr auto input_ni_ci_hi_wi_desc = make_native_tensor_descriptor(Sequence<ni, ci, hi, wi>{}, Sequence<stride_ni, stride_ci, stride_hi, stride_wi>{});

    // no,ko,ho,wo
    constexpr index_t no = CK_PARAM_PROBLEM_N;
    constexpr index_t ko = CK_PARAM_PROBLEM_K;
    constexpr index_t ho = CK_PARAM_PROBLEM_HO;
    constexpr index_t wo = CK_PARAM_PROBLEM_WO;

    constexpr index_t stride_wo = 1;
    constexpr index_t stride_ho = wo * stride_wo;
    constexpr index_t stride_ko = ho * stride_ho;
    constexpr index_t stride_no = ko * stride_ko;
    constexpr auto output_no_ko_ho_wo_desc = make_native_tensor_descriptor(Sequence<no, ko, ho, wo>{}, Sequence<stride_no, stride_ko, stride_ho, stride_wo>{});


    using ConvStrides   = Sequence<ConvStrideH, ConvStrideW>;
    using ConvDilations = Sequence<ConvDilationH, ConvDilationW>;

    using InLeftPads  = Sequence<InLeftPadH, InLeftPadW>;
    using InRightPads = Sequence<InRightPadH, InRightPadW>;

    // read and calculate tuning parameter
    constexpr index_t GemmMPerThreadSubC = CK_PARAM_TUNABLE_GEMM_M_PER_THREAD;
    constexpr index_t GemmNPerThreadSubC = CK_PARAM_TUNABLE_GEMM_N_PER_THREAD;
    constexpr index_t GemmMLevel0Cluster = CK_PARAM_TUNABLE_GEMM_M_LEVEL0_CLUSTER;
    constexpr index_t GemmNLevel0Cluster = CK_PARAM_TUNABLE_GEMM_N_LEVEL0_CLUSTER;
    constexpr index_t GemmMLevel1Cluster = CK_PARAM_TUNABLE_GEMM_M_LEVEL1_CLUSTER;
    constexpr index_t GemmNLevel1Cluster = CK_PARAM_TUNABLE_GEMM_N_LEVEL1_CLUSTER;
    constexpr index_t GemmKPerThreadLoop = 1;

    constexpr index_t GemmThreadGemmDataPerReadM = GemmMPerThreadSubC;
    constexpr index_t GemmThreadGemmDataPerReadN = GemmNPerThreadSubC;

    // A matrix
    constexpr index_t GemmABlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;

    constexpr index_t GemmABlockCopyClusterLengths_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_M;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmABlockCopyClusterLengths_GemmK;

    constexpr index_t GemmABlockCopyThreadSliceLengths_GemmM =
        GemmMPerBlock / GemmABlockCopyClusterLengths_GemmM;

    using GemmABlockCopyThreadSliceLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyThreadSliceLengths_GemmK, GemmABlockCopyThreadSliceLengths_GemmM>;

    using GemmABlockCopyThreadClusterLengths_GemmK_GemmM =
        Sequence<GemmABlockCopyClusterLengths_GemmK, GemmABlockCopyClusterLengths_GemmM>;

    constexpr index_t GemmABlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;

    constexpr index_t GemmABlockCopyDstDataPerWrite_GemmM =
        CK_PARAM_TUNABLE_GEMM_A_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_M;

    // B matrix
    constexpr index_t GemmBBlockCopyClusterLengths_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_K;

    constexpr index_t GemmBBlockCopyClusterLengths_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_CLUSTER_LENGTHS_GEMM_N;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmK =
        GemmKPerBlock / GemmBBlockCopyClusterLengths_GemmK;

    constexpr index_t GemmBBlockCopyThreadSliceLengths_GemmN =
        GemmNPerBlock / GemmBBlockCopyClusterLengths_GemmN;

    using GemmBBlockCopyThreadSliceLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyThreadSliceLengths_GemmK, GemmBBlockCopyThreadSliceLengths_GemmN>;

    using GemmBBlockCopyThreadClusterLengths_GemmK_GemmN =
        Sequence<GemmBBlockCopyClusterLengths_GemmK, GemmBBlockCopyClusterLengths_GemmN>;

    constexpr index_t GemmBBlockCopySrcDataPerRead_GemmK =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_SRC_DATA_PER_READ_GEMM_K;

    constexpr index_t GemmBBlockCopyDstDataPerWrite_GemmN =
        CK_PARAM_TUNABLE_GEMM_B_BLOCK_COPY_DST_DATA_PER_WRITE_GEMM_N;

    // C matrix
    constexpr index_t GemmCThreadCopyDstDataPerWrite_GemmN1 =
        CK_PARAM_TUNABLE_GEMM_C_THREAD_COPY_DST_DATA_PER_WRITE_GEMM_N1;

    constexpr auto gridwise_conv = GridwiseConvolutionBackwardWeightImplicitGemm_v4r4_mlir
        <GridSize,
        BlockSize,
        FLOAT,
        FLOAT_ACCUM,
        decltype(input_ni_ci_hi_wi_desc),
        decltype(weight_k_c_y_x_desc),
        decltype(output_no_ko_ho_wo_desc),
        ConvStrides,
        ConvDilations,
        InLeftPads,
        InRightPads,
        GemmMPerBlock,
        GemmNPerBlock,
        GemmKPerBlock,
        GemmMPerThreadSubC,
        GemmNPerThreadSubC,
        GemmKPerThreadLoop,
        GemmMLevel0Cluster,
        GemmNLevel0Cluster,
        GemmMLevel1Cluster,
        GemmNLevel1Cluster,
        GemmThreadGemmDataPerReadM,
        GemmThreadGemmDataPerReadN,
        GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
        GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
        GemmABlockCopySrcDataPerRead_GemmK,
        GemmABlockCopyDstDataPerWrite_GemmM,
        GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
        GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
        GemmBBlockCopySrcDataPerRead_GemmK,
        GemmBBlockCopyDstDataPerWrite_GemmN,
        GemmCThreadCopyDstDataPerWrite_GemmN1>{};

    gridwise_conv.Run(p_in_global, p_wei_global, p_out_global);
}
