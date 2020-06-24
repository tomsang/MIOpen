
#ifndef CK_GRIDWISE_CONVOLUTION_BACKWARD_WEIGHT_IMPLICIT_GEMM_V4R4_HPP
#define CK_GRIDWISE_CONVOLUTION_BACKWARD_WEIGHT_IMPLICIT_GEMM_V4R4_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm.hpp"

namespace ck {

// GemmM = K
// GemmN = C * Y * X
// GemmK = N * H * W
template <index_t GridSize,
          index_t BlockSize,
          typename Float,
          typename AccFloat,
          typename InGlobalDesc,
          typename WeiGlobalDesc,
          typename OutGlobalDesc,
          typename ConvStrides,
          typename ConvDilations,
          typename InLeftPads,
          typename InRightPads,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          index_t GemmKPerBlock,
          index_t GemmMPerThreadSubC,
          index_t GemmNPerThreadSubC,
          index_t GemmKPerThreadLoop,
          index_t GemmMLevel0Cluster,
          index_t GemmNLevel0Cluster,
          index_t GemmMLevel1Cluster,
          index_t GemmNLevel1Cluster,
          index_t GemmThreadGemmDataPerReadM,
          index_t GemmThreadGemmDataPerReadN,
          typename GemmABlockCopyThreadSliceLengths_GemmK_GemmM,
          typename GemmABlockCopyThreadClusterLengths_GemmK_GemmM,
          index_t GemmABlockCopySrcDataPerRead_GemmK,
          index_t GemmABlockCopyDstDataPerWrite_GemmM,
          typename GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
          typename GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
          index_t GemmBBlockCopySrcDataPerRead_GemmK,
          index_t GemmBBlockCopyDstDataPerWrite_GemmN,
          index_t GemmCThreadCopyDstDataPerWrite_GemmN1>
struct GridwiseConvolutionBackwardWeightImplicitGemm_v4r4_mlir
{
    __device__ void Run(const Float* const __restrict__ p_in_global,
                        Float* __restrict__ p_wei_global,
                        const Float* const __restrict__ p_out_global) const
    {

        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};

        constexpr index_t ConvStrideH = ConvStrides{}[0];
        constexpr index_t ConvStrideW = ConvStrides{}[1];

        constexpr index_t ConvDilationH = ConvDilations{}[0];
        constexpr index_t ConvDilationW = ConvDilations{}[1];

        constexpr auto weight_k_c_y_x_desc = WeiGlobalDesc{};
        constexpr auto input_ni_ci_hi_wi_desc = InGlobalDesc{};
        constexpr auto output_no_ko_ho_wo_desc = OutGlobalDesc{};

        // k, c, y, x
        constexpr index_t k = weight_k_c_y_x_desc.GetLengths()[0];
        constexpr index_t c = weight_k_c_y_x_desc.GetLengths()[1];
        constexpr index_t y = weight_k_c_y_x_desc.GetLengths()[2];
        constexpr index_t x = weight_k_c_y_x_desc.GetLengths()[3];

        // ni, ci, hi, wi
        constexpr index_t ni = input_ni_ci_hi_wi_desc.GetLengths()[0];
        constexpr index_t ci = input_ni_ci_hi_wi_desc.GetLengths()[1];
        constexpr index_t hi = input_ni_ci_hi_wi_desc.GetLengths()[2];
        constexpr index_t wi = input_ni_ci_hi_wi_desc.GetLengths()[3];

        // no, ko, ho, wo
        constexpr index_t no = output_no_ko_ho_wo_desc.GetLengths()[0];
        constexpr index_t ko = output_no_ko_ho_wo_desc.GetLengths()[1];
        constexpr index_t ho = output_no_ko_ho_wo_desc.GetLengths()[2];
        constexpr index_t wo = output_no_ko_ho_wo_desc.GetLengths()[3];

        constexpr auto weight_gemmM_gemmN_desc = transform_tensor_descriptor(
            weight_k_c_y_x_desc,
            make_tuple(PassThrough<k>{}, Merge<Sequence<c, y, x>>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto input_ni_ci_hipad_wipad_desc = transform_tensor_descriptor(
            input_ni_ci_hi_wi_desc,
            make_tuple(PassThrough<ni>{}, PassThrough<ci>{}, Pad<Sequence<hi, wi>, InLeftPads, InRightPads>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}));

        constexpr auto input_ni_ci_y_ho_x_wo_desc = transform_tensor_descriptor(
            input_ni_ci_hipad_wipad_desc,
            make_tuple(PassThrough<ni>{}, PassThrough<ci>{}, Embed<input_ni_ci_hipad_wipad_desc.GetLengths()[2], Sequence<y, ho>, Sequence<ConvDilationH, ConvStrideH, 0>>{}, Embed<input_ni_ci_hipad_wipad_desc.GetLengths()[3], Sequence<x, wo>, Sequence<ConvDilationW, ConvStrideW, 0>>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

        constexpr auto input_gemmK_gemmN_desc = transform_tensor_descriptor(
            input_ni_ci_y_ho_x_wo_desc,
            make_tuple(Merge<Sequence<ni, ho, wo>>{}, Merge<Sequence<ci, y, x>>{}),
            make_tuple(Sequence<0, 3, 5>{}, Sequence<1, 2, 4>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        constexpr auto output_gemmK_gemmM_desc = transform_tensor_descriptor(
            output_no_ko_ho_wo_desc,
            make_tuple(Merge<Sequence<no, ho, wo>>{}, PassThrough<ko>{}),
            make_tuple(Sequence<0, 2, 3>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));


        // GEM
        constexpr auto gridwise_gemm =
            GridwiseGemmTransposedANormalBNormalC_v1<GridSize,
                                                     BlockSize,
                                                     Float,
                                                     AccFloat,
                                                     decltype(output_gemmK_gemmM_desc),
                                                     decltype(input_gemmK_gemmN_desc),
                                                     decltype(weight_gemmM_gemmN_desc),
                                                     InMemoryDataOperation::Set,
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
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     GemmABlockCopySrcDataPerRead_GemmK,
                                                     GemmABlockCopyDstDataPerWrite_GemmM,
                                                     GemmBBlockCopyThreadSliceLengths_GemmK_GemmN,
                                                     GemmBBlockCopyThreadClusterLengths_GemmK_GemmN,
                                                     Sequence<1, 0>,
                                                     Sequence<1, 0>,
                                                     0,
                                                     GemmBBlockCopySrcDataPerRead_GemmK,
                                                     GemmBBlockCopyDstDataPerWrite_GemmN,
                                                     Sequence<0, 1, 2, 3>,
                                                     3,
                                                     GemmCThreadCopyDstDataPerWrite_GemmN1>{};

        gridwise_gemm.Run(p_out_global, p_in_global, p_wei_global);
    }
};

} // namespace ck
#endif
