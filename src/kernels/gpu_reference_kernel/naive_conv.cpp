/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
// #include "bfloat16_dev.hpp"

// hcc seems need __device__ __host__ together to compile, and no extern "C"
typedef union _cvt_bf16_fp32
{
    uint u32;
    ushort2 ushortx2;
    ushort ushortvec[2];
    float f32;
} _cvt_bf16_fp32_t;

__device__ __host__ float __bfloat16_to_float(ushort src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.ushortx2 = make_ushort2(0, src_val);
    return target_val.f32;
}

__device__ __host__ ushort __float_to_bfloat16(float src_val)
{
    _cvt_bf16_fp32_t target_val;
    target_val.f32 = src_val;

    if((~target_val.u32 & 0x7f800000) == 0) // Inf or NaN
    {
        if((target_val.u32 & 0xffff) != 0)
        {
            target_val.u32 |= 0x10000; // Preserve signaling NaN
        }
    }
    else
    {
#ifdef MIOPEN_USE_RNE_BFLOAT16
        target_val.u32 += (0x7fff + (target_val.ushortvec[1] & 1));
#endif // MIOPEN_USE_RNE_BFLOAT16
    }
    return target_val.ushortvec[1];
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_nchw_fp32(const float* __restrict__ p_in,
                                                    const float* __restrict__ p_wei,
                                                    float* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * ho * wo`.
     *  to distribute this workload, let one workgroup compute `ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += in * c * hi * wi + ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iho = tid / wo;
        int iwo = tid % wo;

        float value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        int i_idx = ic * hi * wi + cur_h * wi + cur_w;
                        int w_idx = ic * fy * fx + iy * fx + ix;
                        value += p_in[i_idx] * p_wei[w_idx];
                    }
                }
            }
        }
        int o_idx    = iho * wo + iwo;
        p_out[o_idx] = value;
    }
}

extern "C" __global__ void naive_conv_bwd_nchw_fp32(float* __restrict__ p_in,
                                                    const float* __restrict__ p_wei,
                                                    const float* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * hi * wi`.
     *  to distribute this workload, let one workgroup compute `hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += in * c * hi * wi + ig * c_per_group * hi * wi + ic * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ic * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ihi = tid / wi;
        int iwi = tid % wi;

        float value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                if(cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if(cur_ho >= ho)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if(cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if(cur_wo >= wo)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        int o_idx = ik * ho * wo + cur_ho * wo + cur_wo;
                        int f_idx = ik * c_per_group * fy * fx + iy * fx + ix;
                        value += p_out[o_idx] * p_wei[f_idx];
                    }
                }
            }
        }
        int i_idx   = ihi * wi + iwi;
        p_in[i_idx] = value;
    }
}

extern "C" __global__ void naive_conv_wrw_nchw_fp32(const float* __restrict__ p_in,
                                                    float* __restrict__ p_wei,
                                                    const float* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int ic = tid / (fx * fy);

        float value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int iho = 0; iho < ho; iho++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int iwo = 0; iwo < wo; iwo++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        int i_idx = in * c * hi * wi + ic * hi * wi + cur_h * wi + cur_w;
                        int o_idx = in * k * ho * wo + iho * wo + iwo;
                        value += p_in[i_idx] * p_out[o_idx];
                    }
                }
            }
        }
        int f_idx    = ic * fy * fx + iy * fx + ix;
        p_wei[f_idx] = value;
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ncdhw_fp32(const float* __restrict__ p_in,
                                                     const float* __restrict__ p_wei,
                                                     float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * do_ * ho * wo`.
     *  to distribute this workload, let one workgroup compute `do_ * ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
    p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        float value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_w & valid_h)
                        {
                            int i_idx = ic * di * hi * wi + cur_d * hi * wi + cur_h * wi + cur_w;
                            int w_idx = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                            value += p_in[i_idx] * p_wei[w_idx];
                        }
                    }
                }
            }
        }
        int o_idx    = ido * ho * wo + iho * wo + iwo;
        p_out[o_idx] = value;
    }
}

extern "C" __global__ void naive_conv_bwd_ncdhw_fp32(float* __restrict__ p_in,
                                                     const float* __restrict__ p_wei,
                                                     const float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi + ic * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ic * fz * fy * fx;
    p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        float value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                    if(cur_ho < 0 || cur_ho % sy)
                        valid_h &= 0;
                    cur_ho /= sy;
                    if(cur_ho >= ho)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                        if(cur_wo < 0 || cur_wo % sx)
                            valid_w &= 0;
                        cur_wo /= sx;
                        if(cur_wo >= wo)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            int o_idx =
                                ik * do_ * ho * wo + cur_do * ho * wo + cur_ho * wo + cur_wo;
                            int f_idx =
                                ik * c_per_group * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                            value += p_out[o_idx] * p_wei[f_idx];
                        }
                    }
                }
            }
        }
        int i_idx   = idi * hi * wi + ihi * wi + iwi;
        p_in[i_idx] = value;
    }
}

extern "C" __global__ void naive_conv_wrw_ncdhw_fp32(const float* __restrict__ p_in,
                                                     float* __restrict__ p_wei,
                                                     const float* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += ig * c_per_group * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
    p_out += ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        float value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iho = 0; iho < ho; iho++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int iwo = 0; iwo < wo; iwo++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            int i_idx = in * c * di * hi * wi + ic * di * hi * wi +
                                        cur_d * hi * wi + cur_h * wi + cur_w;
                            int o_idx = in * k * do_ * ho * wo + ido * ho * wo + iho * wo + iwo;
                            value += p_in[i_idx] * p_out[o_idx];
                        }
                    }
                }
            }
        }
        int f_idx    = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
        p_wei[f_idx] = value;
    }
}

extern "C" __global__ void naive_conv_fwd_nchw_fp16(const half* __restrict__ p_in,
                                                    const half* __restrict__ p_wei,
                                                    half* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * ho * wo`.
     *  to distribute this workload, let one workgroup compute `ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += in * c * hi * wi + ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iho = tid / wo;
        int iwo = tid % wo;

        float value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        int i_idx = ic * hi * wi + cur_h * wi + cur_w;
                        int w_idx = ic * fy * fx + iy * fx + ix;
                        value += __half2float(p_in[i_idx]) * __half2float(p_wei[w_idx]);
                    }
                }
            }
        }
        int o_idx    = iho * wo + iwo;
        p_out[o_idx] = __float2half(value);
    }
}

extern "C" __global__ void naive_conv_bwd_nchw_fp16(half* __restrict__ p_in,
                                                    const half* __restrict__ p_wei,
                                                    const half* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * hi * wi`.
     *  to distribute this workload, let one workgroup compute `hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += in * c * hi * wi + ig * c_per_group * hi * wi + ic * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ic * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ihi = tid / wi;
        int iwi = tid % wi;

        float value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                if(cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if(cur_ho >= ho)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if(cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if(cur_wo >= wo)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        int o_idx = ik * ho * wo + cur_ho * wo + cur_wo;
                        int f_idx = ik * c_per_group * fy * fx + iy * fx + ix;
                        value += __half2float(p_out[o_idx]) * __half2float(p_wei[f_idx]);
                    }
                }
            }
        }
        int i_idx   = ihi * wi + iwi;
        p_in[i_idx] = __float2half(value);
    }
}

extern "C" __global__ void naive_conv_wrw_nchw_fp16(const half* __restrict__ p_in,
                                                    half* __restrict__ p_wei,
                                                    const half* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int ic = tid / (fx * fy);

        float value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int iho = 0; iho < ho; iho++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int iwo = 0; iwo < wo; iwo++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        int i_idx = in * c * hi * wi + ic * hi * wi + cur_h * wi + cur_w;
                        int o_idx = in * k * ho * wo + iho * wo + iwo;
                        value += __half2float(p_in[i_idx]) * __half2float(p_out[o_idx]);
                    }
                }
            }
        }
        int f_idx    = ic * fy * fx + iy * fx + ix;
        p_wei[f_idx] = __float2half(value);
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ncdhw_fp16(const half* __restrict__ p_in,
                                                     const half* __restrict__ p_wei,
                                                     half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * do_ * ho * wo`.
     *  to distribute this workload, let one workgroup compute `do_ * ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
    p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        float value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_w & valid_h)
                        {
                            int i_idx = ic * di * hi * wi + cur_d * hi * wi + cur_h * wi + cur_w;
                            int w_idx = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                            value += __half2float(p_in[i_idx]) * __half2float(p_wei[w_idx]);
                        }
                    }
                }
            }
        }
        int o_idx    = ido * ho * wo + iho * wo + iwo;
        p_out[o_idx] = __float2half(value);
    }
}

extern "C" __global__ void naive_conv_bwd_ncdhw_fp16(half* __restrict__ p_in,
                                                     const half* __restrict__ p_wei,
                                                     const half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi + ic * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ic * fz * fy * fx;
    p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        float value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                    if(cur_ho < 0 || cur_ho % sy)
                        valid_h &= 0;
                    cur_ho /= sy;
                    if(cur_ho >= ho)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                        if(cur_wo < 0 || cur_wo % sx)
                            valid_w &= 0;
                        cur_wo /= sx;
                        if(cur_wo >= wo)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            int o_idx =
                                ik * do_ * ho * wo + cur_do * ho * wo + cur_ho * wo + cur_wo;
                            int f_idx =
                                ik * c_per_group * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                            value += __half2float(p_out[o_idx]) * __half2float(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        int i_idx   = idi * hi * wi + ihi * wi + iwi;
        p_in[i_idx] = __float2half(value);
    }
}

extern "C" __global__ void naive_conv_wrw_ncdhw_fp16(const half* __restrict__ p_in,
                                                     half* __restrict__ p_wei,
                                                     const half* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += ig * c_per_group * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
    p_out += ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        float value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iho = 0; iho < ho; iho++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int iwo = 0; iwo < wo; iwo++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            int i_idx = in * c * di * hi * wi + ic * di * hi * wi +
                                        cur_d * hi * wi + cur_h * wi + cur_w;
                            int o_idx = in * k * do_ * ho * wo + ido * ho * wo + iho * wo + iwo;
                            value += __half2float(p_in[i_idx]) * __half2float(p_out[o_idx]);
                        }
                    }
                }
            }
        }
        int f_idx    = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
        p_wei[f_idx] = __float2half(value);
    }
}

extern "C" __global__ void naive_conv_fwd_nchw_bf16(const ushort* __restrict__ p_in,
                                                    const ushort* __restrict__ p_wei,
                                                    ushort* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * ho * wo`.
     *  to distribute this workload, let one workgroup compute `ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += in * c * hi * wi + ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iho = tid / wo;
        int iwo = tid % wo;

        float value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        int i_idx = ic * hi * wi + cur_h * wi + cur_w;
                        int w_idx = ic * fy * fx + iy * fx + ix;
                        value +=
                            __bfloat16_to_float(p_in[i_idx]) * __bfloat16_to_float(p_wei[w_idx]);
                    }
                }
            }
        }
        int o_idx    = iho * wo + iwo;
        p_out[o_idx] = __float_to_bfloat16(value);
    }
}

extern "C" __global__ void naive_conv_bwd_nchw_bf16(ushort* __restrict__ p_in,
                                                    const ushort* __restrict__ p_wei,
                                                    const ushort* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * hi * wi`.
     *  to distribute this workload, let one workgroup compute `hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += in * c * hi * wi + ig * c_per_group * hi * wi + ic * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ic * fy * fx;
    p_out += in * k * ho * wo + ig * k_per_group * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ihi = tid / wi;
        int iwi = tid % wi;

        float value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iy = 0; iy < fy; iy++)
            {
                int valid_h = 1;
                int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                if(cur_ho < 0 || cur_ho % sy)
                    valid_h &= 0;
                cur_ho /= sy;
                if(cur_ho >= ho)
                    valid_h &= 0;
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                    if(cur_wo < 0 || cur_wo % sx)
                        valid_w &= 0;
                    cur_wo /= sx;
                    if(cur_wo >= wo)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        int o_idx = ik * ho * wo + cur_ho * wo + cur_wo;
                        int f_idx = ik * c_per_group * fy * fx + iy * fx + ix;
                        value +=
                            __bfloat16_to_float(p_out[o_idx]) * __bfloat16_to_float(p_wei[f_idx]);
                    }
                }
            }
        }
        int i_idx   = ihi * wi + iwi;
        p_in[i_idx] = __float_to_bfloat16(value);
    }
}

extern "C" __global__ void naive_conv_wrw_nchw_bf16(const ushort* __restrict__ p_in,
                                                    ushort* __restrict__ p_wei,
                                                    const ushort* __restrict__ p_out,
                                                    int hi,
                                                    int wi,
                                                    int n,
                                                    int k_per_group,
                                                    int c_per_group,
                                                    int ho,
                                                    int wo,
                                                    int sy,
                                                    int sx,
                                                    int dy,
                                                    int dx,
                                                    int py,
                                                    int px,
                                                    int fy,
                                                    int fx,
                                                    int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += ig * c_per_group * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx;
    p_out += ig * k_per_group * ho * wo + ik * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int ic = tid / (fx * fy);

        float value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int iho = 0; iho < ho; iho++)
            {
                int valid_h = 1;
                int cur_h   = sy * iho - py + dy * iy;
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                for(int iwo = 0; iwo < wo; iwo++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix;
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_h & valid_w)
                    {
                        int i_idx = in * c * hi * wi + ic * hi * wi + cur_h * wi + cur_w;
                        int o_idx = in * k * ho * wo + iho * wo + iwo;
                        value +=
                            __bfloat16_to_float(p_in[i_idx]) * __bfloat16_to_float(p_out[o_idx]);
                    }
                }
            }
        }
        int f_idx    = ic * fy * fx + iy * fx + ix;
        p_wei[f_idx] = __float_to_bfloat16(value);
    }
}

// design block_size 256
extern "C" __global__ void naive_conv_fwd_ncdhw_bf16(const ushort* __restrict__ p_in,
                                                     const ushort* __restrict__ p_wei,
                                                     ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * do_ * ho * wo`.
     *  to distribute this workload, let one workgroup compute `do_ * ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = do_ * ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int in            = (bid / k_per_group) % n;
    int ig            = bid / (n * k_per_group);

    p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
    p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwo = tid % wo;
        int iho = (tid / wo) % ho;
        int ido = tid / (ho * wo);

        float value = .0f;

        for(int ic = 0; ic < c_per_group; ic++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_w & valid_h)
                        {
                            int i_idx = ic * di * hi * wi + cur_d * hi * wi + cur_h * wi + cur_w;
                            int w_idx = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                            value += __bfloat16_to_float(p_in[i_idx]) *
                                     __bfloat16_to_float(p_wei[w_idx]);
                        }
                    }
                }
            }
        }
        int o_idx    = ido * ho * wo + iho * wo + iwo;
        p_out[o_idx] = __float_to_bfloat16(value);
    }
}

extern "C" __global__ void naive_conv_bwd_ncdhw_bf16(ushort* __restrict__ p_in,
                                                     const ushort* __restrict__ p_wei,
                                                     const ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total input pixel: `group * n * c_per_group * di * hi * wi`.
     *  to distribute this workload, let one workgroup compute `di * hi * wi` pixel,
     *  hence need `group * n * c_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = di * hi * wi;
    int bid           = blockIdx.x;
    int ic            = bid % c_per_group;
    int in            = (bid / c_per_group) % n;
    int ig            = bid / (n * c_per_group);

    p_in += in * c * di * hi * wi + ig * c_per_group * di * hi * wi + ic * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ic * fz * fy * fx;
    p_out += in * k * do_ * ho * wo + ig * k_per_group * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int iwi = tid % wi;
        int ihi = (tid / wi) % hi;
        int idi = tid / (hi * wi);

        float value = .0f;

        for(int ik = 0; ik < k_per_group; ik++)
        {
            for(int iz = 0; iz < fz; iz++)
            {
                int valid_d = 1;
                int cur_do  = idi + pz - dz * iz;
                if(cur_do < 0 || cur_do % sz)
                    valid_d &= 0;
                cur_do /= sz;
                if(cur_do >= do_)
                    valid_d &= 0;
                for(int iy = 0; iy < fy; iy++)
                {
                    int valid_h = 1;
                    int cur_ho  = ihi + py - dy * iy; // cur_h = sy*iho-py+dy*iy;
                    if(cur_ho < 0 || cur_ho % sy)
                        valid_h &= 0;
                    cur_ho /= sy;
                    if(cur_ho >= ho)
                        valid_h &= 0;
                    for(int ix = 0; ix < fx; ix++)
                    {
                        int valid_w = 1;
                        int cur_wo  = iwi + px - dx * ix; // cur_w = sx*iwo-px+dx*ix;
                        if(cur_wo < 0 || cur_wo % sx)
                            valid_w &= 0;
                        cur_wo /= sx;
                        if(cur_wo >= wo)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            int o_idx =
                                ik * do_ * ho * wo + cur_do * ho * wo + cur_ho * wo + cur_wo;
                            int f_idx =
                                ik * c_per_group * fz * fy * fx + iz * fy * fx + iy * fx + ix;
                            value += __bfloat16_to_float(p_out[o_idx]) *
                                     __bfloat16_to_float(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        int i_idx   = idi * hi * wi + ihi * wi + iwi;
        p_in[i_idx] = __float_to_bfloat16(value);
    }
}

extern "C" __global__ void naive_conv_wrw_ncdhw_bf16(const ushort* __restrict__ p_in,
                                                     ushort* __restrict__ p_wei,
                                                     const ushort* __restrict__ p_out,
                                                     int di,
                                                     int hi,
                                                     int wi,
                                                     int n,
                                                     int k_per_group,
                                                     int c_per_group,
                                                     int do_,
                                                     int ho,
                                                     int wo,
                                                     int sz,
                                                     int sy,
                                                     int sx,
                                                     int dz,
                                                     int dy,
                                                     int dx,
                                                     int pz,
                                                     int py,
                                                     int px,
                                                     int fz,
                                                     int fy,
                                                     int fx,
                                                     int group)
{
    /*
     *  need to compute total filter pixel: `group * k_per_group * c_per_group * fz * fy * fx`.
     *  to distribute this workload, let one workgroup compute `c_per_group * fz * fy * fx` pixel,
     *  hence need `group * k_per_group` workgroups (grid_size).
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = c_per_group * fz * fy * fx;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group;
    int ig            = bid / k_per_group;

    p_in += ig * c_per_group * di * hi * wi;
    p_wei += ig * k_per_group * c_per_group * fz * fy * fx + ik * c_per_group * fz * fy * fx;
    p_out += ig * k_per_group * do_ * ho * wo + ik * do_ * ho * wo;

    for(int tid = threadIdx.x; tid < thread_length; tid += 256)
    {
        int ix = tid % fx;
        int iy = (tid / fx) % fy;
        int iz = (tid / (fx * fy)) % fz;
        int ic = tid / (fx * fy * fz);

        float value = .0f;

        for(int in = 0; in < n; in++)
        {
            for(int ido = 0; ido < do_; ido++)
            {
                int valid_d = 1;
                int cur_d   = sz * ido - pz + dz * iz;
                if(cur_d < 0 || cur_d >= di)
                    valid_d &= 0;
                for(int iho = 0; iho < ho; iho++)
                {
                    int valid_h = 1;
                    int cur_h   = sy * iho - py + dy * iy;
                    if(cur_h < 0 || cur_h >= hi)
                        valid_h &= 0;
                    for(int iwo = 0; iwo < wo; iwo++)
                    {
                        int valid_w = 1;
                        int cur_w   = sx * iwo - px + dx * ix;
                        if(cur_w < 0 || cur_w >= wi)
                            valid_w &= 0;

                        if(valid_d & valid_h & valid_w)
                        {
                            int i_idx = in * c * di * hi * wi + ic * di * hi * wi +
                                        cur_d * hi * wi + cur_h * wi + cur_w;
                            int o_idx = in * k * do_ * ho * wo + ido * ho * wo + iho * wo + iwo;
                            value += __bfloat16_to_float(p_in[i_idx]) *
                                     __bfloat16_to_float(p_out[o_idx]);
                        }
                    }
                }
            }
        }
        int f_idx    = ic * fz * fy * fx + iz * fy * fx + iy * fx + ix;
        p_wei[f_idx] = __float_to_bfloat16(value);
    }
}
