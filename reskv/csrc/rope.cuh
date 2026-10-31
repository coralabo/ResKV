// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

namespace reskv {
namespace rope {

template<int GQA_RATIO, int DPT>
__device__ __forceinline__ void rotate_query_inplace(
    float (&q_reg)[GQA_RATIO][DPT],
    float sin0, float cos0,
    float sin1, float cos1)
{
    static_assert(DPT >= 4, "rotate_query_inplace expects 128-d RoPE lanes");
    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        const float q0 = q_reg[q][0];
        const float q1 = q_reg[q][1];
        const float q2 = q_reg[q][2];
        const float q3 = q_reg[q][3];
        q_reg[q][0] = __fmaf_rn(-q2, sin0, q0 * cos0);
        q_reg[q][2] = __fmaf_rn( q0, sin0, q2 * cos0);
        q_reg[q][1] = __fmaf_rn(-q3, sin1, q1 * cos1);
        q_reg[q][3] = __fmaf_rn( q1, sin1, q3 * cos1);
    }
}

template<int DPT>
__device__ __forceinline__ void rotate_key_inplace(
    float (&k_val)[DPT],
    float sin0, float cos0,
    float sin1, float cos1)
{
    static_assert(DPT >= 4, "rotate_key_inplace expects 128-d RoPE lanes");
    const float k0 = k_val[0];
    const float k1 = k_val[1];
    const float k2 = k_val[2];
    const float k3 = k_val[3];
    k_val[0] = __fmaf_rn(-k2, sin0, k0 * cos0);
    k_val[2] = __fmaf_rn( k0, sin0, k2 * cos0);
    k_val[1] = __fmaf_rn(-k3, sin1, k1 * cos1);
    k_val[3] = __fmaf_rn( k1, sin1, k3 * cos1);
}

__device__ __forceinline__ void advance_sincos(
    float& sinv,
    float& cosv,
    float sin_step,
    float cos_step)
{
    const float next_sin = __fmaf_rn(cosv, sin_step, sinv * cos_step);
    const float next_cos = __fmaf_rn(-sinv, sin_step, cosv * cos_step);
    sinv = next_sin;
    cosv = next_cos;
}

template<int GQA_RATIO, int DPT>
__device__ __forceinline__ float query_key_dot(
    float const (&q_reg)[GQA_RATIO][DPT],
    float const (&k_val)[DPT],
    int q)
{
    static_assert(DPT >= 4, "query_key_dot expects at least four lane values");
    float dot = 0.0f;
    #pragma unroll
    for (int i = 0; i < DPT; ++i) {
        dot = __fmaf_rn(q_reg[q][i], k_val[i], dot);
    }
    return dot;
}

}  // namespace rope
}  // namespace reskv
