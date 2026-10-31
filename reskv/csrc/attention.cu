// SPDX-License-Identifier: Apache-2.0
// ResKV fused decode attention with per-token centroid fusion.
//
// One query token attends to a compressed K/V cache. Each compressed token
// stores a cluster id and a sparse (optionally quantized) residual; the kernel
// reconstructs k = centroid_k + residual_k and v = centroid_v + residual_v in
// registers, applies RoPE to the pre-RoPE key on the fly, and accumulates an
// online softmax. Dense K/V are never written to global memory.
//
// Flash Decoding split-K: the sequence is partitioned into `num_splits` chunks
// (grid.x); each block emits a partial (max, sum, output) that
// merge_*_partials_kernel combines. merge_recent folds in the dense recent
// window with the same log-sum-exp merge.
//
// Residual formats (see reskv/tensors.py):
//   FP16 unstructured / tiled: retained values + a 128-bit position bitmap;
//   structured 4:16: four retained values per 16 dims, 4-bit packed indices;
//   INT4/INT2: signed integers packed into int32 words with a per-token scale.
// cluster_mass.cu implements the deferred Value centroid fusion variant.
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "rope.cuh"

#define FK16_WARP 32
#define FK16_WARPS 8
#define FK16_THREADS (FK16_WARP * FK16_WARPS)
#define FK16_D 128
#define FK16_DPT (FK16_D / FK16_WARP)
// Compressed tokens per warp iteration; residual rows are staged in shared
// memory, double-buffered with cp.async when the row is 16-byte aligned.
#define FK16_QTILE_TOKENS_PER_WARP 4
#ifndef FK16_QTILE_STRUCT_INDEX_4X16
#define FK16_QTILE_STRUCT_INDEX_4X16 0
#endif
#define FK16_QTILE_STAGE_ROWS 2
#define FK16_MAX_NNZ 128

namespace {

__device__ __forceinline__ bool fk16_can_cp_async() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return true;
#else
    return false;
#endif
}

__device__ __forceinline__ void fk16_cp_async_ca_16(
    void* __restrict__ smem_ptr,
    const void* __restrict__ gmem_ptr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr));
#endif
}

__device__ __forceinline__ void fk16_cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void fk16_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void fk16_stage_half_row_async(
    __half* __restrict__ smem_dst,
    const __half* __restrict__ gmem_src,
    int n_elems,
    int lane) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int idx = lane * 8; idx + 7 < n_elems;
         idx += FK16_WARP * 8) {
        fk16_cp_async_ca_16(smem_dst + idx, gmem_src + idx);
    }
    for (int idx = ((n_elems / 8) * 8) + lane; idx < n_elems;
         idx += FK16_WARP) {
        smem_dst[idx] = __ldg(&gmem_src[idx]);
    }
#else
    for (int idx = lane; idx < n_elems; idx += FK16_WARP) {
        smem_dst[idx] = __ldg(&gmem_src[idx]);
    }
#endif
}

__device__ __forceinline__ void fk16_stage_half_row_scalar(
    __half* __restrict__ smem_dst,
    const __half* __restrict__ gmem_src,
    int n_elems,
    int lane) {
    for (int idx = lane; idx < n_elems; idx += FK16_WARP) {
        smem_dst[idx] = __ldg(&gmem_src[idx]);
    }
}

inline int resolve_num_splits_fp16(int T, int B) {
    int auto_splits;
    if (B <= 4) {
        auto_splits = (T + 127) / 128;
    } else if (B >= 8 && T >= 4096) {
        auto_splits = (T + 511) / 512;
    } else {
        auto_splits = (T + 255) / 256;
    }
    if (auto_splits < 1) return 1;
    if (auto_splits > 40) return 40;
    return auto_splits;
}


inline int resolve_num_splits_fp16_qtile(int T, int B) {
    if (B == 8 && T >= 7000) {
        int auto_splits = (T + 255) / 256;
        if (auto_splits < 1) return 1;
        if (auto_splits > 40) return 40;
        return auto_splits;
    }
    return resolve_num_splits_fp16(T, B);
}

__device__ __forceinline__ float load_sparse_fp16_qtile_residual_from_mask_row(
    const __half* __restrict__ qvalue_row,
    uint32_t mask,
    int offset,
    int nnz,
    uint32_t bit_mask,
    uint32_t lower_mask) {
    if ((mask & bit_mask) == 0u) {
        return 0.0f;
    }
    const int rank = __popc(mask & lower_mask);
    const int idx = offset + rank;
    if (idx >= nnz) {
        return 0.0f;
    }
    return __half2float(qvalue_row[idx]);
}

__device__ __forceinline__ float load_sparse_fp16_qtile_residual_4x16_index_row(
    const __half* __restrict__ qvalue_row,
    uint32_t index_code,
    int offset,
    int lane_bit) {
    int rank = -1;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int idx = static_cast<int>((index_code >> (r * 4)) & 0xfu);
        if (idx == lane_bit) {
            rank = r;
        }
    }
    return (rank >= 0) ? __half2float(qvalue_row[offset + rank]) : 0.0f;
}

__device__ __forceinline__ float load_sparse_int4_qtile_residual_4x16_index_row(
    const int32_t* __restrict__ qpacked_row,
    uint32_t group_codes, int d, float scale) {
    const int group_in_word = (d >> 2) & 15;
    const int selected = static_cast<int>((group_codes >> (group_in_word * 2)) & 3u);
    if (selected != (d & 3)) return 0.0f;
    const int idx = d >> 2;
    const uint32_t word = static_cast<uint32_t>(qpacked_row[idx >> 3]);
    const int q = static_cast<int>((word >> ((idx & 7) * 4)) & 0xfu);
    return static_cast<float>((q ^ 8) - 8) * scale;
}

__device__ __forceinline__ void fk16_stage_int4_row_scalar(
    int32_t* __restrict__ dst,
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ meta,
    int lane) {
    if (lane < 4) dst[lane] = __ldg(&src[lane]);
    if (lane < 2) dst[4 + lane] = __ldg(&meta[lane]);
}

template<int GQA_RATIO, bool INT4_STRUCT = false>
__global__ void __launch_bounds__(FK16_THREADS, 2)
fused_kqv_fp16_compact_qtile_partial_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ k_centroids,
    const int8_t* __restrict__ k_cids,
    const void* __restrict__ k_qvalues,
    const int32_t* __restrict__ k_qmeta,
    const __half* __restrict__ k_qscale,
    const __half* __restrict__ v_centroids,
    const int8_t* __restrict__ v_cids,
    const void* __restrict__ v_qvalues,
    const int32_t* __restrict__ v_qmeta,
    const __half* __restrict__ v_qscale,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    __half* __restrict__ partial_out,
    int B, int nh, int nh_kv, int T,
    int nc_k, int nc_v, int k_nnz, int v_nnz,
    int start_pos, int pos_q, float rope_base, float scaling,
    int num_splits, const float* inv_freq) {
    const int split_idx = blockIdx.x;
    const int h_kv = blockIdx.y;
    const int b = blockIdx.z;
    const int kv_idx = b * nh_kv + h_kv;
    const int tid = threadIdx.x;
    const int warp_id = tid / FK16_WARP;
    const int lane = tid & (FK16_WARP - 1);
    const int lane_bit = lane & 15;
    const uint32_t lane_bit_mask = 1u << lane_bit;
    const uint32_t lane_lower_mask = lane_bit_mask - 1u;
    const int k_nnz_per_qtile = k_nnz / (FK16_D / 16);
    const int v_nnz_per_qtile = v_nnz / (FK16_D / 16);
#if FK16_QTILE_STRUCT_INDEX_4X16
    if (k_nnz_per_qtile != 4 || v_nnz_per_qtile != 4) {
        return;
    }
#endif

    const int split_start = (split_idx * T) / num_splits;
    const int split_end = ((split_idx + 1) * T) / num_splits;

    extern __shared__ char smem_raw[];
    __half* smem_k_cent = reinterpret_cast<__half*>(smem_raw);
    __half* smem_v_cent = smem_k_cent + nc_k * FK16_D;
    __half* smem_k_qvalues = smem_v_cent + nc_v * FK16_D;
    __half* smem_v_qvalues = smem_k_qvalues
        + FK16_WARPS * FK16_QTILE_STAGE_ROWS * FK16_MAX_NNZ;

    __shared__ float smem_m[GQA_RATIO][FK16_WARPS];
    __shared__ float smem_l[GQA_RATIO][FK16_WARPS];
    __shared__ float smem_acc[GQA_RATIO][FK16_DPT][FK16_WARPS][FK16_WARP];

    const __half* k_cent_base = k_centroids + kv_idx * nc_k * FK16_D;
    const __half* v_cent_base = v_centroids + kv_idx * nc_v * FK16_D;
    for (int i = tid; i < nc_k * FK16_D; i += FK16_THREADS) {
        smem_k_cent[i] = __ldg(&k_cent_base[i]);
    }
    for (int i = tid; i < nc_v * FK16_D; i += FK16_THREADS) {
        smem_v_cent[i] = __ldg(&v_cent_base[i]);
    }
    __syncthreads();

    const float freq0 = (inv_freq ? inv_freq[lane] : 1.0f / powf(rope_base, static_cast<float>(lane * 2) / FK16_D));
    const float freq1 = (inv_freq ? inv_freq[lane+32] : 1.0f / powf(rope_base, static_cast<float>((lane + 32) * 2) / FK16_D));
    float sin_step0, cos_step0, sin_step1, cos_step1;
    __sincosf(freq0, &sin_step0, &cos_step0);
    __sincosf(freq1, &sin_step1, &cos_step1);

    float q_reg[GQA_RATIO][FK16_DPT];
    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        const int q_head = h_kv * GQA_RATIO + q;
        const int q_offset = (b * nh + q_head) * FK16_D;
        #pragma unroll
        for (int i = 0; i < FK16_DPT; ++i) {
            q_reg[q][i] = __half2float(__ldg(&Q[q_offset + lane + i * FK16_WARP]));
        }
    }

    float sin0_q, cos0_q, sin1_q, cos1_q;
    __sincosf(pos_q * freq0, &sin0_q, &cos0_q);
    __sincosf(pos_q * freq1, &sin1_q, &cos1_q);
    reskv::rope::rotate_query_inplace<GQA_RATIO, FK16_DPT>(
        q_reg, sin0_q, cos0_q, sin1_q, cos1_q);

    float m_loc[GQA_RATIO];
    float l_loc[GQA_RATIO];
    float acc[GQA_RATIO][FK16_DPT];
    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        m_loc[q] = -FLT_MAX;
        l_loc[q] = 0.0f;
        #pragma unroll
        for (int i = 0; i < FK16_DPT; ++i) {
            acc[q][i] = 0.0f;
        }
    }

    for (int token_base = split_start + warp_id * FK16_QTILE_TOKENS_PER_WARP;
         token_base < split_end;
         token_base += FK16_WARPS * FK16_QTILE_TOKENS_PER_WARP) {
        float sin0_k_cur, cos0_k_cur, sin1_k_cur, cos1_k_cur;
        const int pos_k_base = start_pos + token_base;
        __sincosf(pos_k_base * freq0, &sin0_k_cur, &cos0_k_cur);
        __sincosf(pos_k_base * freq1, &sin1_k_cur, &cos1_k_cur);

        const bool use_async_qvalues = !INT4_STRUCT
            && fk16_can_cp_async()
            && ((k_nnz & 7) == 0)
            && ((v_nnz & 7) == 0);

        if (use_async_qvalues && token_base < split_end) {
            const int off0 = kv_idx * T + token_base;
            const int stage_base = warp_id * FK16_QTILE_STAGE_ROWS * FK16_MAX_NNZ;
            fk16_stage_half_row_async(
                smem_k_qvalues + stage_base,
                reinterpret_cast<const __half*>(k_qvalues) + off0 * k_nnz,
                k_nnz,
                lane);
            fk16_stage_half_row_async(
                smem_v_qvalues + stage_base,
                reinterpret_cast<const __half*>(v_qvalues) + off0 * v_nnz,
                v_nnz,
                lane);
            fk16_cp_async_commit();
            fk16_cp_async_wait_all();
            __syncwarp();
        }

        #pragma unroll
        for (int ti = 0; ti < FK16_QTILE_TOKENS_PER_WARP; ++ti) {
            const int t = token_base + ti;
            if (t >= split_end) continue;
            const int token_offset = kv_idx * T + t;
            const int qvalue_stage = use_async_qvalues ? (ti & 1) : 0;
            const int qvalue_stage_base =
                (warp_id * FK16_QTILE_STAGE_ROWS + qvalue_stage) * FK16_MAX_NNZ;
            if (use_async_qvalues) {
                const int next_ti = ti + 1;
                const int next_t = token_base + next_ti;
                if (next_ti < FK16_QTILE_TOKENS_PER_WARP && next_t < split_end) {
                    const int next_off = kv_idx * T + next_t;
                    const int next_stage = next_ti & 1;
                    const int next_base =
                        (warp_id * FK16_QTILE_STAGE_ROWS + next_stage) * FK16_MAX_NNZ;
                    fk16_stage_half_row_async(
                        smem_k_qvalues + next_base,
                        reinterpret_cast<const __half*>(k_qvalues) + next_off * k_nnz,
                        k_nnz,
                        lane);
                    fk16_stage_half_row_async(
                        smem_v_qvalues + next_base,
                        reinterpret_cast<const __half*>(v_qvalues) + next_off * v_nnz,
                        v_nnz,
                        lane);
                    fk16_cp_async_commit();
                }
            } else {
                if constexpr (INT4_STRUCT) {
                    fk16_stage_int4_row_scalar(
                        reinterpret_cast<int32_t*>(smem_k_qvalues + qvalue_stage_base),
                        reinterpret_cast<const int32_t*>(k_qvalues) + token_offset * 4,
                        k_qmeta + token_offset * 2, lane);
                    fk16_stage_int4_row_scalar(
                        reinterpret_cast<int32_t*>(smem_v_qvalues + qvalue_stage_base),
                        reinterpret_cast<const int32_t*>(v_qvalues) + token_offset * 4,
                        v_qmeta + token_offset * 2, lane);
                } else {
                    fk16_stage_half_row_scalar(
                        smem_k_qvalues + qvalue_stage_base,
                        reinterpret_cast<const __half*>(k_qvalues) + token_offset * k_nnz,
                        k_nnz, lane);
                    fk16_stage_half_row_scalar(
                        smem_v_qvalues + qvalue_stage_base,
                        reinterpret_cast<const __half*>(v_qvalues) + token_offset * v_nnz,
                        v_nnz, lane);
                }
                __syncwarp();
            }
            const int k_cid = static_cast<int>(__ldg(&k_cids[token_offset]));
            const int v_cid = static_cast<int>(__ldg(&v_cids[token_offset]));
            const __half* k_cent = smem_k_cent + k_cid * FK16_D;
            const __half* v_cent = smem_v_cent + v_cid * FK16_D;
            const __half* k_qvalue_row = smem_k_qvalues + qvalue_stage_base;
            const __half* v_qvalue_row = smem_v_qvalues + qvalue_stage_base;
            float k_scale_val = 1.0f;
            float v_scale_val = 1.0f;
            if constexpr (INT4_STRUCT) {
                if (lane == 0) {
                    k_scale_val = __half2float(__ldg(&k_qscale[token_offset]));
                    v_scale_val = __half2float(__ldg(&v_qscale[token_offset]));
                }
                k_scale_val = __shfl_sync(0xffffffff, k_scale_val, 0);
                v_scale_val = __shfl_sync(0xffffffff, v_scale_val, 0);
            }
            uint32_t k_word_lane = 0;
            uint32_t v_word_lane = 0;
            constexpr int META_WORDS = INT4_STRUCT ? 2 : (FK16_D / 32);
            if (!INT4_STRUCT && lane < META_WORDS) {
                k_word_lane = static_cast<uint32_t>(
                    __ldg(&k_qmeta[token_offset * META_WORDS + lane]));
                v_word_lane = static_cast<uint32_t>(
                    __ldg(&v_qmeta[token_offset * META_WORDS + lane]));
            }
            const uint32_t k_words[FK16_DPT] = {
                __shfl_sync(0xffffffff, k_word_lane, 0),
                __shfl_sync(0xffffffff, k_word_lane, 1),
                __shfl_sync(0xffffffff, k_word_lane, 2),
                __shfl_sync(0xffffffff, k_word_lane, 3),
            };
            const uint32_t v_words[FK16_DPT] = {
                __shfl_sync(0xffffffff, v_word_lane, 0),
                __shfl_sync(0xffffffff, v_word_lane, 1),
                __shfl_sync(0xffffffff, v_word_lane, 2),
                __shfl_sync(0xffffffff, v_word_lane, 3),
            };

            float k_val[FK16_DPT];
            float v_val[FK16_DPT];
            #pragma unroll
            for (int i = 0; i < FK16_DPT; ++i) {
                const int d = lane + i * FK16_WARP;
                const int upper_half = lane >> 4;
                uint32_t k_word = k_words[INT4_STRUCT ? (i >> 1) : i];
                uint32_t v_word = v_words[INT4_STRUCT ? (i >> 1) : i];
                if constexpr (INT4_STRUCT) {
                    k_word = static_cast<uint32_t>(
                        reinterpret_cast<const int32_t*>(k_qvalue_row)[4 + (i >> 1)]);
                    v_word = static_cast<uint32_t>(
                        reinterpret_cast<const int32_t*>(v_qvalue_row)[4 + (i >> 1)]);
                }
                const uint32_t k_mask = upper_half ? (k_word >> 16) : (k_word & 0xffffu);
                const uint32_t v_mask = upper_half ? (v_word >> 16) : (v_word & 0xffffu);
                const int qtile = (i << 1) + upper_half;
                const int k_offset = qtile * k_nnz_per_qtile;
                const int v_offset = qtile * v_nnz_per_qtile;
                float k_res;
                float v_res;
#if FK16_QTILE_STRUCT_INDEX_4X16
                if constexpr (INT4_STRUCT) {
                    k_res = load_sparse_int4_qtile_residual_4x16_index_row(
                        reinterpret_cast<const int32_t*>(k_qvalue_row), k_word, d, k_scale_val);
                    v_res = load_sparse_int4_qtile_residual_4x16_index_row(
                        reinterpret_cast<const int32_t*>(v_qvalue_row), v_word, d, v_scale_val);
                } else {
                    k_res = load_sparse_fp16_qtile_residual_4x16_index_row(
                        k_qvalue_row, k_mask, k_offset, lane_bit);
                    v_res = load_sparse_fp16_qtile_residual_4x16_index_row(
                        v_qvalue_row, v_mask, v_offset, lane_bit);
                }
#else
                k_res = load_sparse_fp16_qtile_residual_from_mask_row(
                    k_qvalue_row, k_mask, k_offset, k_nnz, lane_bit_mask, lane_lower_mask);
                v_res = load_sparse_fp16_qtile_residual_from_mask_row(
                    v_qvalue_row, v_mask, v_offset, v_nnz, lane_bit_mask, lane_lower_mask);
#endif
                k_val[i] = __half2float(k_cent[d]) + k_res;
                v_val[i] = __half2float(v_cent[d]) + v_res;
            }

            reskv::rope::rotate_key_inplace<FK16_DPT>(
                k_val, sin0_k_cur, cos0_k_cur, sin1_k_cur, cos1_k_cur);

            #pragma unroll
            for (int q = 0; q < GQA_RATIO; ++q) {
                float dot = reskv::rope::query_key_dot<GQA_RATIO, FK16_DPT>(
                    q_reg, k_val, q);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, off);
                }
                const float score = dot * scaling;
                const float m_new = fmaxf(m_loc[q], score);
                const float exp_old = (m_loc[q] == -FLT_MAX) ? 0.0f : __expf(m_loc[q] - m_new);
                const float exp_new = __expf(score - m_new);
                l_loc[q] = __fmaf_rn(l_loc[q], exp_old, exp_new);
                #pragma unroll
                for (int i = 0; i < FK16_DPT; ++i) {
                    acc[q][i] = __fmaf_rn(acc[q][i], exp_old, exp_new * v_val[i]);
                }
                m_loc[q] = m_new;
            }

            reskv::rope::advance_sincos(
                sin0_k_cur, cos0_k_cur, sin_step0, cos_step0);
            reskv::rope::advance_sincos(
                sin1_k_cur, cos1_k_cur, sin_step1, cos_step1);
            if (use_async_qvalues) {
                const int next_ti = ti + 1;
                const int next_t = token_base + next_ti;
                if (next_ti < FK16_QTILE_TOKENS_PER_WARP && next_t < split_end) {
                    fk16_cp_async_wait_all();
                    __syncwarp();
                }
            }
        }
    }

    if (lane < GQA_RATIO) {
        smem_m[lane][warp_id] = m_loc[lane];
        smem_l[lane][warp_id] = l_loc[lane];
    }
    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        #pragma unroll
        for (int i = 0; i < FK16_DPT; ++i) {
            smem_acc[q][i][warp_id][lane] = acc[q][i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        const int q_head = h_kv * GQA_RATIO + q;
        float fm = smem_m[q][0];
        float fl = smem_l[q][0];
        float facc[FK16_DPT];
        #pragma unroll
        for (int i = 0; i < FK16_DPT; ++i) {
            facc[i] = smem_acc[q][i][0][lane];
        }
        #pragma unroll
        for (int w = 1; w < FK16_WARPS; ++w) {
            const float wm = smem_m[q][w];
            const float wl = smem_l[q][w];
            const float mn = fmaxf(fm, wm);
            const float e0 = (fm == -FLT_MAX) ? 0.0f : __expf(fm - mn);
            const float ew = (wm == -FLT_MAX) ? 0.0f : __expf(wm - mn);
            fl = __fmaf_rn(fl, e0, wl * ew);
            #pragma unroll
            for (int i = 0; i < FK16_DPT; ++i) {
                facc[i] = __fmaf_rn(facc[i], e0, smem_acc[q][i][w][lane] * ew);
            }
            fm = mn;
        }
        const int flat_q = b * nh + q_head;
        if (lane == 0) {
            partial_m[flat_q * num_splits + split_idx] = fm;
            partial_l[flat_q * num_splits + split_idx] = fl;
        }
        const float inv_fl = 1.0f / fmaxf(fl, 1e-20f);
        #pragma unroll
        for (int i = 0; i < FK16_DPT; ++i) {
            const int d = lane + i * FK16_WARP;
            partial_out[(flat_q * num_splits + split_idx) * FK16_D + d] =
                __float2half(facc[i] * inv_fl);
        }
    }
}

template<int GQA_RATIO, bool INT4_STRUCT = false>
inline void set_compact_qtile_partial_smem_attr(size_t smem_bytes) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused_kqv_fp16_compact_qtile_partial_kernel<GQA_RATIO, INT4_STRUCT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)));
}

__global__ void merge_fp16_partials_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const __half* __restrict__ partial_out,
    __half* __restrict__ output,
    float* __restrict__ final_m,
    float* __restrict__ final_l,
    int B_nh, int num_splits) {
    const int flat_q = blockIdx.x;
    const int d = threadIdx.x;
    if (flat_q >= B_nh || d >= FK16_D) return;

    float fm = -FLT_MAX;
    float fl = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        const int idx = flat_q * num_splits + s;
        const float sm = __ldg(&partial_m[idx]);
        const float sl = __ldg(&partial_l[idx]);
        if (sl <= 0.0f || sm == -FLT_MAX) {
            continue;
        }
        const float sout = __half2float(__ldg(&partial_out[idx * FK16_D + d]));
        if (fl == 0.0f) {
            fm = sm;
            fl = sl;
            acc = sout * sl;
        } else {
            const float mn = fmaxf(fm, sm);
            const float e0 = __expf(fm - mn);
            const float e1 = __expf(sm - mn);
            acc = acc * e0 + sout * sl * e1;
            fl = fl * e0 + sl * e1;
            fm = mn;
        }
    }
    output[flat_q * FK16_D + d] = __float2half((fl > 0.0f) ? (acc / fl) : 0.0f);
    if (d == 0) {
        final_m[flat_q] = fm;
        final_l[flat_q] = fl;
    }
}

template<int GQA_RATIO>
__global__ void merge_fp16_dense_recent_kernel(
    const __half* __restrict__ comp_out,
    const float* __restrict__ comp_m,
    const float* __restrict__ comp_l,
    const __half* __restrict__ Q_rope,
    const __half* __restrict__ k_recent,
    const __half* __restrict__ v_recent,
    __half* __restrict__ output,
    int B, int nh, int nh_kv, int T_recent, float scaling) {
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int h_kv = h / GQA_RATIO;
    const int flat_q = b * nh + h;

    __shared__ float score_smem;

    float m = __ldg(&comp_m[flat_q]);
    float l = __ldg(&comp_l[flat_q]);
    float acc = 0.0f;
    if (tid < FK16_D) {
        acc = __half2float(__ldg(&comp_out[flat_q * FK16_D + tid])) * l;
    }

    for (int t = 0; t < T_recent; ++t) {
        if (tid < 32) {
            // One warp reproduces the original 128D reduction tree exactly:
            // (p[d] + p[d+64]) + (p[d+32] + p[d+96]), then strides 16..1.
            // Explicit rounded products prevent fast-math FMA contraction.
            const size_t q_base = (size_t)flat_q * FK16_D + tid;
            const size_t k_base = ((size_t)b * nh_kv + h_kv) * T_recent * FK16_D
                + (size_t)t * FK16_D + tid;
            float p[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                p[i] = __fmul_rn(__half2float(__ldg(&Q_rope[q_base + i * 32])),
                                 __half2float(__ldg(&k_recent[k_base + i * 32])));
            }
            float dot = (p[0] + p[2]) + (p[1] + p[3]);
            #pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1) {
                dot += __shfl_down_sync(0xffffffff, dot, stride);
            }
            if (tid == 0) score_smem = dot * scaling;
        }
        __syncthreads();

        const float score = score_smem;
        const float new_m = fmaxf(m, score);
        const float old_scale = (m == -FLT_MAX) ? 0.0f : __expf(m - new_m);
        const float w = __expf(score - new_m);
        if (tid < FK16_D) {
            const int v_idx = ((b * nh_kv + h_kv) * T_recent + t) * FK16_D + tid;
            const float v = __half2float(__ldg(&v_recent[v_idx]));
            acc = acc * old_scale + w * v;
        }
        l = l * old_scale + w;
        m = new_m;
        __syncthreads();
    }

    if (tid < FK16_D) {
        output[flat_q * FK16_D + tid] = __float2half((l > 0.0f) ? (acc / l) : 0.0f);
    }
}

template<bool INT4_STRUCT = false>
std::vector<torch::Tensor> sparse_fused_kqv_fp16_compact_qtile_impl(
    torch::Tensor Q,
    torch::Tensor k_centroids,
    torch::Tensor k_cluster_ids,
    torch::Tensor k_qvalues,
    torch::Tensor k_qmeta,
    torch::Tensor k_qscale,
    torch::Tensor v_centroids,
    torch::Tensor v_cluster_ids,
    torch::Tensor v_qvalues,
    torch::Tensor v_qmeta,
    torch::Tensor v_qscale,
    int num_kv_heads,
    int num_q_heads,
    int start_pos,
    int pos_q,
    float rope_base,
    float attn_scaling, int splits, torch::Tensor inv_freq) {
    c10::cuda::CUDAGuard guard(Q.device());
    TORCH_CHECK(Q.is_cuda(), "Q must be CUDA");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q must be fp16");
    TORCH_CHECK(k_centroids.scalar_type() == torch::kFloat16, "k_centroids must be fp16");
    TORCH_CHECK(v_centroids.scalar_type() == torch::kFloat16, "v_centroids must be fp16");
    TORCH_CHECK(k_cluster_ids.scalar_type() == torch::kInt8, "k_cluster_ids must be int8");
    TORCH_CHECK(v_cluster_ids.scalar_type() == torch::kInt8, "v_cluster_ids must be int8");
    TORCH_CHECK(k_qvalues.scalar_type() == (INT4_STRUCT ? torch::kInt32 : torch::kFloat16), "K payload dtype mismatch");
    TORCH_CHECK(v_qvalues.scalar_type() == (INT4_STRUCT ? torch::kInt32 : torch::kFloat16), "V payload dtype mismatch");
    if constexpr (INT4_STRUCT) {
        TORCH_CHECK(k_qscale.scalar_type() == torch::kFloat16 && v_qscale.scalar_type() == torch::kFloat16,
                    "INT4 scales must be fp16");
    }
    TORCH_CHECK(k_qmeta.scalar_type() == torch::kInt32, "k_qmeta must be int32");
    TORCH_CHECK(v_qmeta.scalar_type() == torch::kInt32, "v_qmeta must be int32");
    TORCH_CHECK(Q.dim() == 2 && Q.size(1) == FK16_D, "Q must be [B*nh, 128]");
    TORCH_CHECK(k_centroids.dim() == 3 && k_centroids.size(2) == FK16_D, "k_centroids must be [B*nh_kv, nc, 128]");
    TORCH_CHECK(v_centroids.dim() == 3 && v_centroids.size(2) == FK16_D, "v_centroids must be [B*nh_kv, nc, 128]");
    TORCH_CHECK(k_cluster_ids.dim() == 2 && v_cluster_ids.dim() == 2, "cluster ids must be rank-2");
    TORCH_CHECK(k_qvalues.dim() == 3 && v_qvalues.dim() == 3, "qtile values must be [B*nh_kv, T, nnz]");
    TORCH_CHECK(k_qmeta.dim() == 3 && k_qmeta.size(2) == (INT4_STRUCT ? 2 : FK16_D / 32),
                "k_qmeta width mismatch");
    TORCH_CHECK(v_qmeta.dim() == 3 && v_qmeta.size(2) == (INT4_STRUCT ? 2 : FK16_D / 32),
                "v_qmeta width mismatch");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads");
    const int gqa = num_q_heads / num_kv_heads;
    TORCH_CHECK(gqa == 1 || gqa == 4 || gqa == 8, "Unsupported GQA ratio: ", gqa);

    const int B = k_centroids.size(0) / num_kv_heads;
    const int nh = num_q_heads;
    const int T = k_cluster_ids.size(1);
    const int nc_k = k_centroids.size(1);
    const int nc_v = v_centroids.size(1);
    const int k_nnz = INT4_STRUCT ? 32 : k_qvalues.size(2);
    const int v_nnz = INT4_STRUCT ? 32 : v_qvalues.size(2);
    TORCH_CHECK(k_qvalues.size(2) == (INT4_STRUCT ? 4 : k_nnz) &&
                v_qvalues.size(2) == (INT4_STRUCT ? 4 : v_nnz), "structured payload width mismatch");
    if constexpr (INT4_STRUCT) {
        TORCH_CHECK(k_qscale.dim() == 2 && k_qscale.size(1) == T, "K scale shape mismatch");
        TORCH_CHECK(v_qscale.dim() == 2 && v_qscale.size(1) == T, "V scale shape mismatch");
    }
    const int B_nh = B * nh;
    const int num_splits = splits > 0 ? std::min(splits, T) : resolve_num_splits_fp16_qtile(T, B);
    TORCH_CHECK(
        k_nnz < FK16_D && v_nnz < FK16_D,
        "FP16 sparse qtile fused KQV requires compact sparse qvalues with nnz < 128; "
        "dense-zero qtiles are disabled because they do not reduce KV memory traffic");
    TORCH_CHECK(
        k_nnz % (FK16_D / 16) == 0 && v_nnz % (FK16_D / 16) == 0,
        "structured FP16 sparse qtile requires K/V nnz divisible by 8");
#if FK16_QTILE_STRUCT_INDEX_4X16
    TORCH_CHECK(
        k_nnz == 32 && v_nnz == 32,
        "4:16 struct sparse index FP16 qtile fused KQV is only valid for sparse75 (K/V nnz=32)");
#endif

    TORCH_CHECK(k_centroids.size(0) == B * num_kv_heads, "K centroid batch-head mismatch");
    TORCH_CHECK(v_centroids.size(0) == B * num_kv_heads, "V centroid batch-head mismatch");
    TORCH_CHECK(k_cluster_ids.size(0) == B * num_kv_heads && k_cluster_ids.size(1) == T, "K cid shape mismatch");
    TORCH_CHECK(v_cluster_ids.size(0) == B * num_kv_heads && v_cluster_ids.size(1) == T, "V cid shape mismatch");
    TORCH_CHECK(k_qvalues.size(0) == B * num_kv_heads && k_qvalues.size(1) == T, "K qvalue shape mismatch");
    TORCH_CHECK(v_qvalues.size(0) == B * num_kv_heads && v_qvalues.size(1) == T, "V qvalue shape mismatch");
    TORCH_CHECK(k_qmeta.size(0) == B * num_kv_heads && k_qmeta.size(1) == T, "K qmeta shape mismatch");
    TORCH_CHECK(v_qmeta.size(0) == B * num_kv_heads && v_qmeta.size(1) == T, "V qmeta shape mismatch");

    auto float_opts = Q.options().dtype(torch::kFloat32);
    auto partial_m = torch::empty({B_nh, num_splits}, float_opts);
    auto partial_l = torch::empty({B_nh, num_splits}, float_opts);
    auto partial_out = torch::empty({B_nh, num_splits, FK16_D}, Q.options());
    auto output = torch::empty({B_nh, FK16_D}, Q.options());
    auto final_m = torch::empty({B_nh}, float_opts);
    auto final_l = torch::empty({B_nh}, float_opts);

    dim3 grid(num_splits, num_kv_heads, B);
    dim3 block(FK16_THREADS);
    const size_t centroid_smem_bytes =
        static_cast<size_t>(nc_k + nc_v) * FK16_D * sizeof(__half);
    const size_t qvalue_stage_smem_bytes =
        static_cast<size_t>(2) * FK16_WARPS * FK16_QTILE_STAGE_ROWS * FK16_MAX_NNZ * sizeof(__half);
    const size_t smem_bytes = centroid_smem_bytes + qvalue_stage_smem_bytes;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (gqa == 1) {
        set_compact_qtile_partial_smem_attr<1, INT4_STRUCT>(smem_bytes);
        fused_kqv_fp16_compact_qtile_partial_kernel<1, INT4_STRUCT><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_centroids.data_ptr<at::Half>()),
            k_cluster_ids.data_ptr<int8_t>(),
            k_qvalues.data_ptr(),
            k_qmeta.data_ptr<int32_t>(),
            INT4_STRUCT ? reinterpret_cast<const __half*>(k_qscale.data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<const __half*>(v_centroids.data_ptr<at::Half>()),
            v_cluster_ids.data_ptr<int8_t>(),
            v_qvalues.data_ptr(),
            v_qmeta.data_ptr<int32_t>(),
            INT4_STRUCT ? reinterpret_cast<const __half*>(v_qscale.data_ptr<at::Half>()) : nullptr,
            partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
            reinterpret_cast<__half*>(partial_out.data_ptr<at::Half>()),
            B, nh, num_kv_heads, T, nc_k, nc_v, k_nnz, v_nnz,
            start_pos, pos_q, rope_base, attn_scaling, num_splits, inv_freq.numel() ? inv_freq.data_ptr<float>() : nullptr);
    } else if (gqa == 4) {
        set_compact_qtile_partial_smem_attr<4, INT4_STRUCT>(smem_bytes);
        fused_kqv_fp16_compact_qtile_partial_kernel<4, INT4_STRUCT><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_centroids.data_ptr<at::Half>()),
            k_cluster_ids.data_ptr<int8_t>(),
            k_qvalues.data_ptr(),
            k_qmeta.data_ptr<int32_t>(),
            INT4_STRUCT ? reinterpret_cast<const __half*>(k_qscale.data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<const __half*>(v_centroids.data_ptr<at::Half>()),
            v_cluster_ids.data_ptr<int8_t>(),
            v_qvalues.data_ptr(),
            v_qmeta.data_ptr<int32_t>(),
            INT4_STRUCT ? reinterpret_cast<const __half*>(v_qscale.data_ptr<at::Half>()) : nullptr,
            partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
            reinterpret_cast<__half*>(partial_out.data_ptr<at::Half>()),
            B, nh, num_kv_heads, T, nc_k, nc_v, k_nnz, v_nnz,
            start_pos, pos_q, rope_base, attn_scaling, num_splits, inv_freq.numel() ? inv_freq.data_ptr<float>() : nullptr);
    } else {
        set_compact_qtile_partial_smem_attr<8, INT4_STRUCT>(smem_bytes);
        fused_kqv_fp16_compact_qtile_partial_kernel<8, INT4_STRUCT><<<grid, block, smem_bytes, stream>>>(
            reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_centroids.data_ptr<at::Half>()),
            k_cluster_ids.data_ptr<int8_t>(),
            k_qvalues.data_ptr(),
            k_qmeta.data_ptr<int32_t>(),
            INT4_STRUCT ? reinterpret_cast<const __half*>(k_qscale.data_ptr<at::Half>()) : nullptr,
            reinterpret_cast<const __half*>(v_centroids.data_ptr<at::Half>()),
            v_cluster_ids.data_ptr<int8_t>(),
            v_qvalues.data_ptr(),
            v_qmeta.data_ptr<int32_t>(),
            INT4_STRUCT ? reinterpret_cast<const __half*>(v_qscale.data_ptr<at::Half>()) : nullptr,
            partial_m.data_ptr<float>(), partial_l.data_ptr<float>(),
            reinterpret_cast<__half*>(partial_out.data_ptr<at::Half>()),
            B, nh, num_kv_heads, T, nc_k, nc_v, k_nnz, v_nnz,
            start_pos, pos_q, rope_base, attn_scaling, num_splits, inv_freq.numel() ? inv_freq.data_ptr<float>() : nullptr);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    merge_fp16_partials_kernel<<<B_nh, FK16_D, 0, stream>>>(
        partial_m.data_ptr<float>(),
        partial_l.data_ptr<float>(),
        reinterpret_cast<const __half*>(partial_out.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
        final_m.data_ptr<float>(),
        final_l.data_ptr<float>(),
        B_nh,
        num_splits);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {output, final_m, final_l};
}

}  // namespace

torch::Tensor merge_fused_kqv_fp16_dense_recent(
    torch::Tensor comp_out,
    torch::Tensor comp_m,
    torch::Tensor comp_l,
    torch::Tensor Q_rope,
    torch::Tensor k_recent,
    torch::Tensor v_recent,
    int num_kv_heads,
    int num_q_heads,
    float attn_scaling) {
    c10::cuda::CUDAGuard guard(comp_out.device());
    TORCH_CHECK(comp_out.dim() == 2 && comp_out.size(1) == FK16_D, "comp_out must be [B*nh, 128]");
    TORCH_CHECK(comp_m.dim() == 1 && comp_l.dim() == 1, "comp_m/comp_l must be [B*nh]");
    TORCH_CHECK(Q_rope.dim() == 2 && Q_rope.size(1) == FK16_D, "Q_rope must be [B*nh, 128]");
    TORCH_CHECK(k_recent.dim() == 4 && k_recent.size(3) == FK16_D, "k_recent must be [B, nh_kv, T, 128]");
    TORCH_CHECK(v_recent.dim() == 4 && v_recent.size(3) == FK16_D, "v_recent must be [B, nh_kv, T, 128]");
    TORCH_CHECK(comp_out.scalar_type() == torch::kFloat16, "comp_out must be fp16");
    TORCH_CHECK(Q_rope.scalar_type() == torch::kFloat16, "Q_rope must be fp16");
    TORCH_CHECK(k_recent.scalar_type() == torch::kFloat16, "k_recent must be fp16");
    TORCH_CHECK(v_recent.scalar_type() == torch::kFloat16, "v_recent must be fp16");

    const int B = k_recent.size(0);
    const int T_recent = k_recent.size(2);
    const int gqa = num_q_heads / num_kv_heads;
    TORCH_CHECK(T_recent > 0, "T_recent must be > 0");
    TORCH_CHECK(gqa == 1 || gqa == 4 || gqa == 8, "Unsupported GQA ratio: ", gqa);
    TORCH_CHECK(k_recent.size(1) == num_kv_heads, "k_recent head mismatch");
    TORCH_CHECK(v_recent.size(0) == B && v_recent.size(1) == num_kv_heads && v_recent.size(2) == T_recent,
                "v_recent shape mismatch");
    TORCH_CHECK(Q_rope.size(0) == B * num_q_heads && comp_out.size(0) == B * num_q_heads,
                "Q/comp_out batch-head mismatch");

    auto output = torch::empty_like(comp_out);
    dim3 grid(num_q_heads, B);
    dim3 block(FK16_D);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (gqa == 1) {
        merge_fp16_dense_recent_kernel<1><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(comp_out.data_ptr<at::Half>()),
            comp_m.data_ptr<float>(), comp_l.data_ptr<float>(),
            reinterpret_cast<const __half*>(Q_rope.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_recent.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_recent.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, num_q_heads, num_kv_heads, T_recent, attn_scaling);
    } else if (gqa == 4) {
        merge_fp16_dense_recent_kernel<4><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(comp_out.data_ptr<at::Half>()),
            comp_m.data_ptr<float>(), comp_l.data_ptr<float>(),
            reinterpret_cast<const __half*>(Q_rope.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_recent.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_recent.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, num_q_heads, num_kv_heads, T_recent, attn_scaling);
    } else {
        merge_fp16_dense_recent_kernel<8><<<grid, block, 0, stream>>>(
            reinterpret_cast<const __half*>(comp_out.data_ptr<at::Half>()),
            comp_m.data_ptr<float>(), comp_l.data_ptr<float>(),
            reinterpret_cast<const __half*>(Q_rope.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(k_recent.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(v_recent.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, num_q_heads, num_kv_heads, T_recent, attn_scaling);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


#define RQ_WARP 32
#define RQ_WARPS 8
#define RQ_THREADS (RQ_WARP * RQ_WARPS)
#define RQ_D 128
#define RQ_DPT (RQ_D / RQ_WARP)
#ifndef RQ_QTILE_TOKENS_PER_WARP
#define RQ_QTILE_TOKENS_PER_WARP 16
#endif

namespace {

static_assert(RQ_QTILE_TOKENS_PER_WARP > 0 &&
              RQ_QTILE_TOKENS_PER_WARP <= RQ_WARP,
              "RQ_QTILE_TOKENS_PER_WARP must be in [1, 32]");

template<int BITWIDTH>
struct SignedQuantBitConfig {
    static constexpr int values_per_word = 32 / BITWIDTH;
    static constexpr int mask = (1 << BITWIDTH) - 1;
    static constexpr int sign_bit = 1 << (BITWIDTH - 1);

    __device__ __forceinline__ static int unpack(int32_t word, int idx) {
        const int raw = (word >> (idx * BITWIDTH)) & mask;
        return (raw ^ sign_bit) - sign_bit;
    }
};

inline int resolve_num_splits(int T, int B) {
    int auto_splits;
    if (B <= 4) {
        // Small batches need finer token splits for block parallelism.
        auto_splits = (T + 127) / 128;
    } else if (T <= 4096) {
        // Short sequences use coarser splits to reduce merge overhead.
        auto_splits = (T + 819) / 820;
    } else if (B >= 8) {
        // Balance token parallelism and merge overhead for longer sequences.
        auto_splits = (T + 511) / 512;
    } else {
        // Use additional token parallelism for smaller batches.
        auto_splits = (T + 807) / 808;
    }
    if (auto_splits < 1) return 1;
    if (B <= 4 && auto_splits > 32) return 32;
    if (auto_splits > 40) return 40;
    return auto_splits;
}

inline int resolve_num_splits_qtile(int T, int B) {
    if (B == 8 && T >= 7000) {
        int auto_splits = (T + 240) / 480;
        if (auto_splits < 1) return 1;
        if (auto_splits > 40) return 40;
        return auto_splits;
    }
    return resolve_num_splits(T, B);
}

template<int GQA_RATIO, int KBITWIDTH, int VBITWIDTH>
__global__ void __launch_bounds__(RQ_THREADS, 2)
quant_decode_qtile_partial_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ k_centroids,
    const int8_t* __restrict__ k_cids,
    const int32_t* __restrict__ k_qtiles,
    const __half* __restrict__ k_qscale,
    const __half* __restrict__ v_centroids,
    const int8_t* __restrict__ v_cids,
    const int32_t* __restrict__ v_qtiles,
    const __half* __restrict__ v_qscale,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    __half* __restrict__ partial_out,
    int B, int nh, int nh_kv, int T,
    int nc_k, int nc_v, int k_qw, int v_qw,
    int start_pos, int pos_q, float rope_base, float scaling,
    int num_splits, const float* inv_freq)
{
    const int split_idx = blockIdx.x;
    const int h_kv = blockIdx.y;
    const int b = blockIdx.z;
    const int kv_idx = b * nh_kv + h_kv;
    const int tid = threadIdx.x;
    const int warp_id = tid / RQ_WARP;
    const int lane = tid % RQ_WARP;

    const int split_start = (split_idx * T) / num_splits;
    const int split_end = ((split_idx + 1) * T) / num_splits;

    extern __shared__ char smem_raw[];

    __half* smem_k_cent = reinterpret_cast<__half*>(smem_raw);
    __half* smem_v_cent = smem_k_cent + nc_k * RQ_D;

    __shared__ float smem_m[GQA_RATIO][RQ_WARPS];
    __shared__ float smem_l[GQA_RATIO][RQ_WARPS];
    __shared__ float smem_acc[GQA_RATIO][RQ_DPT][RQ_WARPS][RQ_WARP];

    const __half* k_cent_base = k_centroids + kv_idx * nc_k * RQ_D;
    const __half* v_cent_base = v_centroids + kv_idx * nc_v * RQ_D;

    for (int idx = tid; idx < nc_k * RQ_D; idx += RQ_THREADS) {
        smem_k_cent[idx] = __ldg(&k_cent_base[idx]);
    }
    for (int idx = tid; idx < nc_v * RQ_D; idx += RQ_THREADS) {
        smem_v_cent[idx] = __ldg(&v_cent_base[idx]);
    }
    __syncthreads();

    const float freq0 = (inv_freq ? inv_freq[lane] : 1.0f / powf(rope_base, static_cast<float>(lane * 2) / RQ_D));
    const float freq1 = (inv_freq ? inv_freq[lane+32] : 1.0f / powf(rope_base, static_cast<float>((lane + 32) * 2) / RQ_D));
    float sin_step0 = 0.0f, cos_step0 = 1.0f;
    float sin_step1 = 0.0f, cos_step1 = 1.0f;
    {
        __sincosf(freq0, &sin_step0, &cos_step0);
        __sincosf(freq1, &sin_step1, &cos_step1);
    }

    float q_reg[GQA_RATIO][RQ_DPT];
    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        const int q_head = h_kv * GQA_RATIO + q;
        const int q_offset = (b * nh + q_head) * RQ_D;
        #pragma unroll
        for (int i = 0; i < RQ_DPT; ++i) {
            q_reg[q][i] = __half2float(__ldg(&Q[q_offset + lane + i * RQ_WARP]));
        }
    }

    float sin0_q, cos0_q, sin1_q, cos1_q;
    __sincosf(pos_q * freq0, &sin0_q, &cos0_q);
    __sincosf(pos_q * freq1, &sin1_q, &cos1_q);
    reskv::rope::rotate_query_inplace<GQA_RATIO, RQ_DPT>(
        q_reg, sin0_q, cos0_q, sin1_q, cos1_q);

    float m_loc[GQA_RATIO];
    float l_loc[GQA_RATIO];
    float acc[GQA_RATIO][RQ_DPT];

    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        m_loc[q] = -FLT_MAX;
        l_loc[q] = 0.0f;
        #pragma unroll
        for (int i = 0; i < RQ_DPT; ++i) {
            acc[q][i] = 0.0f;
        }
    }

    using KCfg = SignedQuantBitConfig<KBITWIDTH>;
    using VCfg = SignedQuantBitConfig<VBITWIDTH>;

    for (int token_base = split_start + warp_id * RQ_QTILE_TOKENS_PER_WARP;
         token_base < split_end;
         token_base += RQ_WARPS * RQ_QTILE_TOKENS_PER_WARP) {
        float sin0_k_cur = 0.0f, cos0_k_cur = 1.0f;
        float sin1_k_cur = 0.0f, cos1_k_cur = 1.0f;
        {
            const int pos_k_base = start_pos + token_base;
            __sincosf(pos_k_base * freq0, &sin0_k_cur, &cos0_k_cur);
            __sincosf(pos_k_base * freq1, &sin1_k_cur, &cos1_k_cur);
        }

        int k_cid_lane = 0;
        int v_cid_lane = 0;
        float kqs_lane = 0.0f;
        float vqs_lane = 0.0f;
        if (lane < RQ_QTILE_TOKENS_PER_WARP) {
            const int ti = lane;
            const int t = token_base + ti;
            const bool valid = t < split_end;
            const int token_offset = kv_idx * T + t;
            if (valid) {
                k_cid_lane = static_cast<int>(__ldg(&k_cids[token_offset]));
                v_cid_lane = static_cast<int>(__ldg(&v_cids[token_offset]));
                kqs_lane = __half2float(__ldg(&k_qscale[token_offset]));
                vqs_lane = __half2float(__ldg(&v_qscale[token_offset]));
            }
        }

        #pragma unroll
        for (int ti = 0; ti < RQ_QTILE_TOKENS_PER_WARP; ++ti) {
            const int t = token_base + ti;
            const bool valid = t < split_end;
            const int token_offset = kv_idx * T + t;

            int32_t k_word_lane = 0;
            int32_t v_word_lane = 0;
            if (valid && lane < k_qw) {
                k_word_lane = __ldg(&k_qtiles[token_offset * k_qw + lane]);
            }
            if (valid && lane < v_qw) {
                v_word_lane = __ldg(&v_qtiles[token_offset * v_qw + lane]);
            }

            if (!valid) {
                continue;
            }

            const int k_cid = __shfl_sync(0xffffffff, k_cid_lane, ti);
            const int v_cid = __shfl_sync(0xffffffff, v_cid_lane, ti);
            const float kqs = __shfl_sync(0xffffffff, kqs_lane, ti);
            const float vqs = __shfl_sync(0xffffffff, vqs_lane, ti);

            const __half* k_cent_row = smem_k_cent + k_cid * RQ_D;
            const __half* v_cent_row = smem_v_cent + v_cid * RQ_D;

            float k_val[RQ_DPT];
            float v_val[RQ_DPT];
            #pragma unroll
            for (int i = 0; i < RQ_DPT; ++i) {
                const int d = lane + i * RQ_WARP;
                const int32_t k_word =
                    __shfl_sync(0xffffffff, k_word_lane, d / KCfg::values_per_word);
                const int32_t v_word =
                    __shfl_sync(0xffffffff, v_word_lane, d / VCfg::values_per_word);
                const int kq = KCfg::unpack(k_word, d % KCfg::values_per_word);
                const int vq = VCfg::unpack(v_word, d % VCfg::values_per_word);
                k_val[i] = __fmaf_rn(static_cast<float>(kq), kqs, __half2float(k_cent_row[d]));
                v_val[i] = __fmaf_rn(static_cast<float>(vq), vqs, __half2float(v_cent_row[d]));
            }

            float k_abs[RQ_DPT];
            #pragma unroll
            for (int i = 0; i < RQ_DPT; ++i) {
                k_abs[i] = k_val[i];
            }
            {
                reskv::rope::rotate_key_inplace<RQ_DPT>(
                    k_abs, sin0_k_cur, cos0_k_cur, sin1_k_cur, cos1_k_cur);
            }

            #pragma unroll
            for (int q = 0; q < GQA_RATIO; ++q) {
                float dot = reskv::rope::query_key_dot<GQA_RATIO, RQ_DPT>(
                    q_reg, k_abs, q);

                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, off);
                }

                const float score = dot * scaling;
                const float m_new = fmaxf(m_loc[q], score);
                const float exp_old = __expf(m_loc[q] - m_new);
                const float exp_new = __expf(score - m_new);

                l_loc[q] = __fmaf_rn(l_loc[q], exp_old, exp_new);
                #pragma unroll
                for (int i = 0; i < RQ_DPT; ++i) {
                    acc[q][i] = __fmaf_rn(acc[q][i], exp_old, exp_new * v_val[i]);
                }
                m_loc[q] = m_new;
            }
            {
                reskv::rope::advance_sincos(
                    sin0_k_cur, cos0_k_cur, sin_step0, cos_step0);
                reskv::rope::advance_sincos(
                    sin1_k_cur, cos1_k_cur, sin_step1, cos_step1);
            }
        }
    }

    if (lane < GQA_RATIO) {
        smem_m[lane][warp_id] = m_loc[lane];
        smem_l[lane][warp_id] = l_loc[lane];
    }
    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        #pragma unroll
        for (int i = 0; i < RQ_DPT; ++i) {
            smem_acc[q][i][warp_id][lane] = acc[q][i];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < GQA_RATIO; ++q) {
        float fm = smem_m[q][0];
        float fl = smem_l[q][0];
        float facc[RQ_DPT];
        #pragma unroll
        for (int i = 0; i < RQ_DPT; ++i) {
            facc[i] = smem_acc[q][i][0][lane];
        }

        #pragma unroll
        for (int w = 1; w < RQ_WARPS; ++w) {
            const float wm = smem_m[q][w];
            const float mn = fmaxf(fm, wm);
            const float e0 = __expf(fm - mn);
            const float ew = __expf(wm - mn);
            fl = __fmaf_rn(fl, e0, smem_l[q][w] * ew);
            #pragma unroll
            for (int i = 0; i < RQ_DPT; ++i) {
                facc[i] = __fmaf_rn(facc[i], e0, smem_acc[q][i][w][lane] * ew);
            }
            fm = mn;
        }

        const int q_head = h_kv * GQA_RATIO + q;
        const int flat_q = b * nh + q_head;
        if (lane == 0) {
            partial_m[flat_q * num_splits + split_idx] = fm;
            partial_l[flat_q * num_splits + split_idx] = fl;
        }
        const float inv_fl = 1.0f / fmaxf(fl, 1e-20f);
        #pragma unroll
        for (int i = 0; i < RQ_DPT; ++i) {
            const int d = lane + i * RQ_WARP;
            partial_out[(flat_q * num_splits + split_idx) * RQ_D + d] = __float2half(facc[i] * inv_fl);
        }
    }
}

template<int GQA_RATIO>
__global__ void __launch_bounds__(GQA_RATIO * RQ_WARP, 4)
quant_decode_merge_kernel(
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const __half* __restrict__ partial_out,
    __half* __restrict__ output,
    float* __restrict__ final_m,
    float* __restrict__ final_l,
    int B, int nh, int nh_kv, int num_splits)
{
    const int h_kv = blockIdx.x;
    const int b = blockIdx.y;
    const int warp_id = threadIdx.x / RQ_WARP;
    const int lane = threadIdx.x % RQ_WARP;

    if (warp_id >= GQA_RATIO) {
        return;
    }

    const int q = warp_id;
    const int q_head = h_kv * GQA_RATIO + q;
    const int flat_q = b * nh + q_head;

    float fm = partial_m[flat_q * num_splits];
    float fl = partial_l[flat_q * num_splits];
    float facc[RQ_DPT];
    #pragma unroll
    for (int i = 0; i < RQ_DPT; ++i) {
        const int d = lane + i * RQ_WARP;
        facc[i] = __half2float(partial_out[(flat_q * num_splits) * RQ_D + d]) * fl;
    }

    for (int split = 1; split < num_splits; ++split) {
        const int split_offset = flat_q * num_splits + split;
        const float sm = partial_m[split_offset];
        const float sl = partial_l[split_offset];
        const float mn = fmaxf(fm, sm);
        const float ef = __expf(fm - mn);
        const float es = __expf(sm - mn);
        fl = __fmaf_rn(fl, ef, sl * es);
        #pragma unroll
        for (int i = 0; i < RQ_DPT; ++i) {
            const int d = lane + i * RQ_WARP;
            const float sout = __half2float(partial_out[split_offset * RQ_D + d]);
            facc[i] = __fmaf_rn(facc[i], ef, sout * sl * es);
        }
        fm = mn;
    }

    #pragma unroll
    for (int i = 0; i < RQ_DPT; ++i) {
        const int d = lane + i * RQ_WARP;
        output[flat_q * RQ_D + d] = __float2half(facc[i] / fmaxf(fl, 1e-20f));
    }
    if (lane == 0) {
        if (final_m != nullptr) {
            final_m[flat_q] = fm;
        }
        if (final_l != nullptr) {
            final_l[flat_q] = fl;
        }
    }
}





} // namespace


std::vector<torch::Tensor> decode_quantized(
    torch::Tensor Q, torch::Tensor kc, torch::Tensor ki, torch::Tensor kp, torch::Tensor ks,
    torch::Tensor vc, torch::Tensor vi, torch::Tensor vp, torch::Tensor vs,
    int nh_kv, int nh, int start_pos, int pos_q, float rope_base, float scaling,
    int k_bits, int v_bits, int splits, torch::Tensor inv_freq) {
    c10::cuda::CUDAGuard guard(Q.device());
    const int B = Q.size(0) / nh, T = ki.size(1), nc_k = kc.size(1), nc_v = vc.size(1);
    const int gqa = nh / nh_kv, k_qw = kp.size(2), v_qw = vp.size(2);
    const int num_splits = splits > 0 ? std::min(splits, T) : resolve_num_splits_qtile(T, B);
    TORCH_CHECK(gqa == 1 || gqa == 4 || gqa == 8, "Unsupported GQA");
    TORCH_CHECK(nc_k <= 64 && nc_v <= 64, "At most 64 centroids supported");
    auto f32 = Q.options().dtype(torch::kFloat32);
    auto pm = torch::empty({B*nh, num_splits}, f32), pl = torch::empty_like(pm);
    auto po = torch::empty({B*nh, num_splits, 128}, Q.options());
    auto o = torch::empty_like(Q);
    auto fm = torch::empty({B*nh}, f32), fl = torch::empty_like(fm);
    dim3 grid(num_splits, nh_kv, B), block(RQ_THREADS);
    const size_t smem = (nc_k + nc_v) * 128 * sizeof(__half);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const float* freq = inv_freq.numel() ? inv_freq.data_ptr<float>() : nullptr;
#define RESKV_QUANT_LAUNCH(G, K, V) do { \
    auto kernel = quant_decode_qtile_partial_kernel<G, K, V>; \
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem)); \
    kernel<<<grid, block, smem, stream>>>( \
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), \
        reinterpret_cast<const __half*>(kc.data_ptr<at::Half>()), ki.data_ptr<int8_t>(), kp.data_ptr<int32_t>(), \
        reinterpret_cast<const __half*>(ks.data_ptr<at::Half>()), \
        reinterpret_cast<const __half*>(vc.data_ptr<at::Half>()), vi.data_ptr<int8_t>(), vp.data_ptr<int32_t>(), \
        reinterpret_cast<const __half*>(vs.data_ptr<at::Half>()), \
        pm.data_ptr<float>(), pl.data_ptr<float>(), reinterpret_cast<__half*>(po.data_ptr<at::Half>()), \
        B, nh, nh_kv, T, nc_k, nc_v, k_qw, v_qw, start_pos, pos_q, rope_base, scaling, num_splits, freq); \
} while (0)
#define RESKV_QUANT_BITS(G) \
    if (k_bits == 4 && v_bits == 4) { RESKV_QUANT_LAUNCH(G,4,4); } \
    else if (k_bits == 2 && v_bits == 2) { RESKV_QUANT_LAUNCH(G,2,2); } \
    else if (k_bits == 4 && v_bits == 2) { RESKV_QUANT_LAUNCH(G,4,2); } \
    else { RESKV_QUANT_LAUNCH(G,2,4); }
    if (gqa == 1) { RESKV_QUANT_BITS(1); }
    else if (gqa == 4) { RESKV_QUANT_BITS(4); }
    else { RESKV_QUANT_BITS(8); }
#undef RESKV_QUANT_BITS
#undef RESKV_QUANT_LAUNCH
#define RESKV_QUANT_MERGE(G) quant_decode_merge_kernel<G><<<dim3(nh_kv,B), G*32, 0, stream>>>( \
    pm.data_ptr<float>(), pl.data_ptr<float>(), reinterpret_cast<const __half*>(po.data_ptr<at::Half>()), \
    reinterpret_cast<__half*>(o.data_ptr<at::Half>()), fm.data_ptr<float>(), fl.data_ptr<float>(), \
    B, nh, nh_kv, num_splits)
    if (gqa == 1) { RESKV_QUANT_MERGE(1); }
    else if (gqa == 4) { RESKV_QUANT_MERGE(4); }
    else { RESKV_QUANT_MERGE(8); }
#undef RESKV_QUANT_MERGE
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {o, fm, fl};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_fp16", &sparse_fused_kqv_fp16_compact_qtile_impl<false>);
    m.def("decode_int4", &sparse_fused_kqv_fp16_compact_qtile_impl<true>);
    m.def("decode_quantized", &decode_quantized);
    m.def("merge_recent", &merge_fused_kqv_fp16_dense_recent);
}
