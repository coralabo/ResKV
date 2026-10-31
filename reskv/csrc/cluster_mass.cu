// SPDX-License-Identifier: Apache-2.0
// ResKV fused decode attention with deferred Value centroid fusion.
//
// Since v_t = centroid_v[c_t] + residual_v[t], the attention output is
//   sum_t p_t v_t = sum_t p_t residual_v[t] + sum_c beta_c centroid_v[c],
// where beta_c = sum_{t in cluster c} p_t is the attention mass of cluster c.
// The token loop therefore accumulates only the sparse Value residuals and the
// per-cluster mass beta; Value centroids are applied once after the split-K
// reduction instead of once per token. Keys are reconstructed per token as in
// attention.cu. Layout (unstructured or structured 4:16) and residual dtype
// (FP16 or INT4) are template parameters.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace {

constexpr int CMASS_WARP_SIZE        = 32;
constexpr int CMASS_WARPS            = 8;
constexpr int CMASS_THREADS          = CMASS_WARP_SIZE * CMASS_WARPS;
constexpr int CMASS_HEAD_DIM         = 128;
constexpr int CMASS_DPT              = CMASS_HEAD_DIM / CMASS_WARP_SIZE;     // 4
constexpr int CMASS_MAX_NC           = 64;
constexpr int CMASS_MAX_PW           = 16;
constexpr int CMASS_MAX_NNZ          = 128;

#ifndef CMASS_TOKENS_PER_WARP
#define CMASS_TOKENS_PER_WARP 2
#endif

// Residual rows up to this many values are prefetched for a whole warp iteration.
#define CMASS_SHORT_BULK_NNZ 32

#ifndef CMASS_INT4_FIXED_SHIFT_SOFTMAX
#define CMASS_INT4_FIXED_SHIFT_SOFTMAX 0
#endif

#ifndef CMASS_INT4_ROTATE_K_ONCE
#define CMASS_INT4_ROTATE_K_ONCE 0
#endif

#ifndef CMASS_INT4_DIRECT_PACKED
#define CMASS_INT4_DIRECT_PACKED 0
#endif

template<int BITWIDTH>
struct CmassBitConfig {
    static constexpr int values_per_word = 32 / BITWIDTH;
    static constexpr int mask = (1 << BITWIDTH) - 1;
};

inline int cmass_resolve_num_splits(int T, int B) {
    int auto_splits;
    if (B <= 2) {
        auto_splits = (T + 127) / 128;
    } else if (B <= 4) {
        auto_splits = (T + 191) / 192;
    } else if (T <= 4096) {
        auto_splits = (T + 255) / 256;
    } else if (B >= 8) {
        // Balance token parallelism with the cost of merging centroid partials.
        auto_splits = (T * 3 + 1023) / 1024;
    } else {
        auto_splits = (T + 383) / 384;
    }
    if (auto_splits < 1) return 1;
    if (auto_splits > 64) return 64;
    return auto_splits;
}

inline void cmass_check_launch(const char* tag) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, tag, " launch failed: ", cudaGetErrorString(err));
}

__device__ __forceinline__ bool cmass_can_cp_async() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    return true;
#else
    return false;
#endif
}

__device__ __forceinline__ void cmass_cp_async_ca_16(
    void* __restrict__ smem_ptr,
    const void* __restrict__ gmem_ptr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned smem_addr =
        static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr));
#endif
}

__device__ __forceinline__ void cmass_cp_async_commit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cmass_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void cmass_stage_half_row_async(
    __half* __restrict__ smem_dst,
    const __half* __restrict__ gmem_src,
    int n_elems,
    int lane)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    for (int idx = lane * 8; idx + 7 < n_elems;
         idx += CMASS_WARP_SIZE * 8) {
        cmass_cp_async_ca_16(smem_dst + idx, gmem_src + idx);
    }
#endif
}

__device__ __forceinline__ void cmass_stage_half_row_scalar(
    __half* __restrict__ smem_dst,
    const __half* __restrict__ gmem_src,
    int n_elems,
    int lane)
{
    for (int idx = lane; idx < n_elems; idx += CMASS_WARP_SIZE) {
        smem_dst[idx] = __ldg(&gmem_src[idx]);
    }
}

__device__ __forceinline__ void cmass_stage_int_row_scalar(
    int32_t* __restrict__ smem_dst,
    const int32_t* __restrict__ gmem_src,
    int n_words,
    int lane)
{
    for (int idx = lane; idx < n_words; idx += CMASS_WARP_SIZE) {
        smem_dst[idx] = __ldg(&gmem_src[idx]);
    }
}

__device__ __forceinline__ float cmass_load_4x16_struct_index_row(
    const __half* __restrict__ qvalue_row,
    uint32_t index_code,
    int qtile,
    int lane_bit)
{
    int rank = -1;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int idx = static_cast<int>((index_code >> (r * 4)) & 0xfu);
        if (idx == lane_bit) {
            rank = r;
        }
    }
    return (rank >= 0) ? __half2float(qvalue_row[qtile * 4 + rank]) : 0.0f;
}

__device__ __forceinline__ float cmass_load_int4_4x16_struct_index_row(
    const int32_t* __restrict__ qpacked_row,
    uint32_t group_codes,
    int d,
    float scale)
{
    const int group_in_word = (d >> 2) & 15;
    const int selected = static_cast<int>((group_codes >> (group_in_word * 2)) & 3u);
    if (selected != (d & 3)) return 0.0f;
    const int compact_idx = d >> 2;
    const uint32_t word = static_cast<uint32_t>(qpacked_row[compact_idx >> 3]);
    const int nibble = static_cast<int>((word >> ((compact_idx & 7) * 4)) & 0xfu);
    return static_cast<float>((nibble ^ 8) - 8) * scale;
}

template<int GQA, int NC_V_STATIC>
__global__ void __launch_bounds__(CMASS_THREADS, 2)
sparse_fused_kqv_cmass_fp16_partial_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ k_centroids,
    const int8_t* __restrict__ k_cids,
    const __half* __restrict__ k_sparse,
    const int64_t* __restrict__ k_bitmap,
    const int8_t* __restrict__ v_cids,
    const __half* __restrict__ v_sparse,
    const int64_t* __restrict__ v_bitmap,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_beta,
    float* __restrict__ partial_delta,
    int B, int nh, int nh_kv, int T,
    int nc_k, int nc_v, int k_nnz, int v_nnz,
    int start_pos, int pos_q, float rope_base, float scaling,
    int num_splits, const float* inv_freq)
{
    const int split_idx = blockIdx.x;
    const int h_kv      = blockIdx.y;
    const int b         = blockIdx.z;
    const int kv_idx    = b * nh_kv + h_kv;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / CMASS_WARP_SIZE;
    const int lane      = tid % CMASS_WARP_SIZE;

    const int split_start = (split_idx * T) / num_splits;
    const int split_end   = ((split_idx + 1) * T) / num_splits;

    extern __shared__ char smem_raw[];
    __half* smem_k_cent   = reinterpret_cast<__half*>(smem_raw);
    __half* smem_k_sparse = smem_k_cent + nc_k * CMASS_HEAD_DIM;
    __half* smem_v_sparse = smem_k_sparse
        + CMASS_WARPS * CMASS_TOKENS_PER_WARP * CMASS_MAX_NNZ;
    float* smem_beta      = reinterpret_cast<float*>(
        smem_v_sparse + CMASS_WARPS * CMASS_TOKENS_PER_WARP * CMASS_MAX_NNZ);
    constexpr int BETA_PER_WARP = GQA * NC_V_STATIC;
    constexpr int BETA_SLOTS    = (NC_V_STATIC + CMASS_WARP_SIZE - 1)
                                   / CMASS_WARP_SIZE;

    __shared__ int      smem_kcid_md[CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ int      smem_vcid_md[CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ uint64_t smem_kbmp0  [CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ uint64_t smem_kbmp1  [CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ uint64_t smem_vbmp0  [CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ uint64_t smem_vbmp1  [CMASS_WARPS][CMASS_TOKENS_PER_WARP];

    __shared__ float smem_m  [GQA][CMASS_WARPS];
    __shared__ float smem_l  [GQA][CMASS_WARPS];
    __shared__ float smem_dlt[GQA][CMASS_DPT][CMASS_WARPS][CMASS_WARP_SIZE];

    const __half* k_cent_base = k_centroids + kv_idx * nc_k * CMASS_HEAD_DIM;
    for (int idx = tid; idx < nc_k * CMASS_HEAD_DIM; idx += CMASS_THREADS) {
        smem_k_cent[idx] = __ldg(&k_cent_base[idx]);
    }
    __syncthreads();

    float q_reg[GQA][CMASS_DPT];
    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        const int q_head = h_kv * GQA + q;
        const int q_off  = (b * nh + q_head) * CMASS_HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            q_reg[q][i] = __half2float(
                __ldg(&Q[q_off + lane + i * CMASS_WARP_SIZE]));
        }
    }
    const float freq0 =
        (inv_freq ? inv_freq[lane] : 1.0f / powf(rope_base, static_cast<float>(lane * 2) / CMASS_HEAD_DIM));
    const float freq1 = (inv_freq ? inv_freq[lane+32] : 1.0f / powf(rope_base, static_cast<float>((lane + 32) * 2) / CMASS_HEAD_DIM));

    float m_loc[GQA];
    float l_loc[GQA];
    float dlt  [GQA][CMASS_DPT];
    float bet  [GQA][BETA_SLOTS];
    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        m_loc[q] = -FLT_MAX;
        l_loc[q] = 0.0f;
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            dlt[q][i] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < BETA_SLOTS; ++k) {
            bet[q][k] = 0.0f;
        }
    }

    for (int token_base = split_start + warp_id * CMASS_TOKENS_PER_WARP;
         token_base < split_end;
         token_base += CMASS_WARPS * CMASS_TOKENS_PER_WARP) {

        if (lane < CMASS_TOKENS_PER_WARP) {
            const int ti = lane;
            const int t  = token_base + ti;
            if (t < split_end) {
                const int off = kv_idx * T + t;
                const longlong2 kbv = __ldg(
                    reinterpret_cast<const longlong2*>(&k_bitmap[off * 2]));
                const longlong2 vbv = __ldg(
                    reinterpret_cast<const longlong2*>(&v_bitmap[off * 2]));
                smem_kcid_md[warp_id][ti] = static_cast<int>(__ldg(&k_cids[off]));
                smem_vcid_md[warp_id][ti] = static_cast<int>(__ldg(&v_cids[off]));
                smem_kbmp0  [warp_id][ti] = static_cast<uint64_t>(kbv.x);
                smem_kbmp1  [warp_id][ti] = static_cast<uint64_t>(kbv.y);
                smem_vbmp0  [warp_id][ti] = static_cast<uint64_t>(vbv.x);
                smem_vbmp1  [warp_id][ti] = static_cast<uint64_t>(vbv.y);
            }
        }

        const bool use_short_bulk_sparse =
            cmass_can_cp_async()
            && (k_nnz <= CMASS_SHORT_BULK_NNZ)
            && (v_nnz <= CMASS_SHORT_BULK_NNZ)
            && ((k_nnz & 7) == 0)
            && ((v_nnz & 7) == 0);
        const bool use_async_sparse =
            !use_short_bulk_sparse
            && cmass_can_cp_async()
            && ((k_nnz & 7) == 0)
            && ((v_nnz & 7) == 0);
        if (use_short_bulk_sparse) {
            #pragma unroll
            for (int pre_ti = 0; pre_ti < CMASS_TOKENS_PER_WARP; ++pre_ti) {
                const int pre_t = token_base + pre_ti;
                if (pre_t < split_end) {
                    const int pre_off = kv_idx * T + pre_t;
                    const int pre_base =
                        (warp_id * CMASS_TOKENS_PER_WARP + pre_ti) * CMASS_MAX_NNZ;
                    cmass_stage_half_row_async(
                        smem_k_sparse + pre_base,
                        k_sparse + pre_off * k_nnz,
                        k_nnz,
                        lane);
                    cmass_stage_half_row_async(
                        smem_v_sparse + pre_base,
                        v_sparse + pre_off * v_nnz,
                        v_nnz,
                        lane);
                }
            }
            cmass_cp_async_commit();
            cmass_cp_async_wait_all();
            __syncwarp();
        }
        if (use_async_sparse && token_base < split_end) {
            const int off0 = kv_idx * T + token_base;
            __half* k_stage0 = smem_k_sparse
                + (warp_id * CMASS_TOKENS_PER_WARP) * CMASS_MAX_NNZ;
            __half* v_stage0 = smem_v_sparse
                + (warp_id * CMASS_TOKENS_PER_WARP) * CMASS_MAX_NNZ;
            cmass_stage_half_row_async(
                k_stage0, k_sparse + off0 * k_nnz, k_nnz, lane);
            cmass_stage_half_row_async(
                v_stage0, v_sparse + off0 * v_nnz, v_nnz, lane);
            cmass_cp_async_commit();
            cmass_cp_async_wait_all();
            __syncwarp();
        }

        #pragma unroll
        for (int ti = 0; ti < CMASS_TOKENS_PER_WARP; ++ti) {
            const int t = token_base + ti;
            const bool valid = t < split_end;
            const int off = kv_idx * T + t;
            const int sparse_base =
                (warp_id * CMASS_TOKENS_PER_WARP
                 + (use_async_sparse ? (ti & 1) : ti)) * CMASS_MAX_NNZ;

            if (use_short_bulk_sparse) {
                // 75% sparsity keeps nnz=32: prefetch the whole 4-token warp
                // tile once, then consume it from shared memory without a
                // per-token wait/sync.
            } else if (use_async_sparse) {
                if (!valid) break;
                const int next_ti = ti + 1;
                const int next_t  = token_base + next_ti;
                if (next_ti < CMASS_TOKENS_PER_WARP && next_t < split_end) {
                    const int next_off = kv_idx * T + next_t;
                    const int next_base =
                        (warp_id * CMASS_TOKENS_PER_WARP
                         + (next_ti & 1)) * CMASS_MAX_NNZ;
                    cmass_stage_half_row_async(
                        smem_k_sparse + next_base,
                        k_sparse + next_off * k_nnz,
                        k_nnz,
                        lane);
                    cmass_stage_half_row_async(
                        smem_v_sparse + next_base,
                        v_sparse + next_off * v_nnz,
                        v_nnz,
                        lane);
                    cmass_cp_async_commit();
                }
            } else {
                if (valid) {
                    cmass_stage_half_row_scalar(
                        smem_k_sparse + sparse_base,
                        k_sparse + off * k_nnz,
                        k_nnz,
                        lane);
                    cmass_stage_half_row_scalar(
                        smem_v_sparse + sparse_base,
                        v_sparse + off * v_nnz,
                        v_nnz,
                        lane);
                }
                __syncwarp();
            }
            if (!valid) continue;

            const int k_cid = smem_kcid_md[warp_id][ti];
            const int v_cid = smem_vcid_md[warp_id][ti];
            const uint64_t k_bmp0 = smem_kbmp0[warp_id][ti];
            const uint64_t k_bmp1 = smem_kbmp1[warp_id][ti];
            const uint64_t v_bmp0 = smem_vbmp0[warp_id][ti];
            const uint64_t v_bmp1 = smem_vbmp1[warp_id][ti];
            const int k_pc0 = __popcll(k_bmp0);
            const int v_pc0 = __popcll(v_bmp0);

            const __half* k_cent_row = smem_k_cent + k_cid * CMASS_HEAD_DIM;
            const __half* k_sparse_row = smem_k_sparse + sparse_base;
            const __half* v_sparse_row = smem_v_sparse + sparse_base;

            float k_val[CMASS_DPT];
            float v_res[CMASS_DPT];
            #pragma unroll
            for (int i = 0; i < CMASS_DPT; ++i) {
                const int d       = lane + i * CMASS_WARP_SIZE;
                const int tile    = d >> 6;
                const int bit_pos = 63 - (d & 63);
                const uint64_t k_bmp = (tile == 0) ? k_bmp0 : k_bmp1;
                const uint64_t v_bmp = (tile == 0) ? v_bmp0 : v_bmp1;

                float kval = __half2float(k_cent_row[d]);
                if ((k_bmp >> bit_pos) & 1ULL) {
                    const uint64_t higher =
                        (bit_pos < 63) ? (k_bmp >> (bit_pos + 1)) : 0ULL;
                    const int sparse_idx = (tile == 0)
                        ? __popcll(higher) : (k_pc0 + __popcll(higher));
                    if (sparse_idx < k_nnz) {
                        kval += __half2float(k_sparse_row[sparse_idx]);
                    }
                }
                k_val[i] = kval;

                float vres = 0.0f;
                if ((v_bmp >> bit_pos) & 1ULL) {
                    const uint64_t higher =
                        (bit_pos < 63) ? (v_bmp >> (bit_pos + 1)) : 0ULL;
                    const int sparse_idx = (tile == 0)
                        ? __popcll(higher) : (v_pc0 + __popcll(higher));
                    if (sparse_idx < v_nnz) {
                        vres = __half2float(v_sparse_row[sparse_idx]);
                    }
                }
                v_res[i] = vres;
            }

            const int delta = pos_q - (start_pos + t);
            float sin0, cos0, sin1, cos1;
            __sincosf(static_cast<float>(delta) * freq0, &sin0, &cos0);
            __sincosf(static_cast<float>(delta) * freq1, &sin1, &cos1);

            #pragma unroll
            for (int q = 0; q < GQA; ++q) {
                float dot = 0.0f;
                dot = __fmaf_rn(q_reg[q][0] * cos0 - q_reg[q][2] * sin0,
                                k_val[0], dot);
                dot = __fmaf_rn(q_reg[q][1] * cos1 - q_reg[q][3] * sin1,
                                k_val[1], dot);
                dot = __fmaf_rn(q_reg[q][0] * sin0 + q_reg[q][2] * cos0,
                                k_val[2], dot);
                dot = __fmaf_rn(q_reg[q][1] * sin1 + q_reg[q][3] * cos1,
                                k_val[3], dot);
                #pragma unroll
                for (int off2 = 16; off2 > 0; off2 >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, off2);
                }
                const float score   = dot * scaling;
                const float m_new   = fmaxf(m_loc[q], score);
                const float exp_old = __expf(m_loc[q] - m_new);
                const float exp_new = __expf(score - m_new);

                l_loc[q] = __fmaf_rn(l_loc[q], exp_old, exp_new);
                #pragma unroll
                for (int i = 0; i < CMASS_DPT; ++i) {
                    dlt[q][i] = __fmaf_rn(dlt[q][i], exp_old,
                                           exp_new * v_res[i]);
                }
                #pragma unroll
                for (int k = 0; k < BETA_SLOTS; ++k) {
                    const int c = lane + k * CMASS_WARP_SIZE;
                    const float add = (c == v_cid) ? exp_new : 0.0f;
                    bet[q][k] = __fmaf_rn(bet[q][k], exp_old, add);
                }

                m_loc[q] = m_new;
            }

            if (use_async_sparse) {
                const int next_ti = ti + 1;
                const int next_t  = token_base + next_ti;
                if (next_ti < CMASS_TOKENS_PER_WARP && next_t < split_end) {
                    cmass_cp_async_wait_all();
                    __syncwarp();
                }
            }
        }
    }

    if (lane < GQA) {
        smem_m[lane][warp_id] = m_loc[lane];
        smem_l[lane][warp_id] = l_loc[lane];
    }
    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            smem_dlt[q][i][warp_id][lane] = dlt[q][i];
        }
        #pragma unroll
        for (int k = 0; k < BETA_SLOTS; ++k) {
            const int c = lane + k * CMASS_WARP_SIZE;
            if (c < NC_V_STATIC) {
                smem_beta[warp_id * BETA_PER_WARP + q * NC_V_STATIC + c] = bet[q][k];
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        float fm = smem_m[q][0];
        for (int w = 1; w < CMASS_WARPS; ++w) {
            fm = fmaxf(fm, smem_m[q][w]);
        }

        float fl = 0.0f;
        float facc[CMASS_DPT];
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) facc[i] = 0.0f;
        #pragma unroll
        for (int w = 0; w < CMASS_WARPS; ++w) {
            const float scale_w = __expf(smem_m[q][w] - fm);
            fl += smem_l[q][w] * scale_w;
            #pragma unroll
            for (int i = 0; i < CMASS_DPT; ++i) {
                facc[i] += smem_dlt[q][i][w][lane] * scale_w;
            }
        }

        const int q_head = h_kv * GQA + q;
        const int flat_q = b * nh + q_head;
        if (lane == 0) {
            partial_m[flat_q * num_splits + split_idx] = fm;
            partial_l[flat_q * num_splits + split_idx] = fl;
        }
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            const int d = lane + i * CMASS_WARP_SIZE;
            partial_delta[(flat_q * num_splits + split_idx) * CMASS_HEAD_DIM + d] =
                facc[i];
        }

        for (int c = lane; c < NC_V_STATIC; c += CMASS_WARP_SIZE) {
            if (c >= nc_v) break;
            float bsum = 0.0f;
            #pragma unroll
            for (int w = 0; w < CMASS_WARPS; ++w) {
                const float scale_w = __expf(smem_m[q][w] - fm);
                bsum += smem_beta[w * BETA_PER_WARP + q * NC_V_STATIC + c] * scale_w;
            }
            partial_beta[(flat_q * num_splits + split_idx) * NC_V_STATIC + c] = bsum;
        }
    }
}

template<int GQA, int NC_V_STATIC, bool INT4_STRUCT = false>
__global__ void __launch_bounds__(CMASS_THREADS, 2)
sparse_fused_kqv_cmass_4x16_struct_sparse_index_partial_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ k_centroids,
    const int8_t* __restrict__ k_cids,
    const void* __restrict__ k_qvalues,
    const int32_t* __restrict__ k_qmeta,
    const __half* __restrict__ k_qscale,
    const int8_t* __restrict__ v_cids,
    const void* __restrict__ v_qvalues,
    const int32_t* __restrict__ v_qmeta,
    const __half* __restrict__ v_qscale,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    float* __restrict__ partial_beta,
    float* __restrict__ partial_delta,
    int B, int nh, int nh_kv, int T,
    int nc_k, int nc_v,
    int start_pos, int pos_q, float rope_base, float scaling,
    int num_splits, const float* inv_freq)
{
    constexpr bool kFixedShift =
        (CMASS_INT4_FIXED_SHIFT_SOFTMAX != 0) && INT4_STRUCT;
    const int split_idx = blockIdx.x;
    const int h_kv      = blockIdx.y;
    const int b         = blockIdx.z;
    const int kv_idx    = b * nh_kv + h_kv;
    const int tid       = threadIdx.x;
    const int warp_id   = tid / CMASS_WARP_SIZE;
    const int lane      = tid % CMASS_WARP_SIZE;
    const int lane_bit  = lane & 15;
    const int upper_half = lane >> 4;
    // Hoist the constant per-token RoPE step out of the token-tile loop.
    // These frequencies match freq0/freq1 below; duplicating the two block-
    // initialization powf operations is far cheaper than two sincosf calls
    // for every tile in a long decode.
    const float step_freq0 =
        (inv_freq ? inv_freq[lane] : 1.0f / powf(rope_base, static_cast<float>(lane * 2) / CMASS_HEAD_DIM));
    const float step_freq1 = (inv_freq ? inv_freq[lane+32] : 1.0f / powf(rope_base, static_cast<float>((lane + 32) * 2) / CMASS_HEAD_DIM));
    float sin_step0, cos_step0, sin_step1, cos_step1;
    __sincosf(-step_freq0, &sin_step0, &cos_step0);
    __sincosf(-step_freq1, &sin_step1, &cos_step1);

    const int split_start = (split_idx * T) / num_splits;
    const int split_end   = ((split_idx + 1) * T) / num_splits;

    extern __shared__ char smem_raw[];
    __half* smem_k_cent   = reinterpret_cast<__half*>(smem_raw);
    __half* smem_k_sparse = smem_k_cent + nc_k * CMASS_HEAD_DIM;
    constexpr int kStagedElemsPerToken =
        (INT4_STRUCT && CMASS_INT4_DIRECT_PACKED) ? 0 : 32;
    __half* smem_v_sparse = smem_k_sparse
        + CMASS_WARPS * CMASS_TOKENS_PER_WARP * kStagedElemsPerToken;
    float* smem_beta      = reinterpret_cast<float*>(
        smem_v_sparse + CMASS_WARPS * CMASS_TOKENS_PER_WARP * kStagedElemsPerToken);
    constexpr int BETA_PER_WARP = GQA * NC_V_STATIC;
    constexpr int BETA_SLOTS    = (NC_V_STATIC + CMASS_WARP_SIZE - 1)
                                   / CMASS_WARP_SIZE;

    __shared__ int smem_kcid_md[CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ int smem_vcid_md[CMASS_WARPS][CMASS_TOKENS_PER_WARP];
    __shared__ float smem_m  [GQA][CMASS_WARPS];
    __shared__ float smem_l  [GQA][CMASS_WARPS];
    __shared__ float smem_dlt[GQA][CMASS_DPT][CMASS_WARPS][CMASS_WARP_SIZE];

    const __half* k_cent_base = k_centroids + kv_idx * nc_k * CMASS_HEAD_DIM;
    for (int idx = tid; idx < nc_k * CMASS_HEAD_DIM; idx += CMASS_THREADS) {
        smem_k_cent[idx] = __ldg(&k_cent_base[idx]);
    }
    __syncthreads();

    float q_reg[GQA][CMASS_DPT];
    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        const int q_head = h_kv * GQA + q;
        const int q_off  = (b * nh + q_head) * CMASS_HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            q_reg[q][i] = __half2float(
                __ldg(&Q[q_off + lane + i * CMASS_WARP_SIZE]));
        }
    }
    const float freq0 =
        (inv_freq ? inv_freq[lane] : 1.0f / powf(rope_base, static_cast<float>(lane * 2) / CMASS_HEAD_DIM));
    const float freq1 = (inv_freq ? inv_freq[lane+32] : 1.0f / powf(rope_base, static_cast<float>((lane + 32) * 2) / CMASS_HEAD_DIM));

    float m_loc[GQA];
    float l_loc[GQA];
    float dlt  [GQA][CMASS_DPT];
    float bet  [GQA][BETA_SLOTS];
    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        m_loc[q] = kFixedShift ? 0.0f : -FLT_MAX;
        l_loc[q] = 0.0f;
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            dlt[q][i] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < BETA_SLOTS; ++k) {
            bet[q][k] = 0.0f;
        }
    }

    for (int token_base = split_start + warp_id * CMASS_TOKENS_PER_WARP;
         token_base < split_end;
         token_base += CMASS_WARPS * CMASS_TOKENS_PER_WARP) {

        if (lane < CMASS_TOKENS_PER_WARP) {
            const int ti = lane;
            const int t  = token_base + ti;
            if (t < split_end) {
                const int off = kv_idx * T + t;
                smem_kcid_md[warp_id][ti] = static_cast<int>(__ldg(&k_cids[off]));
                smem_vcid_md[warp_id][ti] = static_cast<int>(__ldg(&v_cids[off]));
            }
        }

        #pragma unroll
        for (int pre_ti = 0; pre_ti < CMASS_TOKENS_PER_WARP; ++pre_ti) {
            const int pre_t = token_base + pre_ti;
            if (pre_t < split_end) {
                const int pre_off = kv_idx * T + pre_t;
                const int pre_base =
                    (warp_id * CMASS_TOKENS_PER_WARP + pre_ti) * 32;
                if constexpr (INT4_STRUCT) {
                    if constexpr (!CMASS_INT4_DIRECT_PACKED) {
                        cmass_stage_int_row_scalar(
                            reinterpret_cast<int32_t*>(smem_k_sparse + pre_base),
                            reinterpret_cast<const int32_t*>(k_qvalues) + pre_off * 4,
                            4, lane);
                        cmass_stage_int_row_scalar(
                            reinterpret_cast<int32_t*>(smem_v_sparse + pre_base),
                            reinterpret_cast<const int32_t*>(v_qvalues) + pre_off * 4,
                            4, lane);
                    }
                } else {
                    cmass_stage_half_row_async(
                        smem_k_sparse + pre_base,
                        reinterpret_cast<const __half*>(k_qvalues) + pre_off * 32,
                        32, lane);
                    cmass_stage_half_row_async(
                        smem_v_sparse + pre_base,
                        reinterpret_cast<const __half*>(v_qvalues) + pre_off * 32,
                        32, lane);
                }
            }
        }
        if constexpr (!INT4_STRUCT || !CMASS_INT4_DIRECT_PACKED) {
            cmass_cp_async_commit();
            cmass_cp_async_wait_all();
        }
        __syncwarp();

        const int delta_base = pos_q - (start_pos + token_base);
        float sin0_cur, cos0_cur, sin1_cur, cos1_cur;
        __sincosf(static_cast<float>(delta_base) * freq0,
                  &sin0_cur, &cos0_cur);
        __sincosf(static_cast<float>(delta_base) * freq1,
                  &sin1_cur, &cos1_cur);

        #pragma unroll
        for (int ti = 0; ti < CMASS_TOKENS_PER_WARP; ++ti) {
            const int t = token_base + ti;
            if (t >= split_end) continue;
            const int off = kv_idx * T + t;
            const int sparse_base = (warp_id * CMASS_TOKENS_PER_WARP + ti) * 32;

            const int k_cid = smem_kcid_md[warp_id][ti];
            const int v_cid = smem_vcid_md[warp_id][ti];
            const __half* k_cent_row = smem_k_cent + k_cid * CMASS_HEAD_DIM;
            const __half* k_sparse_row = smem_k_sparse + sparse_base;
            const __half* v_sparse_row = smem_v_sparse + sparse_base;
            const int32_t* k_packed_row = reinterpret_cast<const int32_t*>(k_sparse_row);
            const int32_t* v_packed_row = reinterpret_cast<const int32_t*>(v_sparse_row);
            float k_scale_val = 1.0f;
            float v_scale_val = 1.0f;
            if constexpr (INT4_STRUCT) {
                if (lane == 0) {
                    k_scale_val = __half2float(__ldg(&k_qscale[off]));
                    v_scale_val = __half2float(__ldg(&v_qscale[off]));
                }
                k_scale_val = __shfl_sync(0xffffffff, k_scale_val, 0);
                v_scale_val = __shfl_sync(0xffffffff, v_scale_val, 0);
            }

            uint32_t k_word_lane = 0;
            uint32_t v_word_lane = 0;
            if (lane < (INT4_STRUCT ? 2 : 4)) {
                const int meta_words = INT4_STRUCT ? 2 : 4;
                k_word_lane = static_cast<uint32_t>(__ldg(&k_qmeta[off * meta_words + lane]));
                v_word_lane = static_cast<uint32_t>(__ldg(&v_qmeta[off * meta_words + lane]));
            }
            const uint32_t k_words[CMASS_DPT] = {
                __shfl_sync(0xffffffff, k_word_lane, 0),
                __shfl_sync(0xffffffff, k_word_lane, 1),
                __shfl_sync(0xffffffff, k_word_lane, 2),
                __shfl_sync(0xffffffff, k_word_lane, 3),
            };
            const uint32_t v_words[CMASS_DPT] = {
                __shfl_sync(0xffffffff, v_word_lane, 0),
                __shfl_sync(0xffffffff, v_word_lane, 1),
                __shfl_sync(0xffffffff, v_word_lane, 2),
                __shfl_sync(0xffffffff, v_word_lane, 3),
            };

            uint32_t k_payload_lane = 0;
            uint32_t v_payload_lane = 0;
            if constexpr (INT4_STRUCT && CMASS_INT4_DIRECT_PACKED) {
                if (lane < 4) {
                    k_payload_lane = static_cast<uint32_t>(__ldg(
                        reinterpret_cast<const int32_t*>(k_qvalues) + off * 4 + lane));
                    v_payload_lane = static_cast<uint32_t>(__ldg(
                        reinterpret_cast<const int32_t*>(v_qvalues) + off * 4 + lane));
                }
            }
            const uint32_t k_payload_words[CMASS_DPT] = {
                __shfl_sync(0xffffffff, k_payload_lane, 0),
                __shfl_sync(0xffffffff, k_payload_lane, 1),
                __shfl_sync(0xffffffff, k_payload_lane, 2),
                __shfl_sync(0xffffffff, k_payload_lane, 3),
            };
            const uint32_t v_payload_words[CMASS_DPT] = {
                __shfl_sync(0xffffffff, v_payload_lane, 0),
                __shfl_sync(0xffffffff, v_payload_lane, 1),
                __shfl_sync(0xffffffff, v_payload_lane, 2),
                __shfl_sync(0xffffffff, v_payload_lane, 3),
            };

            float k_val[CMASS_DPT];
            float v_res[CMASS_DPT];
            #pragma unroll
            for (int i = 0; i < CMASS_DPT; ++i) {
                const int d = lane + i * CMASS_WARP_SIZE;
                const int qtile = (i << 1) + upper_half;
                const uint32_t k_code = INT4_STRUCT
                    ? k_words[i >> 1]
                    : (upper_half ? (k_words[i] >> 16) : (k_words[i] & 0xffffu));
                const uint32_t v_code = INT4_STRUCT
                    ? v_words[i >> 1]
                    : (upper_half ? (v_words[i] >> 16) : (v_words[i] & 0xffffu));
                float kres;
                float vres;
                if constexpr (INT4_STRUCT && CMASS_INT4_DIRECT_PACKED) {
                    const int group_in_word = (d >> 2) & 15;
                    const int k_selected = static_cast<int>(
                        (k_code >> (group_in_word * 2)) & 3u);
                    const int v_selected = static_cast<int>(
                        (v_code >> (group_in_word * 2)) & 3u);
                    const int nibble_shift = ((d >> 2) & 7) * 4;
                    const int k_nibble = static_cast<int>(
                        (k_payload_words[i] >> nibble_shift) & 0xfu);
                    const int v_nibble = static_cast<int>(
                        (v_payload_words[i] >> nibble_shift) & 0xfu);
                    kres = (k_selected == (d & 3))
                        ? static_cast<float>((k_nibble ^ 8) - 8) * k_scale_val
                        : 0.0f;
                    vres = (v_selected == (d & 3))
                        ? static_cast<float>((v_nibble ^ 8) - 8) * v_scale_val
                        : 0.0f;
                } else if constexpr (INT4_STRUCT) {
                    kres = cmass_load_int4_4x16_struct_index_row(
                        k_packed_row, k_code, d, k_scale_val);
                    vres = cmass_load_int4_4x16_struct_index_row(
                        v_packed_row, v_code, d, v_scale_val);
                } else {
                    kres = cmass_load_4x16_struct_index_row(
                        k_sparse_row, k_code, qtile, lane_bit);
                    vres = cmass_load_4x16_struct_index_row(
                        v_sparse_row, v_code, qtile, lane_bit);
                }
                k_val[i] = __half2float(k_cent_row[d]) + kres;
                v_res[i] = vres;
            }

            float k_rope[CMASS_DPT];
            if constexpr (INT4_STRUCT && CMASS_INT4_ROTATE_K_ONCE && GQA > 1) {
                k_rope[0] = __fmaf_rn(k_val[0], cos0_cur,  k_val[2] * sin0_cur);
                k_rope[2] = __fmaf_rn(k_val[2], cos0_cur, -k_val[0] * sin0_cur);
                k_rope[1] = __fmaf_rn(k_val[1], cos1_cur,  k_val[3] * sin1_cur);
                k_rope[3] = __fmaf_rn(k_val[3], cos1_cur, -k_val[1] * sin1_cur);
            }

            #pragma unroll
            for (int q = 0; q < GQA; ++q) {
                float dot = 0.0f;
                if constexpr (INT4_STRUCT && CMASS_INT4_ROTATE_K_ONCE && GQA > 1) {
                    dot = __fmaf_rn(q_reg[q][0], k_rope[0], dot);
                    dot = __fmaf_rn(q_reg[q][1], k_rope[1], dot);
                    dot = __fmaf_rn(q_reg[q][2], k_rope[2], dot);
                    dot = __fmaf_rn(q_reg[q][3], k_rope[3], dot);
                } else {
                    dot = __fmaf_rn(q_reg[q][0] * cos0_cur - q_reg[q][2] * sin0_cur,
                                    k_val[0], dot);
                    dot = __fmaf_rn(q_reg[q][1] * cos1_cur - q_reg[q][3] * sin1_cur,
                                    k_val[1], dot);
                    dot = __fmaf_rn(q_reg[q][0] * sin0_cur + q_reg[q][2] * cos0_cur,
                                    k_val[2], dot);
                    dot = __fmaf_rn(q_reg[q][1] * sin1_cur + q_reg[q][3] * cos1_cur,
                                    k_val[3], dot);
                }
                #pragma unroll
                for (int off2 = 16; off2 > 0; off2 >>= 1) {
                    dot += __shfl_xor_sync(0xffffffff, dot, off2);
                }
                const float score   = dot * scaling;
#if CMASS_INT4_FIXED_SHIFT_SOFTMAX
                if constexpr (INT4_STRUCT) {
                    const float weight = __expf(score);
                    l_loc[q] += weight;
                    #pragma unroll
                    for (int i = 0; i < CMASS_DPT; ++i) {
                        dlt[q][i] = __fmaf_rn(weight, v_res[i], dlt[q][i]);
                    }
                    #pragma unroll
                    for (int k = 0; k < BETA_SLOTS; ++k) {
                        const int c = lane + k * CMASS_WARP_SIZE;
                        if (c == v_cid) bet[q][k] += weight;
                    }
                } else
#endif
                {
                const float m_new   = fmaxf(m_loc[q], score);
                const float exp_old = __expf(m_loc[q] - m_new);
                const float exp_new = __expf(score - m_new);

                l_loc[q] = __fmaf_rn(l_loc[q], exp_old, exp_new);
                #pragma unroll
                for (int i = 0; i < CMASS_DPT; ++i) {
                    dlt[q][i] = __fmaf_rn(dlt[q][i], exp_old,
                                           exp_new * v_res[i]);
                }
                #pragma unroll
                for (int k = 0; k < BETA_SLOTS; ++k) {
                    const int c = lane + k * CMASS_WARP_SIZE;
                    const float add = (c == v_cid) ? exp_new : 0.0f;
                    bet[q][k] = __fmaf_rn(bet[q][k], exp_old, add);
                }

                m_loc[q] = m_new;
                }
            }

            const float next_sin0 = __fmaf_rn(sin0_cur, cos_step0, cos0_cur * sin_step0);
            const float next_cos0 = __fmaf_rn(cos0_cur, cos_step0, -sin0_cur * sin_step0);
            const float next_sin1 = __fmaf_rn(sin1_cur, cos_step1, cos1_cur * sin_step1);
            const float next_cos1 = __fmaf_rn(cos1_cur, cos_step1, -sin1_cur * sin_step1);
            sin0_cur = next_sin0;
            cos0_cur = next_cos0;
            sin1_cur = next_sin1;
            cos1_cur = next_cos1;
        }

    }

    if (lane < GQA) {
        if constexpr (!kFixedShift) {
            smem_m[lane][warp_id] = m_loc[lane];
        }
        smem_l[lane][warp_id] = l_loc[lane];
    }
    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            smem_dlt[q][i][warp_id][lane] = dlt[q][i];
        }
        #pragma unroll
        for (int k = 0; k < BETA_SLOTS; ++k) {
            const int c = lane + k * CMASS_WARP_SIZE;
            if (c < NC_V_STATIC) {
                smem_beta[warp_id * BETA_PER_WARP + q * NC_V_STATIC + c] = bet[q][k];
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < GQA; ++q) {
        float fm = 0.0f;
        if constexpr (!kFixedShift) {
            fm = smem_m[q][0];
            for (int w = 1; w < CMASS_WARPS; ++w) {
                fm = fmaxf(fm, smem_m[q][w]);
            }
        }

        float fl = 0.0f;
        float facc[CMASS_DPT];
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) facc[i] = 0.0f;
        #pragma unroll
        for (int w = 0; w < CMASS_WARPS; ++w) {
            const float scale_w = kFixedShift
                ? 1.0f : __expf(smem_m[q][w] - fm);
            fl += smem_l[q][w] * scale_w;
            #pragma unroll
            for (int i = 0; i < CMASS_DPT; ++i) {
                facc[i] += smem_dlt[q][i][w][lane] * scale_w;
            }
        }

        const int q_head = h_kv * GQA + q;
        const int flat_q = b * nh + q_head;
        if (lane == 0) {
            if constexpr (!kFixedShift) {
                partial_m[flat_q * num_splits + split_idx] = fm;
            }
            partial_l[flat_q * num_splits + split_idx] = fl;
        }
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            const int d = lane + i * CMASS_WARP_SIZE;
            partial_delta[(flat_q * num_splits + split_idx) * CMASS_HEAD_DIM + d] =
                facc[i];
        }

        for (int c = lane; c < NC_V_STATIC; c += CMASS_WARP_SIZE) {
            if (c >= nc_v) break;
            float bsum = 0.0f;
            #pragma unroll
            for (int w = 0; w < CMASS_WARPS; ++w) {
                const float scale_w = kFixedShift
                    ? 1.0f : __expf(smem_m[q][w] - fm);
                bsum += smem_beta[w * BETA_PER_WARP + q * NC_V_STATIC + c] * scale_w;
            }
            partial_beta[(flat_q * num_splits + split_idx) * NC_V_STATIC + c] = bsum;
        }
    }
}

template<int GQA, int NC_V_STATIC>
inline void cmass_set_fp16_partial_smem_attr(size_t smem_size) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sparse_fused_kqv_cmass_fp16_partial_kernel<GQA, NC_V_STATIC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)));
}

template<int GQA, int NC_V_STATIC, bool INT4_STRUCT = false>
inline void cmass_set_4x16_struct_index_partial_smem_attr(size_t smem_size) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sparse_fused_kqv_cmass_4x16_struct_sparse_index_partial_kernel<
            GQA, NC_V_STATIC, INT4_STRUCT>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_size)));
}

template<int GQA, int NC_V_STATIC, bool FIXED_SHIFT = false>
__global__ void __launch_bounds__(GQA * CMASS_WARP_SIZE, 4)
sparse_fused_kqv_cmass_merge_kernel(
    const float*  __restrict__ partial_m,
    const float*  __restrict__ partial_l,
    const float*  __restrict__ partial_beta,
    const float*  __restrict__ partial_delta,
    const __half* __restrict__ v_centroids,
    __half* __restrict__ output,
    float*  __restrict__ final_m,
    float*  __restrict__ final_l,
    int B, int nh, int nh_kv, int num_splits, int nc_v)
{
    const int h_kv    = blockIdx.x;
    const int b       = blockIdx.y;
    const int warp_id = threadIdx.x / CMASS_WARP_SIZE;
    const int lane    = threadIdx.x % CMASS_WARP_SIZE;
    if (warp_id >= GQA) return;

    const int q       = warp_id;
    const int q_head  = h_kv * GQA + q;
    const int flat_q  = b * nh + q_head;
    const int kv_idx  = b * nh_kv + h_kv;

    extern __shared__ float merge_smem[];
    // merge_smem layout: [GQA][NC_V_STATIC] beta_final per warp (q).
    float* beta_q = merge_smem + q * NC_V_STATIC;

    // Step 1: find final m across splits.
    float fm = 0.0f;
    if constexpr (!FIXED_SHIFT) {
        fm = partial_m[flat_q * num_splits];
        for (int s = 1; s < num_splits; ++s) {
            fm = fmaxf(fm, partial_m[flat_q * num_splits + s]);
        }
    }

    // Step 2: build final l + final delta + final beta with online rescale.
    float fl = 0.0f;
    float facc[CMASS_DPT];
    #pragma unroll
    for (int i = 0; i < CMASS_DPT; ++i) facc[i] = 0.0f;
    for (int c = lane; c < nc_v; c += CMASS_WARP_SIZE) {
        beta_q[c] = 0.0f;
    }
    __syncwarp();

    for (int s = 0; s < num_splits; ++s) {
        const int off = flat_q * num_splits + s;
        const float sm = partial_m[off];
        const float sl = partial_l[off];
        const float sc = FIXED_SHIFT ? 1.0f : __expf(sm - fm);
        fl += sl * sc;
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            const int d = lane + i * CMASS_WARP_SIZE;
            facc[i] += partial_delta[off * CMASS_HEAD_DIM + d] * sc;
        }
        for (int c = lane; c < nc_v; c += CMASS_WARP_SIZE) {
            beta_q[c] += partial_beta[off * NC_V_STATIC + c] * sc;
        }
    }
    __syncwarp();

    // Step 3: fold V centroids weighted by beta into facc.
    const __half* v_cent_base = v_centroids + kv_idx * nc_v * CMASS_HEAD_DIM;
    for (int c = 0; c < nc_v; ++c) {
        const float bc = beta_q[c];
        if (bc == 0.0f) continue;
        const __half* v_row = v_cent_base + c * CMASS_HEAD_DIM;
        #pragma unroll
        for (int i = 0; i < CMASS_DPT; ++i) {
            const int d = lane + i * CMASS_WARP_SIZE;
            const float vc = __half2float(__ldg(&v_row[d]));
            facc[i] = __fmaf_rn(bc, vc, facc[i]);
        }
    }

    const float inv = 1.0f / fmaxf(fl, 1e-20f);
    #pragma unroll
    for (int i = 0; i < CMASS_DPT; ++i) {
        const int d = lane + i * CMASS_WARP_SIZE;
        output[flat_q * CMASS_HEAD_DIM + d] = __float2half(facc[i] * inv);
    }
    if (lane == 0) {
        if (final_m != nullptr) final_m[flat_q] = fm;
        if (final_l != nullptr) final_l[flat_q] = fl;
    }
}

template<int GQA>
__global__ void __launch_bounds__(CMASS_HEAD_DIM, 4)
sparse_fused_kqv_cmass_merge_recent_kernel(
    const __half* __restrict__ comp_out,
    const float*  __restrict__ comp_m,
    const float*  __restrict__ comp_l,
    const float*  __restrict__ scores_recent,
    const __half* __restrict__ v_recent,
    __half* __restrict__ output,
    int B, int nh, int nh_kv, int T_recent)
{
    const int q_head = blockIdx.x;
    const int b      = blockIdx.y;
    const int tid    = threadIdx.x;
    const int h_kv   = q_head / GQA;
    const int flat_q = b * nh + q_head;

    __shared__ float sh[CMASS_HEAD_DIM];
    const float* scores = scores_recent + (size_t)flat_q * T_recent;

    float local_m = -FLT_MAX;
    for (int t = tid; t < T_recent; t += CMASS_HEAD_DIM) {
        local_m = fmaxf(local_m, __ldg(&scores[t]));
    }
    sh[tid] = local_m;
    __syncthreads();
    for (int off = CMASS_HEAD_DIM >> 1; off > 0; off >>= 1) {
        if (tid < off) sh[tid] = fmaxf(sh[tid], sh[tid + off]);
        __syncthreads();
    }
    const float recent_m = sh[0];

    float local_l = 0.0f;
    for (int t = tid; t < T_recent; t += CMASS_HEAD_DIM) {
        local_l += __expf(__ldg(&scores[t]) - recent_m);
    }
    sh[tid] = local_l;
    __syncthreads();
    for (int off = CMASS_HEAD_DIM >> 1; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }
    const float recent_l = sh[0];

    const float cm = __ldg(&comp_m[flat_q]);
    const float cl = __ldg(&comp_l[flat_q]);
    const float joint_m = fmaxf(cm, recent_m);
    const float comp_w  = cl * __expf(cm - joint_m);
    const float rec_w   = __expf(recent_m - joint_m);
    const float denom   = fmaxf(comp_w + recent_l * rec_w, 1e-20f);

    const int d = tid;
    float rec_acc = 0.0f;
    const __half* v_base = v_recent +
        ((size_t)b * nh_kv + h_kv) * T_recent * CMASS_HEAD_DIM + d;
    for (int t = 0; t < T_recent; ++t) {
        const float w = __expf(__ldg(&scores[t]) - recent_m);
        rec_acc = __fmaf_rn(w,
            __half2float(__ldg(&v_base[t * CMASS_HEAD_DIM])), rec_acc);
    }
    const float comp_v = __half2float(__ldg(&comp_out[flat_q * CMASS_HEAD_DIM + d]));
    output[flat_q * CMASS_HEAD_DIM + d] = __float2half(
        (comp_v * comp_w + rec_acc * rec_w) / denom);
}

} // namespace

// ============================================================================
// Host launchers
// ============================================================================

std::vector<torch::Tensor> sparse_fused_kqv_cmass_impl(
    torch::Tensor Q,
    torch::Tensor k_centroids, torch::Tensor k_cluster_ids,
    torch::Tensor k_packed,    torch::Tensor k_bitmap,
    torch::Tensor k_scale,     torch::Tensor k_zero,
    torch::Tensor v_centroids, torch::Tensor v_cluster_ids,
    torch::Tensor v_packed,    torch::Tensor v_bitmap,
    torch::Tensor v_scale,     torch::Tensor v_zero,
    int num_kv_heads, int num_q_heads,
    int start_pos, int pos_q, double rope_base, double scaling,
    int k_bitwidth, int v_bitwidth,
    bool return_stats, int splits, torch::Tensor inv_freq)
{
    c10::cuda::CUDAGuard guard(Q.device());
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat16, "Q must be fp16 CUDA");
    const int B_nh = Q.size(0);
    const int D    = Q.size(1);
    TORCH_CHECK(D == CMASS_HEAD_DIM, "cmass only supports head_dim=128");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "num_q_heads must be divisible by num_kv_heads");
    const int B    = B_nh / num_q_heads;
    const int T    = k_cluster_ids.size(1);
    const int nc_k = k_centroids.size(1);
    const int nc_v = v_centroids.size(1);
    const int gqa  = num_q_heads / num_kv_heads;
    const int k_pw = k_packed.size(2);
    const int v_pw = v_packed.size(2);
    const bool fp16_sparse = true;
    TORCH_CHECK(k_bitwidth == 16 && v_bitwidth == 16, "Bitmap family requires FP16 residuals");
    const int beta_stride = (nc_v <= 32) ? 32 : CMASS_MAX_NC;
    TORCH_CHECK(gqa == 1 || gqa == 4 || gqa == 8,
                "cmass fused KQV supports GQA in {1,4,8}, got ", gqa);
    TORCH_CHECK(nc_k <= CMASS_MAX_NC && nc_v <= CMASS_MAX_NC,
                "cmass fused KQV: clusters exceed ", CMASS_MAX_NC);
    TORCH_CHECK(
        fp16_sparse || (k_bitwidth == 2 || k_bitwidth == 4),
        "cmass fused KQV: k_bitwidth must be 2, 4, or sparse-only 16");
    TORCH_CHECK(
        fp16_sparse || (v_bitwidth == 2 || v_bitwidth == 4),
        "cmass fused KQV: v_bitwidth must be 2, 4, or sparse-only 16");
    TORCH_CHECK(
        !((k_bitwidth == 16) ^ (v_bitwidth == 16)),
        "cmass fused KQV sparse-only requires both K and V bitwidth=16");
    if (fp16_sparse) {
        TORCH_CHECK(k_packed.scalar_type() == torch::kFloat16,
                    "cmass fp16 sparse-only: k_sparse must be fp16");
        TORCH_CHECK(v_packed.scalar_type() == torch::kFloat16,
                    "cmass fp16 sparse-only: v_sparse must be fp16");
        TORCH_CHECK(k_pw <= CMASS_MAX_NNZ && v_pw <= CMASS_MAX_NNZ,
                    "cmass fp16 sparse-only: nnz exceeds ", CMASS_MAX_NNZ);
    } else {
        TORCH_CHECK(k_packed.scalar_type() == torch::kInt32,
                    "cmass fused KQV: k_packed must be int32 for 2/4-bit");
        TORCH_CHECK(v_packed.scalar_type() == torch::kInt32,
                    "cmass fused KQV: v_packed must be int32 for 2/4-bit");
        TORCH_CHECK(k_pw <= CMASS_MAX_PW && v_pw <= CMASS_MAX_PW,
                    "cmass fused KQV: packed words exceed ", CMASS_MAX_PW);
    }

    const int num_splits = splits > 0 ? std::min(splits,T) : cmass_resolve_num_splits(T,B);
    auto float_opts = Q.options().dtype(torch::kFloat32);
    auto partial_m  = torch::empty({B_nh, num_splits},                      float_opts);
    auto partial_l  = torch::empty({B_nh, num_splits},                      float_opts);
    auto partial_beta  = torch::empty({B_nh, num_splits, beta_stride},      float_opts);
    auto partial_delta = torch::empty({B_nh, num_splits, CMASS_HEAD_DIM},   float_opts);
    auto output     = torch::empty({B_nh, CMASS_HEAD_DIM},                  Q.options());
    torch::Tensor final_m, final_l;
    if (return_stats) {
        final_m = torch::empty({B_nh}, float_opts);
        final_l = torch::empty({B_nh}, float_opts);
    }
    float* fm_ptr = return_stats ? final_m.data_ptr<float>() : nullptr;
    float* fl_ptr = return_stats ? final_l.data_ptr<float>() : nullptr;

    // Compute beta smem size based on GQA.  Use compile-time constant matching
    // the dispatched template so the partial kernel sees the correct stride.
    const size_t beta_smem_per_warp = static_cast<size_t>(gqa) * beta_stride * sizeof(float);
    const size_t smem_size = fp16_sparse
        ? (nc_k * CMASS_HEAD_DIM * sizeof(__half) +
           2 * CMASS_WARPS * CMASS_TOKENS_PER_WARP * CMASS_MAX_NNZ * sizeof(__half) +
           CMASS_WARPS * beta_smem_per_warp)
        : (nc_k * CMASS_HEAD_DIM * sizeof(__half) +
           2 * CMASS_WARPS * CMASS_TOKENS_PER_WARP * CMASS_MAX_PW * sizeof(int32_t) +
           CMASS_WARPS * beta_smem_per_warp);

    dim3 partial_grid(num_splits, num_kv_heads, B);
    dim3 partial_block(CMASS_THREADS);

    if (fp16_sparse) {
#define CMASS_LAUNCH_FP16_PARTIAL(GQA_VAL, NC_VAL) \
        do { \
            cmass_set_fp16_partial_smem_attr<GQA_VAL, NC_VAL>(smem_size); \
            sparse_fused_kqv_cmass_fp16_partial_kernel<GQA_VAL, NC_VAL> \
                <<<partial_grid, partial_block, smem_size, at::cuda::getCurrentCUDAStream()>>>( \
                    reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), \
                    reinterpret_cast<const __half*>(k_centroids.data_ptr<at::Half>()), \
                    k_cluster_ids.data_ptr<int8_t>(), \
                    reinterpret_cast<const __half*>(k_packed.data_ptr<at::Half>()), \
                    k_bitmap.data_ptr<int64_t>(), \
                    v_cluster_ids.data_ptr<int8_t>(), \
                    reinterpret_cast<const __half*>(v_packed.data_ptr<at::Half>()), \
                    v_bitmap.data_ptr<int64_t>(), \
                    partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), \
                    partial_beta.data_ptr<float>(), partial_delta.data_ptr<float>(), \
                    B, num_q_heads, num_kv_heads, T, nc_k, nc_v, k_pw, v_pw, \
                    start_pos, pos_q, static_cast<float>(rope_base), \
                    static_cast<float>(scaling), num_splits, inv_freq.numel() ? inv_freq.data_ptr<float>() : nullptr); \
        } while (0)
        if (beta_stride == 32) {
            if      (gqa == 1) { CMASS_LAUNCH_FP16_PARTIAL(1, 32); }
            else if (gqa == 4) { CMASS_LAUNCH_FP16_PARTIAL(4, 32); }
            else               { CMASS_LAUNCH_FP16_PARTIAL(8, 32); }
        } else {
            if      (gqa == 1) { CMASS_LAUNCH_FP16_PARTIAL(1, CMASS_MAX_NC); }
            else if (gqa == 4) { CMASS_LAUNCH_FP16_PARTIAL(4, CMASS_MAX_NC); }
            else               { CMASS_LAUNCH_FP16_PARTIAL(8, CMASS_MAX_NC); }
        }
#undef CMASS_LAUNCH_FP16_PARTIAL
    }
    cmass_check_launch("sparse_fused_kqv_cmass_partial");

    dim3 merge_grid(num_kv_heads, B);
    const size_t merge_smem = static_cast<size_t>(gqa) * beta_stride * sizeof(float);
#define CMASS_LAUNCH_MERGE(GQA_VAL, NC_VAL) \
    do { \
        dim3 merge_block(GQA_VAL * CMASS_WARP_SIZE); \
        sparse_fused_kqv_cmass_merge_kernel<GQA_VAL, NC_VAL> \
            <<<merge_grid, merge_block, merge_smem, at::cuda::getCurrentCUDAStream()>>>( \
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), \
                partial_beta.data_ptr<float>(), partial_delta.data_ptr<float>(), \
                reinterpret_cast<const __half*>(v_centroids.data_ptr<at::Half>()), \
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
                fm_ptr, fl_ptr, B, num_q_heads, num_kv_heads, num_splits, nc_v); \
    } while (0)
    if (beta_stride == 32) {
        if      (gqa == 1) { CMASS_LAUNCH_MERGE(1, 32); }
        else if (gqa == 4) { CMASS_LAUNCH_MERGE(4, 32); }
        else               { CMASS_LAUNCH_MERGE(8, 32); }
    } else {
        if      (gqa == 1) { CMASS_LAUNCH_MERGE(1, CMASS_MAX_NC); }
        else if (gqa == 4) { CMASS_LAUNCH_MERGE(4, CMASS_MAX_NC); }
        else               { CMASS_LAUNCH_MERGE(8, CMASS_MAX_NC); }
    }
#undef CMASS_LAUNCH_MERGE
    cmass_check_launch("sparse_fused_kqv_cmass_merge");

    if (return_stats) return {output, final_m, final_l};
    return {output};
}

template<bool INT4_STRUCT = false>
std::vector<torch::Tensor> sparse_fused_kqv_cmass_4x16_struct_sparse_index_impl(
    torch::Tensor Q,
    torch::Tensor k_centroids, torch::Tensor k_cluster_ids,
    torch::Tensor k_qvalues,   torch::Tensor k_qmeta, torch::Tensor k_qscale,
    torch::Tensor v_centroids, torch::Tensor v_cluster_ids,
    torch::Tensor v_qvalues,   torch::Tensor v_qmeta, torch::Tensor v_qscale,
    int num_kv_heads, int num_q_heads,
    int start_pos, int pos_q, double rope_base, double scaling,
    bool return_stats, int splits, torch::Tensor inv_freq)
{
    c10::cuda::CUDAGuard guard(Q.device());
    TORCH_CHECK(Q.is_cuda() && Q.dtype() == torch::kFloat16, "Q must be fp16 CUDA");
    TORCH_CHECK(k_centroids.scalar_type() == torch::kFloat16, "k_centroids must be fp16");
    TORCH_CHECK(v_centroids.scalar_type() == torch::kFloat16, "v_centroids must be fp16");
    TORCH_CHECK(k_cluster_ids.scalar_type() == torch::kInt8, "k_cluster_ids must be int8");
    TORCH_CHECK(v_cluster_ids.scalar_type() == torch::kInt8, "v_cluster_ids must be int8");
    TORCH_CHECK(k_qvalues.scalar_type() == (INT4_STRUCT ? torch::kInt32 : torch::kFloat16),
                "K structured payload dtype mismatch");
    TORCH_CHECK(v_qvalues.scalar_type() == (INT4_STRUCT ? torch::kInt32 : torch::kFloat16),
                "V structured payload dtype mismatch");
    if constexpr (INT4_STRUCT) {
        TORCH_CHECK(k_qscale.scalar_type() == torch::kFloat16 && v_qscale.scalar_type() == torch::kFloat16,
                    "INT4 structured scales must be fp16");
    }
    TORCH_CHECK(k_qmeta.scalar_type() == torch::kInt32, "k_qmeta must be int32");
    TORCH_CHECK(v_qmeta.scalar_type() == torch::kInt32, "v_qmeta must be int32");

    const int B_nh = Q.size(0);
    const int D    = Q.size(1);
    TORCH_CHECK(D == CMASS_HEAD_DIM, "cmass 4:16 struct sparse index only supports head_dim=128");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "num_q_heads must be divisible by num_kv_heads");
    const int B    = B_nh / num_q_heads;
    const int T    = k_cluster_ids.size(1);
    const int nc_k = k_centroids.size(1);
    const int nc_v = v_centroids.size(1);
    const int gqa  = num_q_heads / num_kv_heads;
    const int beta_stride = (nc_v <= 32) ? 32 : CMASS_MAX_NC;
    TORCH_CHECK(gqa == 1 || gqa == 4 || gqa == 8,
                "cmass fused KQV supports GQA in {1,4,8}, got ", gqa);
    TORCH_CHECK(nc_k <= CMASS_MAX_NC && nc_v <= CMASS_MAX_NC,
                "cmass fused KQV: clusters exceed ", CMASS_MAX_NC);
    TORCH_CHECK(k_qvalues.dim() == 3 && v_qvalues.dim() == 3,
                "4:16 struct sparse index values must be [B*nh_kv, T, 32]");
    TORCH_CHECK(k_qmeta.dim() == 3 && k_qmeta.size(2) == (INT4_STRUCT ? 2 : 4),
                "K structured metadata width mismatch");
    TORCH_CHECK(v_qmeta.dim() == 3 && v_qmeta.size(2) == (INT4_STRUCT ? 2 : 4),
                "V structured metadata width mismatch");
    const int payload_words = INT4_STRUCT ? 4 : 32;
    TORCH_CHECK(k_qvalues.size(2) == payload_words && v_qvalues.size(2) == payload_words,
                "4:16 structured payload width mismatch");
    TORCH_CHECK(k_centroids.size(0) == B * num_kv_heads, "K centroid batch-head mismatch");
    TORCH_CHECK(v_centroids.size(0) == B * num_kv_heads, "V centroid batch-head mismatch");
    TORCH_CHECK(k_cluster_ids.size(0) == B * num_kv_heads && k_cluster_ids.size(1) == T,
                "K cid shape mismatch");
    TORCH_CHECK(v_cluster_ids.size(0) == B * num_kv_heads && v_cluster_ids.size(1) == T,
                "V cid shape mismatch");
    TORCH_CHECK(k_qvalues.size(0) == B * num_kv_heads && k_qvalues.size(1) == T,
                "K 4:16 struct sparse index value shape mismatch");
    TORCH_CHECK(v_qvalues.size(0) == B * num_kv_heads && v_qvalues.size(1) == T,
                "V 4:16 struct sparse index value shape mismatch");
    if constexpr (INT4_STRUCT) {
        TORCH_CHECK(k_qscale.dim() == 2 && k_qscale.size(0) == B * num_kv_heads && k_qscale.size(1) == T,
                    "K INT4 scale shape mismatch");
        TORCH_CHECK(v_qscale.dim() == 2 && v_qscale.size(0) == B * num_kv_heads && v_qscale.size(1) == T,
                    "V INT4 scale shape mismatch");
    }
    TORCH_CHECK(k_qmeta.size(0) == B * num_kv_heads && k_qmeta.size(1) == T,
                "K 4:16 struct sparse index meta shape mismatch");
    TORCH_CHECK(v_qmeta.size(0) == B * num_kv_heads && v_qmeta.size(1) == T,
                "V 4:16 struct sparse index meta shape mismatch");

    const int num_splits = splits > 0 ? std::min(splits,T) : cmass_resolve_num_splits(T,B);
    auto float_opts = Q.options().dtype(torch::kFloat32);
    auto partial_m  = torch::empty({B_nh, num_splits},                    float_opts);
    auto partial_l  = torch::empty({B_nh, num_splits},                    float_opts);
    auto partial_beta  = torch::empty({B_nh, num_splits, beta_stride},    float_opts);
    auto partial_delta = torch::empty({B_nh, num_splits, CMASS_HEAD_DIM}, float_opts);
    auto output     = torch::empty({B_nh, CMASS_HEAD_DIM},                Q.options());
    torch::Tensor final_m, final_l;
    if (return_stats) {
        final_m = torch::empty({B_nh}, float_opts);
        final_l = torch::empty({B_nh}, float_opts);
    }
    float* fm_ptr = return_stats ? final_m.data_ptr<float>() : nullptr;
    float* fl_ptr = return_stats ? final_l.data_ptr<float>() : nullptr;

    const size_t beta_smem_per_warp = static_cast<size_t>(gqa) * beta_stride * sizeof(float);
    const size_t smem_size =
        nc_k * CMASS_HEAD_DIM * sizeof(__half) +
        ((INT4_STRUCT && CMASS_INT4_DIRECT_PACKED)
             ? 0
             : 2 * CMASS_WARPS * CMASS_TOKENS_PER_WARP * 32 * sizeof(__half)) +
        CMASS_WARPS * beta_smem_per_warp;

    dim3 partial_grid(num_splits, num_kv_heads, B);
    dim3 partial_block(CMASS_THREADS);

#define CMASS_LAUNCH_4X16_STRUCT_PARTIAL(GQA_VAL, NC_VAL) \
    do { \
        cmass_set_4x16_struct_index_partial_smem_attr<GQA_VAL, NC_VAL, INT4_STRUCT>(smem_size); \
        sparse_fused_kqv_cmass_4x16_struct_sparse_index_partial_kernel<GQA_VAL, NC_VAL, INT4_STRUCT> \
            <<<partial_grid, partial_block, smem_size, at::cuda::getCurrentCUDAStream()>>>( \
                reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()), \
                reinterpret_cast<const __half*>(k_centroids.data_ptr<at::Half>()), \
                k_cluster_ids.data_ptr<int8_t>(), \
                k_qvalues.data_ptr(), \
                k_qmeta.data_ptr<int32_t>(), \
                INT4_STRUCT ? reinterpret_cast<const __half*>(k_qscale.data_ptr<at::Half>()) : nullptr, \
                v_cluster_ids.data_ptr<int8_t>(), \
                v_qvalues.data_ptr(), \
                v_qmeta.data_ptr<int32_t>(), \
                INT4_STRUCT ? reinterpret_cast<const __half*>(v_qscale.data_ptr<at::Half>()) : nullptr, \
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), \
                partial_beta.data_ptr<float>(), partial_delta.data_ptr<float>(), \
                B, num_q_heads, num_kv_heads, T, nc_k, nc_v, \
                start_pos, pos_q, static_cast<float>(rope_base), \
                static_cast<float>(scaling), num_splits, inv_freq.numel() ? inv_freq.data_ptr<float>() : nullptr); \
    } while (0)
    if (beta_stride == 32) {
        if      (gqa == 1) { CMASS_LAUNCH_4X16_STRUCT_PARTIAL(1, 32); }
        else if (gqa == 4) { CMASS_LAUNCH_4X16_STRUCT_PARTIAL(4, 32); }
        else               { CMASS_LAUNCH_4X16_STRUCT_PARTIAL(8, 32); }
    } else {
        if      (gqa == 1) { CMASS_LAUNCH_4X16_STRUCT_PARTIAL(1, CMASS_MAX_NC); }
        else if (gqa == 4) { CMASS_LAUNCH_4X16_STRUCT_PARTIAL(4, CMASS_MAX_NC); }
        else               { CMASS_LAUNCH_4X16_STRUCT_PARTIAL(8, CMASS_MAX_NC); }
    }
#undef CMASS_LAUNCH_4X16_STRUCT_PARTIAL
    cmass_check_launch("sparse_fused_kqv_cmass_4x16_struct_sparse_index_partial");

    dim3 merge_grid(num_kv_heads, B);
    const size_t merge_smem = static_cast<size_t>(gqa) * beta_stride * sizeof(float);
#define CMASS_LAUNCH_4X16_STRUCT_MERGE(GQA_VAL, NC_VAL) \
    do { \
        dim3 merge_block(GQA_VAL * CMASS_WARP_SIZE); \
        sparse_fused_kqv_cmass_merge_kernel< \
            GQA_VAL, NC_VAL, \
            ((CMASS_INT4_FIXED_SHIFT_SOFTMAX != 0) && INT4_STRUCT)> \
            <<<merge_grid, merge_block, merge_smem, at::cuda::getCurrentCUDAStream()>>>( \
                partial_m.data_ptr<float>(), partial_l.data_ptr<float>(), \
                partial_beta.data_ptr<float>(), partial_delta.data_ptr<float>(), \
                reinterpret_cast<const __half*>(v_centroids.data_ptr<at::Half>()), \
                reinterpret_cast<__half*>(output.data_ptr<at::Half>()), \
                fm_ptr, fl_ptr, B, num_q_heads, num_kv_heads, num_splits, nc_v); \
    } while (0)
    if (beta_stride == 32) {
        if      (gqa == 1) { CMASS_LAUNCH_4X16_STRUCT_MERGE(1, 32); }
        else if (gqa == 4) { CMASS_LAUNCH_4X16_STRUCT_MERGE(4, 32); }
        else               { CMASS_LAUNCH_4X16_STRUCT_MERGE(8, 32); }
    } else {
        if      (gqa == 1) { CMASS_LAUNCH_4X16_STRUCT_MERGE(1, CMASS_MAX_NC); }
        else if (gqa == 4) { CMASS_LAUNCH_4X16_STRUCT_MERGE(4, CMASS_MAX_NC); }
        else               { CMASS_LAUNCH_4X16_STRUCT_MERGE(8, CMASS_MAX_NC); }
    }
#undef CMASS_LAUNCH_4X16_STRUCT_MERGE
    cmass_check_launch("sparse_fused_kqv_cmass_4x16_struct_sparse_index_merge");

    if (return_stats) return {output, final_m, final_l};
    return {output};
}

torch::Tensor sparse_fused_kqv_cmass_merge_recent(
    torch::Tensor comp_out,
    torch::Tensor comp_m,
    torch::Tensor comp_l,
    torch::Tensor scores_recent,
    torch::Tensor v_recent,
    int num_kv_heads, int num_q_heads)
{
    c10::cuda::CUDAGuard guard(comp_out.device());
    TORCH_CHECK(comp_out.dim() == 2 && comp_out.size(1) == CMASS_HEAD_DIM,
                "comp_out must be [B*nh, 128]");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "num_q_heads must be divisible by num_kv_heads");
    const int B_nh = comp_out.size(0);
    const int B = B_nh / num_q_heads;
    const int T_recent = scores_recent.size(2);
    const int gqa = num_q_heads / num_kv_heads;
    auto output = torch::empty_like(comp_out);
    dim3 grid(num_q_heads, B);
    dim3 block(CMASS_HEAD_DIM);
    if (gqa == 1) {
        sparse_fused_kqv_cmass_merge_recent_kernel<1><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(comp_out.data_ptr<at::Half>()),
            comp_m.data_ptr<float>(), comp_l.data_ptr<float>(),
            scores_recent.data_ptr<float>(),
            reinterpret_cast<const __half*>(v_recent.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, num_q_heads, num_kv_heads, T_recent);
    } else if (gqa == 4) {
        sparse_fused_kqv_cmass_merge_recent_kernel<4><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(comp_out.data_ptr<at::Half>()),
            comp_m.data_ptr<float>(), comp_l.data_ptr<float>(),
            scores_recent.data_ptr<float>(),
            reinterpret_cast<const __half*>(v_recent.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, num_q_heads, num_kv_heads, T_recent);
    } else {
        sparse_fused_kqv_cmass_merge_recent_kernel<8><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __half*>(comp_out.data_ptr<at::Half>()),
            comp_m.data_ptr<float>(), comp_l.data_ptr<float>(),
            scores_recent.data_ptr<float>(),
            reinterpret_cast<const __half*>(v_recent.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            B, num_q_heads, num_kv_heads, T_recent);
    }
    cmass_check_launch("sparse_fused_kqv_cmass_merge_recent");
    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_fp16", &sparse_fused_kqv_cmass_impl);
    m.def("decode_structured_fp16", &sparse_fused_kqv_cmass_4x16_struct_sparse_index_impl<false>);
    m.def("decode_structured_int4", &sparse_fused_kqv_cmass_4x16_struct_sparse_index_impl<true>);
    m.def("merge_recent", &sparse_fused_kqv_cmass_merge_recent);
}
