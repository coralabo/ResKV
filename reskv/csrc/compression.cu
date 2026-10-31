// SPDX-License-Identifier: Apache-2.0
// ResKV residual compression, shared by Keys (pre-RoPE) and Values.
//
// One block per token: compute the residual to the assigned centroid, keep the
// largest-magnitude coordinates (top-nnz, or the top 4 of every 16 for the
// structured layout) and write them as FP16 values with position metadata, or
// quantize them to INT4/INT2 with a per-token scale. Clustering runs in PyTorch
// (reskv/compression.py).
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdlib>

// ---- compile-time constants ------------------------------------------------
#define HEAD_DIM   128
#define QTILE16_TILES 8     // D / 16
#define QTILE16_SIZE  16
#define WARP_SIZE  32

namespace {

// ---- device helpers --------------------------------------------------------
__device__ __forceinline__ float half_to_float(at::Half h) {
    return __half2float(reinterpret_cast<__half&>(h));
}

__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    return val;
}

__device__ __forceinline__ uint32_t encode_4x16_struct_sparse_index(uint16_t mask) {
    uint32_t code = 0;
    int rank = 0;
    #pragma unroll
    for (int i = 0; i < QTILE16_SIZE; ++i) {
        if ((mask >> i) & 1u) {
            code |= static_cast<uint32_t>(i & 0xf) << (rank * 4);
            ++rank;
        }
    }
    return code;
}

__global__ void compress_residual_fp16_sparse_qtile_kernel(
    const at::Half* __restrict__ keys,
    const at::Half* __restrict__ centroids,
    const int32_t*  __restrict__ cluster_ids,
    at::Half* __restrict__ qvalues,
    int32_t*  __restrict__ qmeta,
    int B, int nh, int T, int D,
    int num_clusters,
    int nnz_per_token,
    bool use_4x16_struct_index
) {
    int row = blockIdx.x;
    if (row >= B * nh * T) return;

    int t = row % T;
    int tmp = row / T;
    int h = tmp % nh;
    int b = tmp / nh;
    int tid = threadIdx.x;

    __shared__ float   s_residuals[HEAD_DIM];
    __shared__ float   s_abs_vals[HEAD_DIM];
    __shared__ uint8_t s_sparse_mask[HEAD_DIM];
    __shared__ uint16_t s_qtile_mask[QTILE16_TILES];

    const int key_off  = ((b * nh + h) * T + t) * D;
    int cid = cluster_ids[(b * nh + h) * T + t];
    cid = max(0, min(cid, num_clusters - 1));
    const int cent_off = ((b * nh + h) * num_clusters + cid) * D;

    if (tid < D) {
        float kv = half_to_float(keys[key_off + tid]);
        float cv = half_to_float(centroids[cent_off + tid]);
        float r = kv - cv;
        s_residuals[tid] = r;
        __half hsel = __float2half_rn(r);
        s_abs_vals[tid] = fabsf(__half2float(hsel));
        s_sparse_mask[tid] = 0;
    }
    __syncthreads();

    const int nnz_per_qtile = nnz_per_token / QTILE16_TILES;
    if (tid < QTILE16_TILES) {
        uint16_t mask = 0;
        const int base = tid * QTILE16_SIZE;
        for (int sel = 0; sel < nnz_per_qtile; ++sel) {
            float best = -1.0f;
            int best_i = 0;
            #pragma unroll
            for (int i = 0; i < QTILE16_SIZE; ++i) {
                if ((mask & static_cast<uint16_t>(1u << i)) == 0) {
                    const float val = s_abs_vals[base + i];
                    if (val > best || (val == best && i < best_i)) {
                        best = val;
                        best_i = i;
                    }
                }
            }
            mask |= static_cast<uint16_t>(1u << best_i);
        }
        s_qtile_mask[tid] = mask;
    }
    __syncthreads();

    if (tid < D) {
        const int tile = tid >> 4;
        const int bit = tid & 15;
        s_sparse_mask[tid] = (s_qtile_mask[tile] >> bit) & 1u;
    }
    __syncthreads();

    if (tid < QTILE16_TILES / 2) {
        uint32_t lo = static_cast<uint32_t>(s_qtile_mask[tid * 2]);
        uint32_t hi = static_cast<uint32_t>(s_qtile_mask[tid * 2 + 1]);
        if (use_4x16_struct_index) {
            lo = encode_4x16_struct_sparse_index(s_qtile_mask[tid * 2]);
            hi = encode_4x16_struct_sparse_index(s_qtile_mask[tid * 2 + 1]);
        }
        qmeta[row * (QTILE16_TILES / 2) + tid] = static_cast<int32_t>(lo | (hi << 16));
    }
    __syncthreads();

    if (tid < D && s_sparse_mask[tid]) {
        const int tile = tid >> 4;
        const int bit = tid & 15;
        const uint16_t mask = s_qtile_mask[tile];
        const int rank = __popc(static_cast<unsigned>(mask & ((1u << bit) - 1u)));
        const int idx = tile * nnz_per_qtile + rank;
        qvalues[row * nnz_per_token + idx] = (at::Half)__float2half_rn(s_residuals[tid]);
    }
}

// Sparse75 4:16 compressor with a physically compact signed-INT4 payload.
// Each row stores 32 residuals in four int32 words, one 2-bit selected index
// per 4D subgroup (two int32 words total), and one symmetric FP16 scale.  This
// constrained 1:4 layout is also exactly 4:16 and permits rank-free decode.
__global__ void compress_residual_int4_4x16_struct_kernel(
    const at::Half* __restrict__ keys,
    const at::Half* __restrict__ centroids,
    const int32_t*  __restrict__ cluster_ids,
    int32_t* __restrict__ qpacked,
    int32_t* __restrict__ qmeta,
    at::Half* __restrict__ qscale,
    int B, int nh, int T, int D, int num_clusters
) {
    const int row = blockIdx.x;
    if (row >= B * nh * T) return;
    const int t = row % T;
    const int tmp = row / T;
    const int h = tmp % nh;
    const int b = tmp / nh;
    const int tid = threadIdx.x;

    __shared__ float s_residuals[HEAD_DIM];
    __shared__ float s_abs_vals[HEAD_DIM];
    __shared__ float s_compact[32];
    __shared__ uint16_t s_qtile_mask[QTILE16_TILES];
    __shared__ uint8_t s_group_index[32];
    __shared__ float s_scale;

    const int key_off = ((b * nh + h) * T + t) * D;
    int cid = cluster_ids[(b * nh + h) * T + t];
    cid = max(0, min(cid, num_clusters - 1));
    const int cent_off = ((b * nh + h) * num_clusters + cid) * D;
    if (tid < D) {
        const float r = half_to_float(keys[key_off + tid]) -
                        half_to_float(centroids[cent_off + tid]);
        s_residuals[tid] = r;
        s_abs_vals[tid] = fabsf(__half2float(__float2half_rn(r)));
    }
    __syncthreads();

    if (tid < 32) {
        const int base = tid * 4;
        float best = -1.0f;
        int best_i = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float val = s_abs_vals[base + i];
            if (val > best || (val == best && i < best_i)) {
                best = val;
                best_i = i;
            }
        }
        s_group_index[tid] = static_cast<uint8_t>(best_i);
        s_compact[tid] = s_residuals[base + best_i];
    }
    __syncthreads();

    if (tid < 2) {
        uint32_t word = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            word |= static_cast<uint32_t>(s_group_index[tid * 16 + i]) << (i * 2);
        }
        qmeta[row * 2 + tid] = static_cast<int32_t>(word);
    }
    __syncthreads();

    float amax = tid < 32 ? fabsf(s_compact[tid]) : 0.0f;
    s_abs_vals[tid] = amax;
    __syncthreads();
    for (int stride = 64; stride > 0; stride >>= 1) {
        if (tid < stride) s_abs_vals[tid] = fmaxf(s_abs_vals[tid], s_abs_vals[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) {
        s_scale = s_abs_vals[0] > 0.0f ? s_abs_vals[0] * (1.0f / 7.0f) : 1.0f;
        qscale[row] = (at::Half)__float2half_rn(s_scale);
    }
    __syncthreads();

    if (tid < 4) {
        uint32_t word = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q = __float2int_rn(s_compact[tid * 8 + i] / s_scale);
            q = max(-7, min(7, q));
            word |= static_cast<uint32_t>(q & 0xf) << (i * 4);
        }
        qpacked[row * 4 + tid] = static_cast<int32_t>(word);
    }
}

// ============================================================================
// Signed symmetric global-top-k quantization, shared INT2/INT4 specialization
// ============================================================================
template<int BITWIDTH>
__global__ void compress_residual_multibit_kernel(
    const at::Half* __restrict__ keys,
    const at::Half* __restrict__ centroids,
    const int32_t*  __restrict__ cluster_ids,
    int32_t*  __restrict__ residual_qtiles,
    at::Half* __restrict__ residual_qscale,
    int B, int nh, int T, int D,
    int num_clusters,
    int nnz_per_token
) {
    constexpr int VALS_PER_WORD = 32 / BITWIDTH;
    constexpr int SIGNED_QMAX = (1 << (BITWIDTH - 1)) - 1;
    constexpr int SIGNED_QMIN = -SIGNED_QMAX;

    int row = blockIdx.x;
    if (row >= B * nh * T) return;

    int t = row % T;
    int tmp = row / T;
    int h = tmp % nh;
    int b = tmp / nh;
    int tid = threadIdx.x;

    __shared__ float   s_residuals[HEAD_DIM];
    __shared__ float   s_abs_vals[HEAD_DIM];
    __shared__ uint8_t s_sparse_mask[HEAD_DIM];
    __shared__ float   s_nz_values[HEAD_DIM];
    __shared__ float   s_min, s_max;
    __shared__ int     s_top_idx[HEAD_DIM];
    __shared__ int     s_indices[HEAD_DIM];
    __shared__ float   s_sorted_abs[HEAD_DIM];

    // Step 1 – residual
    const int key_off  = ((b * nh + h) * T + t) * D;
    int cid = cluster_ids[(b * nh + h) * T + t];
    cid = max(0, min(cid, num_clusters - 1));
    const int cent_off = ((b * nh + h) * num_clusters + cid) * D;

    if (tid < D) {
        float kv = half_to_float(keys[key_off + tid]);
        float cv = half_to_float(centroids[cent_off + tid]);
        float r  = kv - cv;
        s_residuals[tid] = r;
        __half hsel = __float2half_rn(r);
        s_abs_vals[tid] = fabsf(__half2float(hsel));
        s_sparse_mask[tid] = 0;
    }
    __syncthreads();

    // Step 2 – bitonic sort
    if (tid < D) { s_indices[tid] = tid; s_sorted_abs[tid] = s_abs_vals[tid]; }
    __syncthreads();
    for (int k = 2; k <= D; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && tid < D && ixj < D) {
                float va = s_sorted_abs[tid], vb = s_sorted_abs[ixj];
                int   ia = s_indices[tid],    ib = s_indices[ixj];
                bool desc = ((tid & k) == 0);
                bool swap = desc ? (va < vb || (va == vb && ia > ib))
                                 : (va > vb || (va == vb && ia < ib));
                if (swap) {
                    s_sorted_abs[tid] = vb; s_sorted_abs[ixj] = va;
                    s_indices[tid]    = ib; s_indices[ixj]    = ia;
                }
            }
            __syncthreads();
        }
    }
    if (tid < D)
        s_top_idx[tid] = (tid < nnz_per_token) ? s_indices[tid] : 0x7fffffff;
    __syncthreads();
    for (int k = 2; k <= D; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;
            if (ixj > tid && tid < D && ixj < D) {
                bool asc = ((tid & k) == 0);
                int a = s_top_idx[tid], b2 = s_top_idx[ixj];
                if (asc ? (a > b2) : (a < b2)) {
                    s_top_idx[tid] = b2; s_top_idx[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    // Step 3 – selected values and dimension mask
    if (tid < nnz_per_token) {
        int pos = s_top_idx[tid];
        s_sparse_mask[pos] = 1;
        s_nz_values[tid] = s_residuals[pos];
    }
    __syncthreads();
    // Step 4: compute the symmetric scale with FP16 rounding.
    int nz_count = nnz_per_token;
    float lmin = 1e10f, lmax = -1e10f;
    if (tid < nz_count) { float v = s_nz_values[tid]; lmin = v; lmax = v; }
    lmin = warp_reduce_min(lmin);
    lmax = warp_reduce_max(lmax);
    __shared__ float s_warp_min[4], s_warp_max[4];
    int warp_id = tid / WARP_SIZE, lane = tid % WARP_SIZE;
    if (lane == 0 && warp_id < 4) { s_warp_min[warp_id] = lmin; s_warp_max[warp_id] = lmax; }
    __syncthreads();
    if (tid == 0) {
        float gmin = s_warp_min[0], gmax = s_warp_max[0];
        for (int w = 1; w < 4; ++w) { gmin = fminf(gmin, s_warp_min[w]); gmax = fmaxf(gmax, s_warp_max[w]); }
        s_min = gmin; s_max = gmax;
    }
    __syncthreads();
    const float abs_max_f = fmaxf(fabsf(s_min), fabsf(s_max));
    __half hqscale = (abs_max_f > 0.f) ? __float2half_rn(abs_max_f / (float)SIGNED_QMAX) : __float2half_rn(1.f);
    if (tid == 0) {
        int po = (b * nh + h) * T + t;
        residual_qscale[po] = (at::Half)hqscale;
    }

    // Step 5 – signed dimension qtiles; discarded dimensions are zero.
    {
        __shared__ uint8_t s_qtile[HEAD_DIM];
        if (tid < D) {
            int enc = 0;
            if (s_sparse_mask[tid]) {
                float q = s_residuals[tid] / __half2float(hqscale);
                int qi = max(SIGNED_QMIN, min(SIGNED_QMAX, (int)lrintf(q)));
                enc = qi & ((1 << BITWIDTH) - 1);
            }
            s_qtile[tid] = (uint8_t)enc;
        }
        __syncthreads();
        if (tid < HEAD_DIM / VALS_PER_WORD) {
            uint32_t word = 0;
            #pragma unroll
            for (int i = 0; i < VALS_PER_WORD; ++i) {
                word |= ((uint32_t)s_qtile[tid * VALS_PER_WORD + i] & ((1 << BITWIDTH) - 1)) << (i * BITWIDTH);
            }
            residual_qtiles[row * (HEAD_DIM / VALS_PER_WORD) + tid] = (int32_t)word;
        }
    }
}

} // namespace

// ============================================================================
// C++ / Python bindings
// ============================================================================

// Shared signed quantization binding. No obsolete affine or WMMA layouts.
std::vector<torch::Tensor> compress_residual_multibit_qtile(
    torch::Tensor keys, torch::Tensor centroids, torch::Tensor cluster_ids,
    int nnz_per_token, int bitwidth
) {
    c10::cuda::CUDAGuard guard(keys.device());
    TORCH_CHECK(keys.is_cuda() && centroids.is_cuda() && cluster_ids.is_cuda());
    TORCH_CHECK(keys.scalar_type() == torch::kFloat16 && centroids.scalar_type() == torch::kFloat16);
    TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32);
    TORCH_CHECK(keys.dim() == 4 && centroids.dim() == 4 && cluster_ids.dim() == 3);
    TORCH_CHECK(keys.size(3) == HEAD_DIM && centroids.size(3) == HEAD_DIM);
    TORCH_CHECK(keys.is_contiguous() && centroids.is_contiguous() && cluster_ids.is_contiguous());
    TORCH_CHECK(nnz_per_token == 32 || nnz_per_token == 64);
    TORCH_CHECK(bitwidth == 2 || bitwidth == 4);
    const int B = keys.size(0), nh = keys.size(1), T = keys.size(2), C = centroids.size(2);
    auto qtiles = torch::empty({B, nh, T, 4 * bitwidth},
        keys.options().dtype(torch::kInt32));
    auto qscale = torch::empty({B, nh, T}, keys.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (bitwidth == 2) {
        compress_residual_multibit_kernel<2><<<B * nh * T, HEAD_DIM, 0, stream>>>(
            keys.data_ptr<at::Half>(), centroids.data_ptr<at::Half>(), cluster_ids.data_ptr<int32_t>(),
            qtiles.data_ptr<int32_t>(), qscale.data_ptr<at::Half>(), B, nh, T, HEAD_DIM, C, nnz_per_token);
    } else {
        compress_residual_multibit_kernel<4><<<B * nh * T, HEAD_DIM, 0, stream>>>(
            keys.data_ptr<at::Half>(), centroids.data_ptr<at::Half>(), cluster_ids.data_ptr<int32_t>(),
            qtiles.data_ptr<int32_t>(), qscale.data_ptr<at::Half>(), B, nh, T, HEAD_DIM, C, nnz_per_token);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qtiles, qscale};
}

std::vector<torch::Tensor> compress_residual_fp16_sparse_qtile(
    torch::Tensor keys,
    torch::Tensor centroids,
    torch::Tensor cluster_ids,
    int nnz_per_token, bool use_4x16_struct_index
) {
    c10::cuda::CUDAGuard guard(keys.device());
    TORCH_CHECK(keys.is_cuda(),            "keys must be CUDA tensor");
    TORCH_CHECK(centroids.is_cuda(),       "centroids must be CUDA tensor");
    TORCH_CHECK(cluster_ids.is_cuda(),     "cluster_ids must be CUDA tensor");
    TORCH_CHECK(keys.scalar_type()        == torch::kFloat16, "keys must be float16");
    TORCH_CHECK(centroids.scalar_type()   == torch::kFloat16, "centroids must be float16");
    TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32,   "cluster_ids must be int32");
    TORCH_CHECK(keys.dim() == 4,       "keys must be [B, nh, T, D]");
    TORCH_CHECK(centroids.dim() == 4,  "centroids must be [B, nh, C, D]");
    TORCH_CHECK(cluster_ids.dim() == 3,"cluster_ids must be [B, nh, T]");

    const int B  = keys.size(0);
    const int nh = keys.size(1);
    const int T  = keys.size(2);
    const int D  = keys.size(3);
    const int C  = centroids.size(2);

    TORCH_CHECK(D == HEAD_DIM, "Only D=128 is supported");
    TORCH_CHECK(keys.is_contiguous() && centroids.is_contiguous() && cluster_ids.is_contiguous());
    TORCH_CHECK(nnz_per_token > 0 && nnz_per_token < D,
                "FP16 sparse qtile requires true sparse packing with nnz_per_token < D");
    TORCH_CHECK(
        nnz_per_token % QTILE16_TILES == 0 && nnz_per_token / QTILE16_TILES <= QTILE16_SIZE,
        "FP16 structured sparse qtile requires nnz_per_token divisible by 8 and <128");
    TORCH_CHECK(
        !use_4x16_struct_index || nnz_per_token == 32,
        "4:16 struct sparse index qtile compression is only valid for sparse75 (nnz_per_token=32)");

    auto opt_f16 = torch::TensorOptions().dtype(torch::kFloat16).device(keys.device());
    auto opt_i32 = torch::TensorOptions().dtype(torch::kInt32).device(keys.device());
    auto qvalues = torch::empty({B, nh, T, nnz_per_token}, opt_f16);
    auto qmeta = torch::empty({B, nh, T, QTILE16_TILES / 2}, opt_i32);

    const int total = B * nh * T;
    auto stream = at::cuda::getCurrentCUDAStream();
    compress_residual_fp16_sparse_qtile_kernel<<<total, HEAD_DIM, 0, stream>>>(
        keys.data_ptr<at::Half>(),
        centroids.data_ptr<at::Half>(),
        cluster_ids.data_ptr<int32_t>(),
        qvalues.data_ptr<at::Half>(),
        qmeta.data_ptr<int32_t>(),
        B, nh, T, D, C, nnz_per_token, use_4x16_struct_index);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qvalues, qmeta};
}

std::vector<torch::Tensor> compress_residual_int4_4x16_struct(
    torch::Tensor keys, torch::Tensor centroids, torch::Tensor cluster_ids)
{
    c10::cuda::CUDAGuard guard(keys.device());
    TORCH_CHECK(keys.is_cuda() && centroids.is_cuda() && cluster_ids.is_cuda(),
                "inputs must be CUDA tensors");
    TORCH_CHECK(keys.scalar_type() == torch::kFloat16 &&
                centroids.scalar_type() == torch::kFloat16,
                "keys and centroids must be float16");
    TORCH_CHECK(cluster_ids.scalar_type() == torch::kInt32, "cluster_ids must be int32");
    TORCH_CHECK(keys.dim() == 4 && centroids.dim() == 4 && cluster_ids.dim() == 3,
                "expected keys[B,nh,T,D], centroids[B,nh,C,D], cluster_ids[B,nh,T]");
    const int B = keys.size(0), nh = keys.size(1), T = keys.size(2), D = keys.size(3);
    const int C = centroids.size(2);
    TORCH_CHECK(D == HEAD_DIM, "Only D=128 is supported");
    TORCH_CHECK(keys.is_contiguous() && centroids.is_contiguous() && cluster_ids.is_contiguous(),
                "inputs must be contiguous");
    auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(keys.device());
    auto f16 = torch::TensorOptions().dtype(torch::kFloat16).device(keys.device());
    auto qpacked = torch::empty({B, nh, T, 4}, i32);
    auto qmeta = torch::empty({B, nh, T, 2}, i32);
    auto qscale = torch::empty({B, nh, T}, f16);
    auto stream = at::cuda::getCurrentCUDAStream();
    compress_residual_int4_4x16_struct_kernel<<<B * nh * T, HEAD_DIM, 0, stream>>>(
        keys.data_ptr<at::Half>(), centroids.data_ptr<at::Half>(),
        cluster_ids.data_ptr<int32_t>(), qpacked.data_ptr<int32_t>(),
        qmeta.data_ptr<int32_t>(), qscale.data_ptr<at::Half>(), B, nh, T, D, C);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {qpacked, qmeta, qscale};
}

// --- PYBIND -----------------------------------------------------------------

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compress_fp16", &compress_residual_fp16_sparse_qtile);
    m.def("compress_quantized", &compress_residual_multibit_qtile);
    m.def("compress_structured_int4", &compress_residual_int4_4x16_struct);
}
