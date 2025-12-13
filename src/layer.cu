#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <cuda_runtime.h>

// ============================================================================
// CUDA Kernel Definitions
// ============================================================================

#define TILE_SIZE 32
#define WARP_SIZE 32

// Optimized tiled matrix multiplication: C = A @ B^T
// A: (M, K), B: (N, K), C: (M, N)
__global__ void matmul_transposed_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int tile_col = t * TILE_SIZE + threadIdx.x;
        int tile_row = t * TILE_SIZE + threadIdx.y;

        // Load tile from A
        if (row < M && tile_col < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile_col];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B (transposed access for B^T)
        if (col < N && tile_row < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[col * K + tile_row];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// RMS Normalization kernel
__global__ void rms_norm_kernel(const float* __restrict__ input,
                                const float* __restrict__ weight,
                                float* __restrict__ output,
                                int outer_size, int hidden_size, float eps) {
    int idx = blockIdx.x;
    if (idx >= outer_size) return;

    const float* in_ptr = input + idx * hidden_size;
    float* out_ptr = output + idx * hidden_size;

    // Compute sum of squares with warp reduction
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        float val = in_ptr[d];
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    // Final reduction across warps
    if (warp_id == 0) {
        sum_sq = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? warp_sums[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
        }
    }

    __shared__ float rms;
    if (threadIdx.x == 0) {
        rms = sqrtf(sum_sq / hidden_size + eps);
    }
    __syncthreads();

    // Normalize and apply weight
    for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
        out_ptr[d] = (in_ptr[d] / rms) * weight[d];
    }
}

// SiLU activation kernel
__global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// Element-wise multiply kernel
__global__ void mul_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// Element-wise add kernel
__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// Softmax kernel with numerical stability
__global__ void softmax_kernel(float* __restrict__ data,
                               int outer_size, int inner_size) {
    int idx = blockIdx.x;
    if (idx >= outer_size) return;

    float* row = data + idx * inner_size;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j]);
    }

    // Warp reduction for max
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        float exp_val = expf(row[j] - max_val);
        row[j] = exp_val;
        sum += exp_val;
    }

    // Warp reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;

    // Normalize
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
        row[j] /= sum;
    }
}

// RoPE kernel with transpose: input (b,s,h,d) -> output (b,h,s,d) with rotation
__global__ void rope_transpose_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      const float* __restrict__ cos,
                                      const float* __restrict__ sin,
                                      int batch, int seq_len, int num_heads, int head_dim) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x;

    if (b >= batch || h >= num_heads || s >= seq_len) return;

    int half_dim = head_dim / 2;
    int idx_in_base = ((b * seq_len + s) * num_heads + h) * head_dim;
    int idx_out_base = ((b * num_heads + h) * seq_len + s) * head_dim;
    int cos_sin_offset = s * head_dim;

    for (int d = threadIdx.x; d < half_dim; d += blockDim.x) {
        float x1 = input[idx_in_base + d];
        float x2 = input[idx_in_base + d + half_dim];

        float cos_val = cos[cos_sin_offset + d];
        float sin_val = sin[cos_sin_offset + d];
        float cos_val2 = cos[cos_sin_offset + d + half_dim];
        float sin_val2 = sin[cos_sin_offset + d + half_dim];

        output[idx_out_base + d] = x1 * cos_val - x2 * sin_val;
        output[idx_out_base + d + half_dim] = x2 * cos_val2 + x1 * sin_val2;
    }
}

// Repeat KV kernel for GQA
__global__ void repeat_kv_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int batch, int seq_len, int num_kv_heads,
                                 int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_kv_heads * seq_len * head_dim;

    if (idx >= total) return;

    int num_kv_groups = num_heads / num_kv_heads;

    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h_kv = (idx / (head_dim * seq_len)) % num_kv_heads;
    int b = idx / (head_dim * seq_len * num_kv_heads);

    float val = input[idx];

    for (int r = 0; r < num_kv_groups; r++) {
        int h_out = h_kv * num_kv_groups + r;
        int idx_out = ((b * num_heads + h_out) * seq_len + s) * head_dim + d;
        output[idx_out] = val;
    }
}

// Attention scores kernel: Q @ K^T with scale
__global__ void attn_scores_kernel(const float* __restrict__ Q,
                                   const float* __restrict__ K,
                                   float* __restrict__ scores,
                                   int batch, int num_heads, int seq_len,
                                   int head_dim, float scale) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;

    if (b >= batch || h >= num_heads || i >= seq_len) return;

    int q_offset = ((b * num_heads + h) * seq_len + i) * head_dim;
    int k_base = (b * num_heads + h) * seq_len * head_dim;
    int scores_offset = ((b * num_heads + h) * seq_len + i) * seq_len;

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float sum = 0.0f;
        int k_offset = k_base + j * head_dim;

        for (int d = 0; d < head_dim; d++) {
            sum += Q[q_offset + d] * K[k_offset + d];
        }
        scores[scores_offset + j] = sum * scale;
    }
}

// Causal masked softmax kernel
__global__ void causal_softmax_kernel(float* __restrict__ scores,
                                      int batch, int num_heads, int seq_len) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;

    if (b >= batch || h >= num_heads || i >= seq_len) return;

    int offset = ((b * num_heads + h) * seq_len + i) * seq_len;

    // Find max (only up to causal position)
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        max_val = fmaxf(max_val, scores[offset + j]);
    }

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, off));
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        float exp_val = expf(scores[offset + j] - max_val);
        scores[offset + j] = exp_val;
        sum += exp_val;
    }

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;

    // Normalize and apply causal mask
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        scores[offset + j] = (j <= i) ? (scores[offset + j] / sum) : 0.0f;
    }
}

// Attention output kernel: scores @ V
__global__ void attn_output_kernel(const float* __restrict__ scores,
                                   const float* __restrict__ V,
                                   float* __restrict__ output,
                                   int batch, int num_heads, int seq_len, int head_dim) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;

    if (b >= batch || h >= num_heads || i >= seq_len) return;

    int scores_offset = ((b * num_heads + h) * seq_len + i) * seq_len;
    int v_base = (b * num_heads + h) * seq_len * head_dim;
    int out_offset = ((b * num_heads + h) * seq_len + i) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += scores[scores_offset + j] * V[v_base + j * head_dim + d];
        }
        output[out_offset + d] = sum;
    }
}

// Transpose kernel: (b, h, s, d) -> (b, s, h, d)
__global__ void transpose_bhsd_to_bshd_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int batch, int num_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);

    int idx_out = ((b * seq_len + s) * num_heads + h) * head_dim + d;
    output[idx_out] = input[idx];
}

// Causal Conv1D kernel
__global__ void causal_conv1d_kernel(const float* __restrict__ input,
                                     const float* __restrict__ weight,
                                     float* __restrict__ output,
                                     int batch, int channels, int seq_len, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * seq_len;

    if (idx >= total) return;

    int s = idx % seq_len;
    int c = (idx / seq_len) % channels;
    int b = idx / (seq_len * channels);

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int input_pos = s - (kernel_size - 1) + k;
        if (input_pos >= 0) {
            sum += input[(b * channels + c) * seq_len + input_pos] * weight[c * kernel_size + k];
        }
    }
    output[idx] = sum;
}

// Fused transpose and split kernel for ShortConv
__global__ void transpose_split_kernel(const float* __restrict__ input,
                                       float* __restrict__ B,
                                       float* __restrict__ C,
                                       float* __restrict__ x_gate,
                                       int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;

    if (idx >= total) return;

    int s = idx % seq_len;
    int h = (idx / seq_len) % hidden_size;
    int b = idx / (hidden_size * seq_len);

    int base_idx = (b * seq_len + s) * 3 * hidden_size;
    B[idx] = input[base_idx + h];
    C[idx] = input[base_idx + h + hidden_size];
    x_gate[idx] = input[base_idx + h + 2 * hidden_size];
}

// Fused mul and conv kernel for ShortConv
__global__ void fused_mul_conv_kernel(const float* __restrict__ B,
                                      const float* __restrict__ x_gate,
                                      const float* __restrict__ conv_weight,
                                      float* __restrict__ conv_out,
                                      int batch, int hidden_size, int seq_len, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;

    if (idx >= total) return;

    int s = idx % seq_len;
    int c = (idx / seq_len) % hidden_size;
    int b = idx / (hidden_size * seq_len);

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int input_pos = s - (kernel_size - 1) + k;
        if (input_pos >= 0) {
            int input_idx = (b * hidden_size + c) * seq_len + input_pos;
            float bx_val = B[input_idx] * x_gate[input_idx];
            sum += bx_val * conv_weight[c * kernel_size + k];
        }
    }
    conv_out[idx] = sum;
}

// Fused mul and transpose kernel for ShortConv
__global__ void fused_mul_transpose_kernel(const float* __restrict__ C,
                                           const float* __restrict__ conv_out,
                                           float* __restrict__ output,
                                           int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_size;

    if (idx >= total) return;

    int h = idx % hidden_size;
    int s = (idx / hidden_size) % seq_len;
    int b = idx / (seq_len * hidden_size);

    int input_idx = (b * hidden_size + h) * seq_len + s;
    output[idx] = C[input_idx] * conv_out[input_idx];
}

// ============================================================================
// GPU Buffer Management
// ============================================================================

// Global GPU buffers (persistent across forward passes)
static float* g_gpu_buffer = nullptr;
static size_t g_gpu_buffer_size = 0;

// Pre-allocated GPU weight buffers
static std::unordered_map<std::string, float*> g_gpu_weights;

void ensure_gpu_buffer(size_t required_size) {
    if (required_size > g_gpu_buffer_size) {
        if (g_gpu_buffer) {
            cudaFree(g_gpu_buffer);
        }
        cudaMalloc(&g_gpu_buffer, required_size * sizeof(float));
        g_gpu_buffer_size = required_size;
    }
}

// ============================================================================
// Tensor Operations - Basic operations on tensors (CPU fallback + GPU)
// ============================================================================

namespace tensor_ops {

// Persistent GPU buffers for matmul
static float* d_matmul_A = nullptr;
static float* d_matmul_B = nullptr;
static float* d_matmul_C = nullptr;
static size_t d_matmul_A_size = 0;
static size_t d_matmul_B_size = 0;
static size_t d_matmul_C_size = 0;

// Matrix multiplication: C = A @ B^T (GPU accelerated with persistent buffers)
void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t M = a.size(0);
    size_t K = a.size(1);
    size_t N = b.size(0);

    size_t size_A = M * K;
    size_t size_B = N * K;
    size_t size_C = M * N;

    // Reallocate GPU memory only if needed (persistent buffers)
    if (size_A > d_matmul_A_size) {
        if (d_matmul_A) cudaFree(d_matmul_A);
        cudaMalloc(&d_matmul_A, size_A * sizeof(float));
        d_matmul_A_size = size_A;
    }
    if (size_B > d_matmul_B_size) {
        if (d_matmul_B) cudaFree(d_matmul_B);
        cudaMalloc(&d_matmul_B, size_B * sizeof(float));
        d_matmul_B_size = size_B;
    }
    if (size_C > d_matmul_C_size) {
        if (d_matmul_C) cudaFree(d_matmul_C);
        cudaMalloc(&d_matmul_C, size_C * sizeof(float));
        d_matmul_C_size = size_C;
    }

    // Copy data to GPU
    cudaMemcpy(d_matmul_A, a.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matmul_B, b.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_transposed_kernel<<<grid, block>>>(d_matmul_A, d_matmul_B, d_matmul_C, M, N, K);

    // Copy result back
    cudaMemcpy(c.data(), d_matmul_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
}

void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a.at(i, p) * b.at(p, j);
            }
            c.at(i, j) = sum;
        }
    }
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b[i];
    }
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] + b;
    }
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b[i];
    }
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    #pragma omp parallel for
    for (size_t i = 0; i < a.size(); i++) {
        c[i] = a[i] * b;
    }
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = 1.0f / (1.0f + std::exp(-x[i]));
    }
}

void silu(const Tensor& x, Tensor& y) {
    #pragma omp parallel for
    for (size_t i = 0; i < x.size(); i++) {
        y[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);

    #pragma omp parallel for
    for (size_t i = 0; i < outer_size; i++) {
        float max_val = x[i * inner_size];
        for (size_t j = 1; j < inner_size; j++) {
            max_val = std::max(max_val, x[i * inner_size + j]);
        }

        float sum = 0.0f;
        for (size_t j = 0; j < inner_size; j++) {
            y[i * inner_size + j] = std::exp(x[i * inner_size + j] - max_val);
            sum += y[i * inner_size + j];
        }

        for (size_t j = 0; j < inner_size; j++) {
            y[i * inner_size + j] /= sum;
        }
    }
}

// Normalization - GPU accelerated using rms_norm_kernel
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    // =========================================================================
    // OPTIMIZATION 3.1: GPU RMSNorm Implementation
    // Replaces CPU fallback with GPU kernel (rms_norm_kernel)
    // =========================================================================

    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    size_t total_size = outer_size * hidden_size;

    // Allocate GPU memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, total_size * sizeof(float));
    cudaMalloc(&d_weight, hidden_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));

    // Upload data to GPU
    cudaMemcpy(d_input, x.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch RMSNorm kernel
    // Each block handles one row (outer_size rows, hidden_size columns)
    int block_size = 256;  // Threads per block for reduction
    rms_norm_kernel<<<outer_size, block_size>>>(d_input, d_weight, d_output, outer_size, hidden_size, eps);

    // Download result
    cudaMemcpy(y.data(), d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    std::vector<float> inv_freq(head_dim / 2);
    for (size_t i = 0; i < head_dim / 2; i++) {
        inv_freq[i] = 1.0f / std::pow(theta, (float)(2 * i) / head_dim);
    }

    #pragma omp parallel for
    for (size_t pos = 0; pos < max_seq_len; pos++) {
        for (size_t i = 0; i < head_dim / 2; i++) {
            float angle = pos * inv_freq[i];
            cos.at(pos, i) = std::cos(angle);
            cos.at(pos, i + head_dim / 2) = std::cos(angle);
            sin.at(pos, i) = std::sin(angle);
            sin.at(pos, i + head_dim / 2) = std::sin(angle);
        }
    }
}

void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
    size_t batch = q.size(0);
    size_t num_q_heads = q.size(1);
    size_t num_kv_heads = k.size(1);
    size_t seq_len = q.size(2);
    size_t head_dim = q.size(3);
    size_t half_dim = head_dim / 2;

    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_q_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < half_dim; d++) {
                    float q1 = q.at(b, h, s, d);
                    float q2 = q.at(b, h, s, d + half_dim);

                    q.at(b, h, s, d) = q1 * cos.at(s, d) + (-q2) * sin.at(s, d);
                    q.at(b, h, s, d + half_dim) = q2 * cos.at(s, d + half_dim) + q1 * sin.at(s, d + half_dim);
                }
            }
        }
    }

    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t s = 0; s < seq_len; s++) {
                for (size_t d = 0; d < half_dim; d++) {
                    float k1 = k.at(b, h, s, d);
                    float k2 = k.at(b, h, s, d + half_dim);

                    k.at(b, h, s, d) = k1 * cos.at(s, d) + (-k2) * sin.at(s, d);
                    k.at(b, h, s, d + half_dim) = k2 * cos.at(s, d + half_dim) + k1 * sin.at(s, d + half_dim);
                }
            }
        }
    }
}

// GQA operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    if (n_rep == 1) {
        std::memcpy(y.data(), x.data(), x.size() * sizeof(float));
        return;
    }

    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);

    #pragma omp parallel for collapse(4)
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < num_kv_heads; h++) {
            for (size_t r = 0; r < n_rep; r++) {
                for (size_t s = 0; s < seq_len; s++) {
                    size_t out_h = h * n_rep + r;
                    for (size_t d = 0; d < head_dim; d++) {
                        y.at(b, out_h, s, d) = x.at(b, h, s, d);
                    }
                }
            }
        }
    }
}

// Convolution operations
void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias, Tensor& y) {
    size_t batch = x.size(0);
    size_t channels = x.size(1);
    size_t seq_len = x.size(2);
    size_t kernel_size = weight.size(2);

    if (y.size() == 0) {
        y = Tensor({batch, channels, seq_len});
    }
    y.zero();

    #pragma omp parallel for collapse(3)
    for (size_t b = 0; b < batch; b++) {
        for (size_t c = 0; c < channels; c++) {
            for (size_t s = 0; s < seq_len; s++) {
                float sum = 0.0f;
                for (size_t k = 0; k < kernel_size; k++) {
                    int input_pos = (int)s - ((int)kernel_size - 1) + (int)k;
                    if (input_pos >= 0) {
                        sum += x.at(b, c, input_pos) * weight.at(c, 0, k);
                    }
                }
                if (bias != nullptr) {
                    sum += (*bias)[c];
                }
                y.at(b, c, s) = sum;
            }
        }
    }
}

} // namespace tensor_ops

// ============================================================================
// Layer Implementations
// ============================================================================

RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file);
}

void RMSNorm::forward(const Tensor& x, Tensor& y) {
    tensor_ops::rms_norm(x, weight_, RMS_NORM_EPS, y);
}

RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
    cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    tensor_ops::compute_rope_embeddings(HEAD_DIM, max_seq_len_, ROPE_THETA,
                                        cos_cached_, sin_cached_);
}

void RotaryEmbedding::forward(size_t seq_len, Tensor& cos, Tensor& sin) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t j = 0; j < HEAD_DIM; j++) {
            cos.at(i, j) = cos_cached_.at(i, j);
            sin.at(i, j) = sin_cached_.at(i, j);
        }
    }
}
