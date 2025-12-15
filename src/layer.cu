#include "layer.h"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================================
// CUDA Kernels for Tensor Operations
// ============================================================================

// Block and thread configuration
#define BLOCK_SIZE 256
#define TILE_SIZE 16

// ============================================================================
// Element-wise operation kernels
// ============================================================================

__global__ void add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void add_scalar_kernel(const float* a, float b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b;
    }
}

__global__ void mul_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

__global__ void mul_scalar_kernel(const float* a, float b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b;
    }
}

// ============================================================================
// Activation function kernels
// ============================================================================

__global__ void sigmoid_kernel(const float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float inv_denom = 1.0f / (1.0f + expf(-x[idx]));
        y[idx] = 1.0f * inv_denom;
    }
}

__global__ void silu_kernel(const float* x, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        float inv_sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * inv_sigmoid;
    }
}

// ============================================================================
// Softmax kernel (optimized with shared memory)
// ============================================================================

__global__ void softmax_kernel(const float* x, float* y, size_t outer_size, size_t inner_size) {
    extern __shared__ float shared[];

    size_t row = blockIdx.x;
    if (row >= outer_size) return;

    const float* x_row = x + row * inner_size;
    float* y_row = y + row * inner_size;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        max_val = fmaxf(max_val, x_row[i]);
    }

    // Reduce to find global max
    shared[threadIdx.x] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        float exp_val = expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum += exp_val;
    }

    // Reduce to find global sum
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = shared[0];
    __syncthreads();

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = threadIdx.x; i < inner_size; i += blockDim.x) {
        y_row[i] *= inv_sum;
    }
}

// ============================================================================
// RMSNorm kernel
// ============================================================================

__global__ void rms_norm_kernel(const float* x, const float* weight, float eps, float* y,
                                 size_t outer_size, size_t hidden_size, float inv_hidden_size) {
    extern __shared__ float shared[];

    size_t row = blockIdx.x;
    if (row >= outer_size) return;

    const float* x_row = x + row * hidden_size;
    float* y_row = y + row * hidden_size;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    // Reduce sum
    shared[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms = 1.0f / sqrtf(shared[0] * inv_hidden_size + eps);
    __syncthreads();

    // Normalize and scale
    for (size_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        y_row[i] = x_row[i] * rms * weight[i];
    }
}

// ============================================================================
// Matrix multiplication kernels
// ============================================================================

// Basic matmul: c = a @ b
// a: (m, k), b: (k, n), c: (m, n)
__global__ void matmul_kernel(const float* a, const float* b, float* c,
                               size_t m, size_t k, size_t n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (size_t t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        size_t a_col = t * TILE_SIZE + threadIdx.x;
        size_t b_row = t * TILE_SIZE + threadIdx.y;

        if (row < m && a_col < k) {
            As[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < k && col < n) {
            Bs[threadIdx.y][threadIdx.x] = b[b_row * n + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Transposed matmul: c = a @ b^T
// a: (m, k), b: (n, k), c: (m, n)
__global__ void matmul_transposed_kernel(const float* a, const float* b, float* c,
                                          size_t m, size_t k, size_t n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (size_t t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        size_t a_col = t * TILE_SIZE + threadIdx.x;
        size_t b_col = t * TILE_SIZE + threadIdx.y;  // Note: b is transposed

        if (row < m && a_col < k) {
            As[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // b^T access: b[col, b_col] = b[col * k + b_col]
        if (col < n && b_col < k) {
            Bs[threadIdx.y][threadIdx.x] = b[col * k + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// ============================================================================
// RoPE (Rotary Position Embedding) kernels
// ============================================================================

__global__ void compute_rope_embeddings_kernel(float* cos_out, float* sin_out,
                                                size_t head_dim, size_t max_seq_len, float theta, float inv_head_dim) {
    size_t pos = blockIdx.x;
    size_t i = threadIdx.x;

    if (pos < max_seq_len && i < head_dim / 2) {
        float inv_freq = 1.0f / powf(theta, (float)(2 * i) * inv_head_dim);
        float angle = pos * inv_freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);

        cos_out[pos * head_dim + i] = cos_val;
        cos_out[pos * head_dim + i + head_dim / 2] = cos_val;
        sin_out[pos * head_dim + i] = sin_val;
        sin_out[pos * head_dim + i + head_dim / 2] = sin_val;
    }
}

__global__ void apply_rotary_pos_emb_kernel(float* q, float* k,
                                             const float* cos, const float* sin,
                                             size_t batch, size_t num_q_heads, size_t num_kv_heads,
                                             size_t seq_len, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    size_t total_q = batch * num_q_heads * seq_len * half_dim;
    size_t total_kv = batch * num_kv_heads * seq_len * half_dim;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process Q
    if (idx < total_q) {
        size_t d = idx % half_dim;
        size_t remaining = idx / half_dim;
        size_t s = remaining % seq_len;
        remaining = remaining / seq_len;
        size_t h = remaining % num_q_heads;
        size_t b = remaining / num_q_heads;

        size_t base_idx = ((b * num_q_heads + h) * seq_len + s) * head_dim;

        float q1 = q[base_idx + d];
        float q2 = q[base_idx + d + half_dim];
        float cos_val = cos[s * head_dim + d];
        float sin_val = sin[s * head_dim + d];
        float cos_val2 = cos[s * head_dim + d + half_dim];
        float sin_val2 = sin[s * head_dim + d + half_dim];

        q[base_idx + d] = q1 * cos_val + (-q2) * sin_val;
        q[base_idx + d + half_dim] = q2 * cos_val2 + q1 * sin_val2;
    }

    // Process K
    if (idx < total_kv) {
        size_t d = idx % half_dim;
        size_t remaining = idx / half_dim;
        size_t s = remaining % seq_len;
        remaining = remaining / seq_len;
        size_t h = remaining % num_kv_heads;
        size_t b = remaining / num_kv_heads;

        size_t base_idx = ((b * num_kv_heads + h) * seq_len + s) * head_dim;

        float k1 = k[base_idx + d];
        float k2 = k[base_idx + d + half_dim];
        float cos_val = cos[s * head_dim + d];
        float sin_val = sin[s * head_dim + d];
        float cos_val2 = cos[s * head_dim + d + half_dim];
        float sin_val2 = sin[s * head_dim + d + half_dim];

        k[base_idx + d] = k1 * cos_val + (-k2) * sin_val;
        k[base_idx + d + half_dim] = k2 * cos_val2 + k1 * sin_val2;
    }
}

// ============================================================================
// Repeat KV kernel (for Grouped Query Attention)
// ============================================================================

__global__ void repeat_kv_kernel(const float* x, float* y, size_t n_rep,
                                  size_t batch, size_t num_kv_heads, size_t seq_len, size_t head_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * num_kv_heads * n_rep * seq_len * head_dim;

    if (idx < total) {
        size_t d = idx % head_dim;
        size_t remaining = idx / head_dim;
        size_t s = remaining % seq_len;
        remaining = remaining / seq_len;
        size_t out_h = remaining % (num_kv_heads * n_rep);
        size_t b = remaining / (num_kv_heads * n_rep);

        size_t h = out_h / n_rep;  // Original head index

        y[idx] = x[((b * num_kv_heads + h) * seq_len + s) * head_dim + d];
    }
}

// ============================================================================
// Causal Conv1D kernel
// ============================================================================

__global__ void causal_conv1d_kernel(const float* x, const float* weight, const float* bias,
                                      float* y, size_t batch, size_t channels, size_t seq_len,
                                      size_t kernel_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * channels * seq_len;

    if (idx < total) {
        size_t s = idx % seq_len;
        size_t c = (idx / seq_len) % channels;
        size_t b = idx / (seq_len * channels);

        float sum = 0.0f;
        for (size_t k = 0; k < kernel_size; k++) {
            int input_pos = (int)s - ((int)kernel_size - 1) + (int)k;
            if (input_pos >= 0) {
                sum += x[(b * channels + c) * seq_len + input_pos] * weight[c * kernel_size + k];
            }
        }
        if (bias != nullptr) {
            sum += bias[c];
        }
        y[idx] = sum;
    }
}

// ============================================================================
// Tensor Operations namespace - CUDA implementations
// ============================================================================

namespace tensor_ops {

// Matrix operations
void matmul(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(1);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, k, n);
    CHECK_CUDA(cudaGetLastError());
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    matmul_transposed_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, k, n);
    CHECK_CUDA(cudaGetLastError());
}

// Element-wise operations
void add(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t n = a.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_kernel<<<blocks, BLOCK_SIZE>>>(a.data(), b.data(), c.data(), n);
    CHECK_CUDA(cudaGetLastError());
}

void add_scalar(const Tensor& a, float b, Tensor& c) {
    size_t n = a.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    add_scalar_kernel<<<blocks, BLOCK_SIZE>>>(a.data(), b, c.data(), n);
    CHECK_CUDA(cudaGetLastError());
}

void mul(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t n = a.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_kernel<<<blocks, BLOCK_SIZE>>>(a.data(), b.data(), c.data(), n);
    CHECK_CUDA(cudaGetLastError());
}

void mul_scalar(const Tensor& a, float b, Tensor& c) {
    size_t n = a.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mul_scalar_kernel<<<blocks, BLOCK_SIZE>>>(a.data(), b, c.data(), n);
    CHECK_CUDA(cudaGetLastError());
}

// Activation functions
void sigmoid(const Tensor& x, Tensor& y) {
    size_t n = x.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sigmoid_kernel<<<blocks, BLOCK_SIZE>>>(x.data(), y.data(), n);
    CHECK_CUDA(cudaGetLastError());
}

void silu(const Tensor& x, Tensor& y) {
    size_t n = x.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_kernel<<<blocks, BLOCK_SIZE>>>(x.data(), y.data(), n);
    CHECK_CUDA(cudaGetLastError());
}

void softmax(const Tensor& x, Tensor& y, int dim) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t inner_size = x.size(-1);

    int threads = std::min((int)inner_size, BLOCK_SIZE);
    size_t shared_mem = threads * sizeof(float);

    softmax_kernel<<<outer_size, threads, shared_mem>>>(x.data(), y.data(), outer_size, inner_size);
    CHECK_CUDA(cudaGetLastError());
}

// Normalization
void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y) {
    size_t outer_size = 1;
    for (size_t i = 0; i < x.ndim() - 1; i++) {
        outer_size *= x.size(i);
    }
    size_t hidden_size = x.size(-1);
    float inv_hidden_size = 1.0f / (float)hidden_size;

    int threads = std::min((int)hidden_size, BLOCK_SIZE);
    size_t shared_mem = threads * sizeof(float);

    rms_norm_kernel<<<outer_size, threads, shared_mem>>>(x.data(), weight.data(), eps, y.data(),
                                                          outer_size, hidden_size, inv_hidden_size);
    CHECK_CUDA(cudaGetLastError());
}

// RoPE operations
void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta,
                             Tensor& cos, Tensor& sin) {
    int threads = head_dim / 2;
    float inv_head_dim = 1.0f / (float)head_dim;
    compute_rope_embeddings_kernel<<<max_seq_len, threads>>>(cos.data(), sin.data(),
                                                              head_dim, max_seq_len, theta, inv_head_dim);
    CHECK_CUDA(cudaGetLastError());
}

void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin) {
    size_t batch = q.size(0);
    size_t num_q_heads = q.size(1);
    size_t num_kv_heads = k.size(1);
    size_t seq_len = q.size(2);
    size_t head_dim = q.size(3);
    size_t half_dim = head_dim / 2;

    size_t total_q = batch * num_q_heads * seq_len * half_dim;
    size_t total_kv = batch * num_kv_heads * seq_len * half_dim;
    size_t total = std::max(total_q, total_kv);

    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    apply_rotary_pos_emb_kernel<<<blocks, BLOCK_SIZE>>>(q.data(), k.data(), cos.data(), sin.data(),
                                                         batch, num_q_heads, num_kv_heads,
                                                         seq_len, head_dim);
    CHECK_CUDA(cudaGetLastError());
}

// Grouped Query Attention operations
void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y) {
    if (n_rep == 1) {
        CHECK_CUDA(cudaMemcpy(y.data(), x.data(), x.size() * sizeof(float), cudaMemcpyDeviceToDevice));
        return;
    }

    size_t batch = x.size(0);
    size_t num_kv_heads = x.size(1);
    size_t seq_len = x.size(2);
    size_t head_dim = x.size(3);

    size_t total = batch * num_kv_heads * n_rep * seq_len * head_dim;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    repeat_kv_kernel<<<blocks, BLOCK_SIZE>>>(x.data(), y.data(), n_rep,
                                              batch, num_kv_heads, seq_len, head_dim);
    CHECK_CUDA(cudaGetLastError());
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

    size_t total = batch * channels * seq_len;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float* bias_ptr = (bias != nullptr && bias->size() > 0) ? bias->data() : nullptr;

    causal_conv1d_kernel<<<blocks, BLOCK_SIZE>>>(x.data(), weight.data(), bias_ptr, y.data(),
                                                   batch, channels, seq_len, kernel_size);
    CHECK_CUDA(cudaGetLastError());
}

} // namespace tensor_ops

// ============================================================================
// Layer Implementations - Small building blocks
// ============================================================================

// RMSNorm implementation
RMSNorm::RMSNorm(const std::string& weight_file) {
    weight_ = Tensor::load_from_file(weight_file);
}

void RMSNorm::forward(const Tensor& x, Tensor& y) {
    tensor_ops::rms_norm(x, weight_, RMS_NORM_EPS, y);
}

// RotaryEmbedding implementation
RotaryEmbedding::RotaryEmbedding() : max_seq_len_(MAX_POSITION_EMBEDDINGS) {
    cos_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    sin_cached_ = Tensor({max_seq_len_, HEAD_DIM});
    tensor_ops::compute_rope_embeddings(HEAD_DIM, max_seq_len_, ROPE_THETA,
                                        cos_cached_, sin_cached_);
}

void RotaryEmbedding::forward(size_t seq_len, Tensor& cos, Tensor& sin) {
    // Copy cached values for the given sequence length
    CHECK_CUDA(cudaMemcpy(cos.data(), cos_cached_.data(), seq_len * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(sin.data(), sin_cached_.data(), seq_len * HEAD_DIM * sizeof(float), cudaMemcpyDeviceToDevice));
}
