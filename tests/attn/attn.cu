#include "attn.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

#define TILE_SIZE 32
#define WARP_SIZE 32

static float *x_gpu, *cos_gpu, *sin_gpu;
static float *q_proj_gpu, *k_proj_gpu, *v_proj_gpu, *o_proj_gpu;
static float *q_norm_gpu, *k_norm_gpu, *output_gpu;
static float *q_proj_out_gpu, *k_proj_out_gpu, *v_proj_out_gpu;
static float *q_normed_gpu, *k_normed_gpu;
static float *q_transposed_gpu, *k_transposed_gpu, *k_repeated_gpu, *v_transposed_gpu;
static float *attn_scores_gpu, *attn_out_gpu, *attn_out_transposed_gpu;

void attn_initialize(int batch, int seq_len, int num_heads, int head_dim, int num_kv_heads,
                     float *cos, float *sin, float *q_proj, float *k_proj,
                     float *v_proj, float *o_proj, float *q_norm, float *k_norm) {
    int hidden_size = num_heads * head_dim;

    // Allocate input and output tensors on GPU
    // x_gpu : input tensor
    // cos_gpu, sin_gpu : RoPE parameters
    // q_proj_gpu, k_proj_gpu, v_proj_gpu, o_proj_gpu : projection weights
    // q_norm_gpu, k_norm_gpu : RMSNorm weights
    // output_gpu : output tensor
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cos_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sin_gpu, seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_proj_gpu, num_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_proj_gpu, num_kv_heads * head_dim * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&o_proj_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_norm_gpu, head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));

    // Intermediate tensors
    CHECK_CUDA(cudaMalloc(&q_proj_out_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_proj_out_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&q_normed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_normed_gpu, batch * seq_len * num_kv_heads * head_dim * sizeof(float)));

    // Transposed and repeated tensors
    CHECK_CUDA(cudaMalloc(&q_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_transposed_gpu, batch * num_kv_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&k_repeated_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&v_transposed_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_scores_gpu, batch * num_heads * seq_len * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_gpu, batch * num_heads * seq_len * head_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&attn_out_transposed_gpu, batch * seq_len * num_heads * head_dim * sizeof(float)));

    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(cos_gpu, cos, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(sin_gpu, sin, seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(q_proj_gpu, q_proj, num_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_proj_gpu, k_proj, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(v_proj_gpu, v_proj, num_kv_heads * head_dim * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(o_proj_gpu, o_proj, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(q_norm_gpu, q_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(k_norm_gpu, k_norm, head_dim * sizeof(float), cudaMemcpyHostToDevice));
}

// matrix multiplication: C = A @ B^T
// A: (M, K), B: (N, K), C: (M, N)
__global__ void matmul_tiled_kernel(float *A, float *B, float *C, int M, int N, int K) {
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
            tile_A[threadIdx.y][threadIdx.x] = 0.0f; // Padding
        }

        // Load tile from B (transposed access for B^T)
        if (col < N && tile_row < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[col * K + tile_row];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f; // Padding
        }

        __syncthreads();

        // Compute partial sum for this tile
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

//  RMS Normalization
__global__ void rms_norm_kernel(float *input, float *output, float *norm_weight,
                                int batch, int seq_len, int num_heads, int head_dim) {
    int idx_base = (blockIdx.z * seq_len * num_heads + blockIdx.y * num_heads + blockIdx.x) * head_dim;

    // Compute sum of squares with warp reduction
    float sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = input[idx_base + d];
        sum_sq += val * val;
    }

    // Warp-level reduction
    __shared__ float warp_sums[32];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

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
        rms = sqrtf(sum_sq / head_dim + 1e-5f);
    }
    __syncthreads();

    // Normalize and apply weight
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        output[idx_base + d] = (input[idx_base + d] / rms) * norm_weight[d];
    }
}

// RoPE with transpose
__global__ void rope_kernel(float *input, float *output, float *cos, float *sin,
                           int batch, int seq_len, int num_heads, int head_dim) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int s = blockIdx.x;

    if (b >= batch || h >= num_heads || s >= seq_len) return;

    int idx_in = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim;
    int idx_out = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + s * head_dim;
    int cos_sin_offset = s * head_dim;

    for (int d = threadIdx.x; d < head_dim / 2; d += blockDim.x) {
        float x1 = input[idx_in + d];
        float x2 = input[idx_in + d + head_dim / 2];

        float cos_val = cos[cos_sin_offset + d];
        float sin_val = sin[cos_sin_offset + d];
        float cos_val2 = cos[cos_sin_offset + d + head_dim / 2];
        float sin_val2 = sin[cos_sin_offset + d + head_dim / 2];

        output[idx_out + d] = x1 * cos_val - x2 * sin_val;
        output[idx_out + d + head_dim / 2] = x2 * cos_val2 + x1 * sin_val2;
    }
}

// K, V repetition kernel for GQA
__global__ void repeat_kv_kernel(float *input, float *output,
                                int batch, int seq_len, int num_kv_heads, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_kv_heads * seq_len * head_dim;

    if (idx >= total) return;

    int num_kv_groups = num_heads / num_kv_heads;

    // Decompose index
    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h_kv = (idx / (head_dim * seq_len)) % num_kv_heads;
    int b = idx / (head_dim * seq_len * num_kv_heads);

    int idx_in = b * num_kv_heads * seq_len * head_dim + h_kv * seq_len * head_dim + s * head_dim + d;
    float val = input[idx_in];

    // Repeat to all groups
    for (int r = 0; r < num_kv_groups; r++) {
        int h_out = h_kv * num_kv_groups + r;
        int idx_out = b * num_heads * seq_len * head_dim + h_out * seq_len * head_dim + s * head_dim + d;
        output[idx_out] = val;
    }
}

// V repetition and transpose kernel
__global__ void repeat_transpose_v_kernel(float *input, float *output,
                                         int batch, int seq_len, int num_kv_heads, int num_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * num_kv_heads * head_dim;

    if (idx >= total) return;

    int num_kv_groups = num_heads / num_kv_heads;

    int d = idx % head_dim;
    int h_kv = (idx / head_dim) % num_kv_heads;
    int s = (idx / (head_dim * num_kv_heads)) % seq_len;
    int b = idx / (head_dim * num_kv_heads * seq_len);

    float val = input[idx];

    for (int r = 0; r < num_kv_groups; r++) {
        int h_out = h_kv * num_kv_groups + r;
        int idx_out = b * num_heads * seq_len * head_dim + h_out * seq_len * head_dim + s * head_dim + d;
        output[idx_out] = val;
    }
}

// Optimized attention scores: Q @ K^T with vectorized loads
__global__ void attn_scores_kernel(float *Q, float *K, float *scores,
                                  int batch, int num_heads, int seq_len, int head_dim, float scale) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;

    if (b >= batch || h >= num_heads || i >= seq_len) return;

    int q_offset = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim;
    int k_base = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
    int scores_offset = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len;

    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float sum = 0.0f;
        int k_offset = k_base + j * head_dim;

        #pragma unroll 8
        for (int d = 0; d < head_dim; d++) {
            sum += Q[q_offset + d] * K[k_offset + d];
        }
        scores[scores_offset + j] = sum * scale;
    }
}

// Optimized masked softmax with warp reduction
__global__ void masked_softmax_kernel(float *scores, int batch, int num_heads, int seq_len) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;

    if (b >= batch || h >= num_heads || i >= seq_len) return;

    int offset = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len;

    // Find max with parallel reduction
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        max_val = fmaxf(max_val, scores[offset + j]);
    }

    // Warp reduction for max
    for (int offset_reduce = WARP_SIZE / 2; offset_reduce > 0; offset_reduce >>= 1) {
        float other = __shfl_down_sync(0xffffffff, max_val, offset_reduce);
        max_val = fmaxf(max_val, other);
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) {
        shared_max = max_val;
    }
    __syncthreads();
    max_val = shared_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        float exp_val = expf(scores[offset + j] - max_val);
        scores[offset + j] = exp_val;
        sum += exp_val;
    }

    // Warp reduction for sum
    for (int offset_reduce = WARP_SIZE / 2; offset_reduce > 0; offset_reduce >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset_reduce);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) {
        shared_sum = sum;
    }
    __syncthreads();
    sum = shared_sum;

    // Normalize and apply causal mask
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        scores[offset + j] = (j <= i) ? (scores[offset + j] / sum) : 0.0f;
    }
}

// Optimized attention output: scores @ V
__global__ void attn_output_kernel(float *scores, float *V, float *output,
                                  int batch, int num_heads, int seq_len, int head_dim) {
    int b = blockIdx.z;
    int h = blockIdx.y;
    int i = blockIdx.x;

    if (b >= batch || h >= num_heads || i >= seq_len) return;

    int scores_offset = b * num_heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len;
    int v_base = b * num_heads * seq_len * head_dim + h * seq_len * head_dim;
    int out_offset = b * num_heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll 4
        for (int j = 0; j < seq_len; j++) {
            sum += scores[scores_offset + j] * V[v_base + j * head_dim + d];
        }
        output[out_offset + d] = sum;
    }
}

// Transpose kernel - vectorized
__global__ void transpose_kernel(float *input, float *output,
                                int batch, int num_heads, int seq_len, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * num_heads * seq_len * head_dim;

    if (idx >= total) return;

    int d = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % num_heads;
    int b = idx / (head_dim * seq_len * num_heads);

    int idx_out = b * seq_len * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d;
    output[idx_out] = input[idx];
}

void attn(float *x, float *cos, float *sin, float *q_proj, float *k_proj,
          float *v_proj, float *o_proj, float *q_norm, float *k_norm,
          float *output, int batch, int seq_len, int num_heads,
          int head_dim, int num_kv_heads) {

    int hidden_size = num_heads * head_dim;

    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // Step 1: Q, K, V projection with tiled matmul
    dim3 block_tile(TILE_SIZE, TILE_SIZE);

    // Q projection
    int M_q = batch * seq_len;
    int N_q = num_heads * head_dim;
    int K_q = hidden_size;
    dim3 grid_q((N_q + TILE_SIZE - 1) / TILE_SIZE, (M_q + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid_q, block_tile>>>(x_gpu, q_proj_gpu, q_proj_out_gpu, M_q, N_q, K_q);

    // K projection
    int M_k = batch * seq_len;
    int N_k = num_kv_heads * head_dim;
    int K_k = hidden_size;
    dim3 grid_k((N_k + TILE_SIZE - 1) / TILE_SIZE, (M_k + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid_k, block_tile>>>(x_gpu, k_proj_gpu, k_proj_out_gpu, M_k, N_k, K_k);

    // V projection
    int M_v = batch * seq_len;
    int N_v = num_kv_heads * head_dim;
    int K_v = hidden_size;
    dim3 grid_v((N_v + TILE_SIZE - 1) / TILE_SIZE, (M_v + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid_v, block_tile>>>(x_gpu, v_proj_gpu, v_proj_out_gpu, M_v, N_v, K_v);

    // Step 2: Apply RMS normalization to Q and K
    dim3 norm_grid(num_heads, seq_len, batch);
    rms_norm_kernel<<<norm_grid, 128>>>(q_proj_out_gpu, q_normed_gpu, q_norm_gpu,
                                        batch, seq_len, num_heads, head_dim);

    dim3 norm_k_grid(num_kv_heads, seq_len, batch);
    rms_norm_kernel<<<norm_k_grid, 128>>>(k_proj_out_gpu, k_normed_gpu, k_norm_gpu,
                                          batch, seq_len, num_kv_heads, head_dim);

    // Step 3: Apply RoPE and transpose Q and K
    dim3 rope_grid(seq_len, num_heads, batch);
    rope_kernel<<<rope_grid, 128>>>(q_normed_gpu, q_transposed_gpu, cos_gpu, sin_gpu,
                                    batch, seq_len, num_heads, head_dim);

    dim3 rope_k_grid(seq_len, num_kv_heads, batch);
    rope_kernel<<<rope_k_grid, 128>>>(k_normed_gpu, k_transposed_gpu, cos_gpu, sin_gpu,
                                      batch, seq_len, num_kv_heads, head_dim);

    // Step 4: Repeat K and V for GQA
    int total_kv = batch * num_kv_heads * seq_len * head_dim;
    int threads_kv = 256;
    int blocks_kv = (total_kv + threads_kv - 1) / threads_kv;
    repeat_kv_kernel<<<blocks_kv, threads_kv>>>(k_transposed_gpu, k_repeated_gpu,
                                                batch, seq_len, num_kv_heads, num_heads, head_dim);

    int total_v = batch * seq_len * num_kv_heads * head_dim;
    int threads_v = 256;
    int blocks_v = (total_v + threads_v - 1) / threads_v;
    repeat_transpose_v_kernel<<<blocks_v, threads_v>>>(v_proj_out_gpu, v_transposed_gpu,
                                                       batch, seq_len, num_kv_heads, num_heads, head_dim);

    // Step 5: Compute attention scores: Q @ K^T
    float scale = 1.0f / sqrtf((float)head_dim);
    dim3 scores_grid(seq_len, num_heads, batch);
    attn_scores_kernel<<<scores_grid, 128>>>(q_transposed_gpu, k_repeated_gpu, attn_scores_gpu,
                                             batch, num_heads, seq_len, head_dim, scale);

    // Step 6: Apply causal mask and softmax
    dim3 softmax_grid(seq_len, num_heads, batch);
    masked_softmax_kernel<<<softmax_grid, WARP_SIZE>>>(attn_scores_gpu, batch, num_heads, seq_len);

    // Step 7: Multiply attention weights by V
    dim3 output_attn_grid(seq_len, num_heads, batch);
    attn_output_kernel<<<output_attn_grid, 128>>>(attn_scores_gpu, v_transposed_gpu, attn_out_gpu,
                                                  batch, num_heads, seq_len, head_dim);

    // Step 8: Transpose back to (batch, seq, num_heads, head_dim)
    int total_trans = batch * num_heads * seq_len * head_dim;
    int threads_trans = 256;
    int blocks_trans = (total_trans + threads_trans - 1) / threads_trans;
    transpose_kernel<<<blocks_trans, threads_trans>>>(attn_out_gpu, attn_out_transposed_gpu,
                                                      batch, num_heads, seq_len, head_dim);

    // Step 9: Output projection
    int M_o = batch * seq_len;
    int N_o = hidden_size;
    int K_o = hidden_size;
    dim3 grid_o((N_o + TILE_SIZE - 1) / TILE_SIZE, (M_o + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid_o, block_tile>>>(attn_out_transposed_gpu, o_proj_gpu, output_gpu, M_o, N_o, K_o);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void attn_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(cos_gpu));
    CHECK_CUDA(cudaFree(sin_gpu));
    CHECK_CUDA(cudaFree(q_proj_gpu));
    CHECK_CUDA(cudaFree(k_proj_gpu));
    CHECK_CUDA(cudaFree(v_proj_gpu));
    CHECK_CUDA(cudaFree(o_proj_gpu));
    CHECK_CUDA(cudaFree(q_norm_gpu));
    CHECK_CUDA(cudaFree(k_norm_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
    CHECK_CUDA(cudaFree(q_proj_out_gpu));
    CHECK_CUDA(cudaFree(k_proj_out_gpu));
    CHECK_CUDA(cudaFree(v_proj_out_gpu));
    CHECK_CUDA(cudaFree(q_normed_gpu));
    CHECK_CUDA(cudaFree(k_normed_gpu));
    CHECK_CUDA(cudaFree(q_transposed_gpu));
    CHECK_CUDA(cudaFree(k_transposed_gpu));
    CHECK_CUDA(cudaFree(k_repeated_gpu));
    CHECK_CUDA(cudaFree(v_transposed_gpu));
    CHECK_CUDA(cudaFree(attn_scores_gpu));
    CHECK_CUDA(cudaFree(attn_out_gpu));
    CHECK_CUDA(cudaFree(attn_out_transposed_gpu));
}
