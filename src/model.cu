#include "model.h"
#include "model_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cstring>
#include <cuda_runtime.h>
#include <mpi.h>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// Global parallel context
ParallelContext g_parallel_ctx;

// ============================================================================
// Parallel Context Implementation
// ============================================================================

void ParallelContext::init(int rank, int size) {
    world_rank = rank;
    world_size = size;

    // Determine local rank within node (assuming 4 GPUs per node)
    local_rank = rank % NUM_GPUS_PER_NODE;
    node_rank = rank / NUM_GPUS_PER_NODE;
    num_nodes = (size + NUM_GPUS_PER_NODE - 1) / NUM_GPUS_PER_NODE;

    // Create communicator for processes within same node
    MPI_Comm_split(MPI_COMM_WORLD, node_rank, local_rank, &node_comm);

    // Set GPU device based on local rank
    CHECK_CUDA(cudaSetDevice(local_rank));

    // Create CUDA streams for async operations
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&h2d_stream));
    CHECK_CUDA(cudaStreamCreate(&d2h_stream));

    // Calculate expert assignment for this GPU
    expert_start = local_rank * NUM_EXPERTS_PER_GPU;
    expert_end = expert_start + NUM_EXPERTS_PER_GPU;

    if (world_rank == 0) {
        std::cout << "Parallel Context Initialized:" << std::endl;
        std::cout << "  World size: " << world_size << std::endl;
        std::cout << "  Num nodes: " << num_nodes << std::endl;
        std::cout << "  GPUs per node: " << NUM_GPUS_PER_NODE << std::endl;
        std::cout << "  Experts per GPU: " << NUM_EXPERTS_PER_GPU << std::endl;
    }
}

void ParallelContext::finalize() {
    // Destroy CUDA streams
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(h2d_stream));
    CHECK_CUDA(cudaStreamDestroy(d2h_stream));

    MPI_Comm_free(&node_comm);
}

// ============================================================================
// CUDA Kernels for Model Operations
// ============================================================================

#define BLOCK_SIZE 256
#define TILE_SIZE 16

// Embedding lookup kernel
__global__ void embedding_lookup_kernel(const float* embed_table, const int* input_ids,
                                         float* output, size_t seq_len, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = seq_len * hidden_size;

    if (idx < total) {
        size_t pos = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int token_id = input_ids[pos];
        output[idx] = embed_table[token_id * hidden_size + dim];
    }
}

// Tensor transpose kernel: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
__global__ void transpose_0213_kernel(const float* input, float* output,
                                       size_t batch, size_t seq_len, size_t num_heads, size_t head_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * num_heads * seq_len * head_dim;

    if (idx < total) {
        size_t d = idx % head_dim;
        size_t remaining = idx / head_dim;
        size_t s = remaining % seq_len;
        remaining = remaining / seq_len;
        size_t h = remaining % num_heads;
        size_t b = remaining / num_heads;

        size_t in_idx = ((b * seq_len + s) * num_heads + h) * head_dim + d;
        output[idx] = input[in_idx];
    }
}

// Reshape projection output kernel
__global__ void reshape_proj_output_kernel(const float* input, float* output,
                                            size_t batch, size_t seq_len, size_t num_heads, size_t head_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * num_heads * head_dim;

    if (idx < total) {
        size_t d = idx % head_dim;
        size_t remaining = idx / head_dim;
        size_t h = remaining % num_heads;
        remaining = remaining / num_heads;
        size_t s = remaining % seq_len;
        size_t b = remaining / seq_len;

        size_t in_idx = (b * seq_len + s) * (num_heads * head_dim) + h * head_dim + d;
        output[idx] = input[in_idx];
    }
}

// Flatten attention output kernel
__global__ void flatten_attn_output_kernel(const float* input, float* output,
                                            size_t batch, size_t num_heads, size_t seq_len, size_t head_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * num_heads * head_dim;

    if (idx < total) {
        size_t flat_idx = idx;
        size_t hidden_size = num_heads * head_dim;
        size_t token_idx = flat_idx / hidden_size;
        size_t h_d = flat_idx % hidden_size;
        size_t h = h_d / head_dim;
        size_t d = h_d % head_dim;

        size_t b = token_idx / seq_len;
        size_t s = token_idx % seq_len;

        size_t in_idx = ((b * num_heads + h) * seq_len + s) * head_dim + d;
        output[flat_idx] = input[in_idx];
    }
}

// Batched attention scores kernel
__global__ void batched_attn_scores_kernel(const float* q, const float* k, float* scores,
                                            size_t batch, size_t num_heads, size_t seq_len,
                                            size_t head_dim, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * num_heads * seq_len * seq_len;

    if (idx < total) {
        size_t j = idx % seq_len;
        size_t remaining = idx / seq_len;
        size_t i = remaining % seq_len;
        remaining = remaining / seq_len;
        size_t h = remaining % num_heads;
        size_t b = remaining / num_heads;

        float sum = 0.0f;
        for (size_t d = 0; d < head_dim; d++) {
            sum += q[((b * num_heads + h) * seq_len + i) * head_dim + d] *
                   k[((b * num_heads + h) * seq_len + j) * head_dim + d];
        }
        scores[idx] = sum * scale;
    }
}

// Apply causal mask kernel
__global__ void apply_causal_mask_kernel(float* scores, size_t batch, size_t num_heads, size_t seq_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * num_heads * seq_len * seq_len;

    if (idx < total) {
        size_t j = idx % seq_len;
        size_t remaining = idx / seq_len;
        size_t i = remaining % seq_len;

        if (j > i) {
            scores[idx] = -INFINITY;
        }
    }
}

// Batched softmax kernel for attention
__global__ void batched_softmax_kernel(float* scores, size_t batch, size_t num_heads, size_t seq_len) {
    extern __shared__ float shared[];

    size_t bh = blockIdx.x;
    size_t i = blockIdx.y;
    if (bh >= batch * num_heads || i >= seq_len) return;

    float* row = scores + (bh * seq_len + i) * seq_len;

    float max_val = -INFINITY;
    for (size_t j = threadIdx.x; j < seq_len; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j]);
    }
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

    float sum = 0.0f;
    for (size_t j = threadIdx.x; j < seq_len; j += blockDim.x) {
        float exp_val = expf(row[j] - max_val);
        row[j] = exp_val;
        sum += exp_val;
    }
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

    float inv_sum = 1.0f / sum;
    for (size_t j = threadIdx.x; j < seq_len; j += blockDim.x) {
        row[j] *= inv_sum;
    }
}

// Batched attention output kernel
__global__ void batched_attn_output_kernel(const float* weights, const float* v, float* output,
                                            size_t batch, size_t num_heads, size_t seq_len, size_t head_dim) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * num_heads * seq_len * head_dim;

    if (idx < total) {
        size_t d = idx % head_dim;
        size_t remaining = idx / head_dim;
        size_t i = remaining % seq_len;
        remaining = remaining / seq_len;
        size_t h = remaining % num_heads;
        size_t b = remaining / num_heads;

        float sum = 0.0f;
        for (size_t j = 0; j < seq_len; j++) {
            sum += weights[(b * num_heads + h) * seq_len * seq_len + i * seq_len + j] *
                   v[((b * num_heads + h) * seq_len + j) * head_dim + d];
        }
        output[idx] = sum;
    }
}

// Transpose for conv
__global__ void transpose_conv_kernel(const float* input, float* output,
                                       size_t batch, size_t seq_len, size_t channels) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * channels * seq_len;

    if (idx < total) {
        size_t s = idx % seq_len;
        size_t remaining = idx / seq_len;
        size_t c = remaining % channels;
        size_t b = remaining / channels;

        size_t in_idx = (b * seq_len + s) * channels + c;
        output[idx] = input[in_idx];
    }
}

// Reverse transpose for conv
__global__ void transpose_conv_reverse_kernel(const float* input, float* output,
                                               size_t batch, size_t channels, size_t seq_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * seq_len * channels;

    if (idx < total) {
        size_t c = idx % channels;
        size_t remaining = idx / channels;
        size_t s = remaining % seq_len;
        size_t b = remaining / seq_len;

        size_t in_idx = (b * channels + c) * seq_len + s;
        output[idx] = input[in_idx];
    }
}

// Add bias kernel
__global__ void add_bias_kernel(float* data, const float* bias, size_t rows, size_t cols) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        size_t col = idx % cols;
        data[idx] += bias[col];
    }
}

// Split BCx kernel
__global__ void split_bcx_kernel(const float* bcx, float* B, float* C, float* x_gate,
                                  size_t batch, size_t hidden_size, size_t seq_len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch * hidden_size * seq_len;

    if (idx < total) {
        size_t s = idx % seq_len;
        size_t remaining = idx / seq_len;
        size_t h = remaining % hidden_size;
        size_t b = remaining / hidden_size;

        size_t bcx_idx_B = (b * 3 * hidden_size + h) * seq_len + s;
        size_t bcx_idx_C = (b * 3 * hidden_size + h + hidden_size) * seq_len + s;
        size_t bcx_idx_x = (b * 3 * hidden_size + h + 2 * hidden_size) * seq_len + s;

        B[idx] = bcx[bcx_idx_B];
        C[idx] = bcx[bcx_idx_C];
        x_gate[idx] = bcx[bcx_idx_x];
    }
}

// Copy last token kernel (single batch)
__global__ void copy_last_token_kernel(const float* input, float* output,
                                        size_t seq_len, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        output[idx] = input[(seq_len - 1) * hidden_size + idx];
    }
}

// Copy last token kernel (batched version)
// input: [batch, seq_len, hidden_size]
// output: [batch, hidden_size]
__global__ void copy_last_token_batched_kernel(const float* input, float* output,
                                                size_t batch_size, size_t seq_len, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * hidden_size;

    if (idx < total) {
        size_t b = idx / hidden_size;
        size_t d = idx % hidden_size;
        // Last token of each batch: input[b, seq_len-1, d]
        output[idx] = input[(b * seq_len + (seq_len - 1)) * hidden_size + d];
    }
}

// Batched embedding lookup kernel
// input_ids: [batch_size * seq_len]
// output: [batch_size, seq_len, hidden_size]
__global__ void embedding_lookup_batched_kernel(const float* embed_table, const int* input_ids,
                                                 float* output, size_t batch_size, size_t seq_len,
                                                 size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * seq_len * hidden_size;

    if (idx < total) {
        size_t d = idx % hidden_size;
        size_t pos = idx / hidden_size;  // position in [batch_size * seq_len]
        int token_id = input_ids[pos];
        output[idx] = embed_table[token_id * hidden_size + d];
    }
}

// MoE routing kernels
__global__ void compute_routing_weights_kernel(const float* router_logits, float* routing_weights,
                                                size_t num_tokens, size_t num_experts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_tokens * num_experts) {
        float logit = router_logits[idx];
        float inv_denom = 1.0f / (1.0f + expf(-logit));
        routing_weights[idx] = 1.0f * inv_denom;
    }
}

// Top-k selection kernel
__global__ void topk_routing_kernel(const float* routing_weights, const float* expert_bias,
                                     int* top_k_indices, float* top_k_weights,
                                     size_t num_tokens, size_t num_experts, size_t k,
                                     bool use_bias, bool norm_prob, float scaling_factor) {
    extern __shared__ float shared[];
    float* scores = shared;
    int* indices = (int*)(shared + num_experts);

    size_t t = blockIdx.x;
    if (t >= num_tokens) return;

    for (size_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
        float weight = routing_weights[t * num_experts + e];
        float score = weight;
        if (use_bias && expert_bias != nullptr) {
            score += expert_bias[e];
        }
        scores[e] = score;
        indices[e] = e;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (size_t i = 0; i < k; i++) {
            size_t max_idx = i;
            for (size_t j = i + 1; j < num_experts; j++) {
                if (scores[j] > scores[max_idx]) {
                    max_idx = j;
                }
            }
            float tmp_score = scores[i];
            scores[i] = scores[max_idx];
            scores[max_idx] = tmp_score;
            int tmp_idx = indices[i];
            indices[i] = indices[max_idx];
            indices[max_idx] = tmp_idx;
        }

        float selected_weights[8];
        float sum = 0.0f;
        for (size_t i = 0; i < k; i++) {
            int expert_idx = indices[i];
            selected_weights[i] = routing_weights[t * num_experts + expert_idx];
            sum += selected_weights[i];
        }

        float inv_sum = (norm_prob && sum > 1e-6f) ? (1.0f / sum) : 1.0f;
        for (size_t i = 0; i < k; i++) {
            top_k_indices[t * k + i] = indices[i];
            float weight = selected_weights[i];
            if (norm_prob && sum > 1e-6f) {
                weight *= inv_sum;
            }
            top_k_weights[t * k + i] = weight * scaling_factor;
        }
    }
}

// Expert output accumulation kernel
__global__ void accumulate_expert_output_kernel(float* y, const float* expert_out, float weight,
                                                 size_t offset, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        atomicAdd(&y[offset + idx], weight * expert_out[idx]);
    }
}

// Copy token input kernel
__global__ void copy_token_input_kernel(const float* x_flat, float* token_in,
                                         size_t token_idx, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < hidden_size) {
        token_in[idx] = x_flat[token_idx * hidden_size + idx];
    }
}

// Batched gather kernel: gather multiple tokens in parallel
// token_indices: array of token indices to gather
// batch_size: number of tokens to gather
__global__ void batched_gather_kernel(const float* x_flat, float* batched_input,
                                       const int* token_indices, size_t batch_size,
                                       size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * hidden_size;

    if (idx < total) {
        size_t token_batch_idx = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int src_token = token_indices[token_batch_idx];
        batched_input[idx] = x_flat[src_token * hidden_size + dim];
    }
}

// Batched scatter-accumulate kernel: scatter and accumulate multiple expert outputs in parallel
// Uses atomicAdd since different tokens may map to the same output position
__global__ void batched_scatter_accumulate_kernel(float* y, const float* batched_output,
                                                   const int* token_indices, const float* weights,
                                                   size_t batch_size, size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * hidden_size;

    if (idx < total) {
        size_t token_batch_idx = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int dst_token = token_indices[token_batch_idx];
        float weight = weights[token_batch_idx];
        atomicAdd(&y[dst_token * hidden_size + dim], weight * batched_output[idx]);
    }
}

// Count tokens per expert for a specific GPU's local experts
// expert_start: first global expert index for this GPU
// num_local_experts: number of experts on this GPU
__global__ void count_tokens_per_expert_kernel(const int* top_k_indices, int* expert_counts,
                                                size_t num_tokens, size_t k,
                                                int expert_start, int num_local_experts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_tokens * k;

    if (idx < total) {
        int expert_idx = top_k_indices[idx];
        // Check if expert belongs to this GPU
        int local_expert = expert_idx - expert_start;
        if (local_expert >= 0 && local_expert < num_local_experts) {
            atomicAdd(&expert_counts[local_expert], 1);
        }
    }
}

// Build sorted indices for each expert
// Given top_k_indices and offsets, populate sorted_indices and sorted_weights
__global__ void build_expert_indices_kernel(const int* top_k_indices, const float* top_k_weights,
                                             int* sorted_indices, float* sorted_weights,
                                             int* expert_write_pos, const int* expert_offsets,
                                             size_t num_tokens, size_t k,
                                             int expert_start, int num_local_experts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = num_tokens * k;

    if (idx < total) {
        int expert_idx = top_k_indices[idx];
        int local_expert = expert_idx - expert_start;

        if (local_expert >= 0 && local_expert < num_local_experts) {
            // Get write position for this expert using atomicAdd
            int write_pos = atomicAdd(&expert_write_pos[local_expert], 1);
            int offset = expert_offsets[local_expert];

            // Token index is idx / k (which token this routing entry belongs to)
            int token_idx = idx / k;

            sorted_indices[offset + write_pos] = token_idx;
            sorted_weights[offset + write_pos] = top_k_weights[idx];
        }
    }
}

// ============================================================================
// Grouped Expert GEMM Kernels
// ============================================================================

// Tile sizes for grouped expert GEMM
#define EXPERT_BM 64
#define EXPERT_BN 64
#define EXPERT_BK 8
#define EXPERT_TM 4
#define EXPERT_TN 4

// Grouped gather kernel: gather all tokens for all experts in one kernel
// input: [num_tokens, hidden_size]
// output: [total_expert_tokens, hidden_size] (tokens sorted by expert)
// sorted_indices: [total_expert_tokens] (maps output position to input token)
__global__ void grouped_gather_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       const int* __restrict__ sorted_indices,
                                       size_t total_tokens,
                                       size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = total_tokens * hidden_size;

    if (idx < total) {
        size_t out_token = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int in_token = sorted_indices[out_token];
        output[idx] = input[in_token * hidden_size + dim];
    }
}

// Grouped scatter-accumulate kernel: scatter all expert outputs back
// expert_output: [total_expert_tokens, hidden_size]
// output: [num_tokens, hidden_size]
// sorted_indices: maps expert_output position to output token
// sorted_weights: routing weights for each expert_output
__global__ void grouped_scatter_accumulate_kernel(float* __restrict__ output,
                                                   const float* __restrict__ expert_output,
                                                   const int* __restrict__ sorted_indices,
                                                   const float* __restrict__ sorted_weights,
                                                   size_t total_tokens,
                                                   size_t hidden_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = total_tokens * hidden_size;

    if (idx < total) {
        size_t token_idx = idx / hidden_size;
        size_t dim = idx % hidden_size;
        int out_token = sorted_indices[token_idx];
        float weight = sorted_weights[token_idx];
        atomicAdd(&output[out_token * hidden_size + dim], weight * expert_output[idx]);
    }
}

// Grouped expert GEMM kernel for gate/up projections
// Processes all local experts in parallel using 3D grid
// input: [total_tokens, hidden_size] - tokens sorted by expert
// weight: [num_experts, out_features, in_features] - stacked expert weights
// output: [total_tokens, out_features]
// expert_offsets: [num_experts+1] - start position of each expert's tokens
__global__ void grouped_expert_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    const int* __restrict__ expert_offsets,
    int num_experts,
    int in_features,
    int out_features) {

    // blockIdx.z = expert_id
    int expert_id = blockIdx.z;
    if (expert_id >= num_experts) return;

    int token_start = expert_offsets[expert_id];
    int token_end = expert_offsets[expert_id + 1];
    int num_tokens = token_end - token_start;
    if (num_tokens == 0) return;

    // Thread block tile position
    int block_row = blockIdx.y * EXPERT_BM;  // Row in output (token dimension)
    int block_col = blockIdx.x * EXPERT_BN;  // Column in output (out_features dimension)

    // Skip if this block is outside the token range for this expert
    if (block_row >= num_tokens) return;

    // Thread position within block
    int tx = threadIdx.x;  // 0-15
    int ty = threadIdx.y;  // 0-15
    int thread_row = ty * EXPERT_TM;
    int thread_col = tx * EXPERT_TN;

    // Pointers to this expert's data
    const float* expert_input = input + token_start * in_features;
    const float* expert_weight = weight + expert_id * out_features * in_features;
    float* expert_output = output + token_start * out_features;

    // Shared memory for tiling
    __shared__ float As[EXPERT_BK][EXPERT_BM];  // Input tile (transposed for coalescing)
    __shared__ float Bs[EXPERT_BK][EXPERT_BN];  // Weight tile

    // Register accumulation
    float reg_c[EXPERT_TM][EXPERT_TN] = {0.0f};
    float reg_a[EXPERT_TM];
    float reg_b[EXPERT_TN];

    int num_threads = (EXPERT_BM / EXPERT_TM) * (EXPERT_BN / EXPERT_TN);  // 256
    int tid = ty * (EXPERT_BN / EXPERT_TN) + tx;

    int num_k_tiles = (in_features + EXPERT_BK - 1) / EXPERT_BK;

    for (int tile = 0; tile < num_k_tiles; tile++) {
        // Load input tile (A): [BM, BK] from input[block_row:, tile*BK:]
        // Transposed storage: As[k][m]
        int a_elements = EXPERT_BM * EXPERT_BK;
        for (int i = tid; i < a_elements; i += num_threads) {
            int m = i / EXPERT_BK;
            int k = i % EXPERT_BK;
            int global_m = block_row + m;
            int global_k = tile * EXPERT_BK + k;
            if (global_m < num_tokens && global_k < in_features) {
                As[k][m] = expert_input[global_m * in_features + global_k];
            } else {
                As[k][m] = 0.0f;
            }
        }

        // Load weight tile (B): [BK, BN] from weight[tile*BK:, block_col:]
        // weight is [out_features, in_features], we want weight^T @ input^T
        // Actually we compute input @ weight^T, so we load weight[block_col:, tile*BK:]
        int b_elements = EXPERT_BK * EXPERT_BN;
        for (int i = tid; i < b_elements; i += num_threads) {
            int k = i / EXPERT_BN;
            int n = i % EXPERT_BN;
            int global_k = tile * EXPERT_BK + k;
            int global_n = block_col + n;
            if (global_k < in_features && global_n < out_features) {
                // weight[global_n, global_k] for transposed access
                Bs[k][n] = expert_weight[global_n * in_features + global_k];
            } else {
                Bs[k][n] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial results
        #pragma unroll
        for (int kk = 0; kk < EXPERT_BK; kk++) {
            #pragma unroll
            for (int i = 0; i < EXPERT_TM; i++) {
                reg_a[i] = As[kk][thread_row + i];
            }
            #pragma unroll
            for (int j = 0; j < EXPERT_TN; j++) {
                reg_b[j] = Bs[kk][thread_col + j];
            }
            #pragma unroll
            for (int i = 0; i < EXPERT_TM; i++) {
                #pragma unroll
                for (int j = 0; j < EXPERT_TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    #pragma unroll
    for (int i = 0; i < EXPERT_TM; i++) {
        int global_m = block_row + thread_row + i;
        if (global_m < num_tokens) {
            #pragma unroll
            for (int j = 0; j < EXPERT_TN; j++) {
                int global_n = block_col + thread_col + j;
                if (global_n < out_features) {
                    expert_output[global_m * out_features + global_n] = reg_c[i][j];
                }
            }
        }
    }
}

// Fused SiLU + Mul kernel for grouped experts
// gate: [total_tokens, intermediate_size]
// up: [total_tokens, intermediate_size]
// output: [total_tokens, intermediate_size]
__global__ void grouped_silu_mul_kernel(const float* __restrict__ gate,
                                         const float* __restrict__ up,
                                         float* __restrict__ output,
                                         size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * up[idx];
    }
}

// ============================================================================
// MLP Implementation
// ============================================================================

MLP::MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file) {
    w1_ = Tensor::load_from_file(w1_file);
    w2_ = Tensor::load_from_file(w2_file);
    w3_ = Tensor::load_from_file(w3_file);
}

void MLP::forward(const Tensor& x, Tensor& y) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t intermediate_size = w1_.size(0);

    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // Gate projection: gate = x @ w1^T
    Tensor gate({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w1_, gate);

    // Up projection: up = x @ w3^T
    Tensor up({batch * seq_len, intermediate_size});
    tensor_ops::matmul_transposed(x_flat, w3_, up);

    // Fused SiLU + Mul: hidden = silu(gate) * up
    Tensor hidden({batch * seq_len, intermediate_size});
    tensor_ops::silu_mul(gate, up, hidden);

    // Down projection directly into y
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    Tensor y_flat = y.view({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(hidden, w2_, y_flat);
}

// ============================================================================
// SparseMoeBlock Implementation with Grouped Expert GEMM
// ============================================================================

SparseMoeBlock::SparseMoeBlock(int layer_idx)
    : layer_idx_(layer_idx), d_top_k_indices_(nullptr), d_top_k_weights_(nullptr), routing_buffer_size_(0),
      d_gather_indices_(nullptr), d_scatter_weights_(nullptr), gather_scatter_buffer_size_(0),
      d_expert_counts_(nullptr), d_expert_offsets_(nullptr), d_expert_write_pos_(nullptr),
      d_sorted_indices_(nullptr), d_sorted_weights_(nullptr),
      d_expert_input_(nullptr), d_expert_output_(nullptr), d_gate_buf_(nullptr), d_up_buf_(nullptr),
      expert_buffer_size_(0) {
    std::stringstream ss;
    ss << "layers." << layer_idx << ".feed_forward.gate.weight";
    gate_ = Tensor::load_from_file(ss.str());

    // Only load experts assigned to this GPU
    int expert_start = g_parallel_ctx.expert_start;
    int expert_end = g_parallel_ctx.expert_end;

    local_to_global_expert_.reserve(NUM_EXPERTS_PER_GPU);

    // Load and stack expert weights for grouped GEMM
    // Shape: [NUM_EXPERTS_PER_GPU, out_dim, in_dim]
    stacked_w1_ = Tensor({NUM_EXPERTS_PER_GPU, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE});
    stacked_w2_ = Tensor({NUM_EXPERTS_PER_GPU, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE});
    stacked_w3_ = Tensor({NUM_EXPERTS_PER_GPU, MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE});

    for (int e = expert_start; e < expert_end; e++) {
        int local_idx = e - expert_start;
        local_to_global_expert_.push_back(e);

        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << e << ".w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << e << ".w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << e << ".w3.weight";

        // Load individual expert weights
        Tensor w1 = Tensor::load_from_file(ss_w1.str());
        Tensor w2 = Tensor::load_from_file(ss_w2.str());
        Tensor w3 = Tensor::load_from_file(ss_w3.str());

        // Copy to stacked tensor at the correct offset
        size_t w1_offset = local_idx * MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE;
        size_t w2_offset = local_idx * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE;
        size_t w3_offset = local_idx * MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE;

        CHECK_CUDA(cudaMemcpy(stacked_w1_.data() + w1_offset, w1.data(),
                              MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(stacked_w2_.data() + w2_offset, w2.data(),
                              HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE * sizeof(float),
                              cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(stacked_w3_.data() + w3_offset, w3.data(),
                              MOE_INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float),
                              cudaMemcpyDeviceToDevice));
    }

    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str());
    }

    // Pre-allocate routing buffers for typical batch size
    ensure_routing_buffers(64);
    ensure_gather_scatter_buffers(64);
    ensure_expert_buffers(256);  // Initial size for intermediate buffers
}

void SparseMoeBlock::ensure_routing_buffers(size_t num_tokens) {
    if (num_tokens <= routing_buffer_size_) return;

    // Free old buffers if they exist
    if (d_top_k_indices_ != nullptr) cudaFree(d_top_k_indices_);
    if (d_top_k_weights_ != nullptr) cudaFree(d_top_k_weights_);
    if (d_expert_counts_ != nullptr) cudaFree(d_expert_counts_);
    if (d_expert_offsets_ != nullptr) cudaFree(d_expert_offsets_);
    if (d_expert_write_pos_ != nullptr) cudaFree(d_expert_write_pos_);
    if (d_sorted_indices_ != nullptr) cudaFree(d_sorted_indices_);
    if (d_sorted_weights_ != nullptr) cudaFree(d_sorted_weights_);

    // Allocate new buffers with some headroom
    size_t new_size = num_tokens * 2;  // 2x headroom
    CHECK_CUDA(cudaMalloc(&d_top_k_indices_, new_size * NUM_EXPERTS_PER_TOK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_top_k_weights_, new_size * NUM_EXPERTS_PER_TOK * sizeof(float)));

    // Allocate GPU-only routing buffers
    CHECK_CUDA(cudaMalloc(&d_expert_counts_, NUM_EXPERTS_PER_GPU * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_offsets_, (NUM_EXPERTS_PER_GPU + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_expert_write_pos_, NUM_EXPERTS_PER_GPU * sizeof(int)));
    // Max possible: every token could go to every local expert with top-k routing
    CHECK_CUDA(cudaMalloc(&d_sorted_indices_, new_size * NUM_EXPERTS_PER_TOK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sorted_weights_, new_size * NUM_EXPERTS_PER_TOK * sizeof(float)));

    routing_buffer_size_ = new_size;
}

void SparseMoeBlock::ensure_gather_scatter_buffers(size_t max_tokens_per_expert) {
    if (max_tokens_per_expert <= gather_scatter_buffer_size_) return;

    // Free old buffers if they exist
    if (d_gather_indices_ != nullptr) {
        cudaFree(d_gather_indices_);
    }
    if (d_scatter_weights_ != nullptr) {
        cudaFree(d_scatter_weights_);
    }

    // Allocate new buffers with some headroom
    size_t new_size = max_tokens_per_expert * 2;
    CHECK_CUDA(cudaMalloc(&d_gather_indices_, new_size * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_scatter_weights_, new_size * sizeof(float)));
    gather_scatter_buffer_size_ = new_size;
}

void SparseMoeBlock::ensure_expert_buffers(size_t total_tokens) {
    if (total_tokens <= expert_buffer_size_) return;

    // Free old buffers if they exist
    if (d_expert_input_ != nullptr) cudaFree(d_expert_input_);
    if (d_expert_output_ != nullptr) cudaFree(d_expert_output_);
    if (d_gate_buf_ != nullptr) cudaFree(d_gate_buf_);
    if (d_up_buf_ != nullptr) cudaFree(d_up_buf_);

    // Allocate new buffers with some headroom
    size_t new_size = total_tokens * 2;
    CHECK_CUDA(cudaMalloc(&d_expert_input_, new_size * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_expert_output_, new_size * HIDDEN_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_gate_buf_, new_size * MOE_INTERMEDIATE_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_up_buf_, new_size * MOE_INTERMEDIATE_SIZE * sizeof(float)));
    expert_buffer_size_ = new_size;
}

void SparseMoeBlock::route_tokens_gpu(const Tensor& router_logits, size_t num_tokens) {
    // Ensure pre-allocated buffers are large enough
    ensure_routing_buffers(num_tokens);

    Tensor routing_weights({num_tokens, NUM_EXPERTS});

    int blocks = (num_tokens * NUM_EXPERTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_routing_weights_kernel<<<blocks, BLOCK_SIZE>>>(
        router_logits.data(), routing_weights.data(), num_tokens, NUM_EXPERTS);
    CHECK_CUDA(cudaGetLastError());

    size_t shared_mem = NUM_EXPERTS * sizeof(float) + NUM_EXPERTS * sizeof(int);
    topk_routing_kernel<<<num_tokens, 32, shared_mem>>>(
        routing_weights.data(),
        USE_EXPERT_BIAS ? expert_bias_.data() : nullptr,
        d_top_k_indices_, d_top_k_weights_,
        num_tokens, NUM_EXPERTS, NUM_EXPERTS_PER_TOK,
        USE_EXPERT_BIAS, NORM_TOPK_PROB, ROUTED_SCALING_FACTOR);
    CHECK_CUDA(cudaGetLastError());

    // Count tokens per local expert (GPU kernel)
    CHECK_CUDA(cudaMemset(d_expert_counts_, 0, NUM_EXPERTS_PER_GPU * sizeof(int)));
    int count_blocks = (num_tokens * NUM_EXPERTS_PER_TOK + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_tokens_per_expert_kernel<<<count_blocks, BLOCK_SIZE>>>(
        d_top_k_indices_, d_expert_counts_,
        num_tokens, NUM_EXPERTS_PER_TOK,
        g_parallel_ctx.expert_start, NUM_EXPERTS_PER_GPU);
    CHECK_CUDA(cudaGetLastError());
}

void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t num_tokens = batch * seq_len;

    Tensor x_flat = x.view({num_tokens, hidden_size});

    // Compute router logits (all GPUs compute this)
    router_logits = Tensor({num_tokens, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits);

    // Route tokens entirely on GPU (no D2H sync for routing)
    route_tokens_gpu(router_logits, num_tokens);

    // Initialize output
    y = Tensor({batch, seq_len, hidden_size});
    y.zero();

    // Copy expert counts to host (small transfer: NUM_EXPERTS_PER_GPU ints)
    int h_expert_counts[NUM_EXPERTS_PER_GPU];
    int h_expert_offsets[NUM_EXPERTS_PER_GPU + 1];
    CHECK_CUDA(cudaMemcpy(h_expert_counts, d_expert_counts_,
                          NUM_EXPERTS_PER_GPU * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute prefix sums (offsets) on host
    h_expert_offsets[0] = 0;
    int total_expert_tokens = 0;
    for (int i = 0; i < NUM_EXPERTS_PER_GPU; i++) {
        h_expert_offsets[i + 1] = h_expert_offsets[i] + h_expert_counts[i];
        total_expert_tokens += h_expert_counts[i];
    }

    // Skip if no tokens routed to this GPU's experts
    if (total_expert_tokens == 0) {
        goto allreduce;
    }

    // Copy offsets to GPU
    CHECK_CUDA(cudaMemcpy(d_expert_offsets_, h_expert_offsets,
                          (NUM_EXPERTS_PER_GPU + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Reset write positions to 0 (will be used atomically)
    CHECK_CUDA(cudaMemset(d_expert_write_pos_, 0, NUM_EXPERTS_PER_GPU * sizeof(int)));

    // Build sorted indices on GPU
    {
        int build_blocks = (num_tokens * NUM_EXPERTS_PER_TOK + BLOCK_SIZE - 1) / BLOCK_SIZE;
        build_expert_indices_kernel<<<build_blocks, BLOCK_SIZE>>>(
            d_top_k_indices_, d_top_k_weights_,
            d_sorted_indices_, d_sorted_weights_,
            d_expert_write_pos_, d_expert_offsets_,
            num_tokens, NUM_EXPERTS_PER_TOK,
            g_parallel_ctx.expert_start, NUM_EXPERTS_PER_GPU);
        CHECK_CUDA(cudaGetLastError());
    }

    // Ensure intermediate buffers are large enough
    ensure_expert_buffers(total_expert_tokens);

    // ========================================================================
    // Grouped Expert GEMM: Process ALL experts in parallel
    // ========================================================================

    // Step 1: Gather all tokens for all experts (single kernel)
    {
        int gather_blocks = (total_expert_tokens * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grouped_gather_kernel<<<gather_blocks, BLOCK_SIZE>>>(
            x_flat.data(), d_expert_input_, d_sorted_indices_,
            total_expert_tokens, hidden_size);
        CHECK_CUDA(cudaGetLastError());
    }

    // Step 2: Gate projection for all experts (grouped GEMM)
    // input @ w1^T: [total_tokens, hidden] @ [experts, intermediate, hidden]^T
    {
        // Find max tokens per expert for grid sizing
        int max_tokens = 0;
        for (int i = 0; i < NUM_EXPERTS_PER_GPU; i++) {
            if (h_expert_counts[i] > max_tokens) max_tokens = h_expert_counts[i];
        }

        dim3 block(EXPERT_BN / EXPERT_TN, EXPERT_BM / EXPERT_TM);  // 16x16 = 256 threads
        dim3 grid(
            (MOE_INTERMEDIATE_SIZE + EXPERT_BN - 1) / EXPERT_BN,
            (max_tokens + EXPERT_BM - 1) / EXPERT_BM,
            NUM_EXPERTS_PER_GPU
        );

        grouped_expert_gemm_kernel<<<grid, block>>>(
            d_expert_input_, stacked_w1_.data(), d_gate_buf_,
            d_expert_offsets_, NUM_EXPERTS_PER_GPU,
            HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE);
        CHECK_CUDA(cudaGetLastError());
    }

    // Step 3: Up projection for all experts (grouped GEMM)
    // input @ w3^T
    {
        int max_tokens = 0;
        for (int i = 0; i < NUM_EXPERTS_PER_GPU; i++) {
            if (h_expert_counts[i] > max_tokens) max_tokens = h_expert_counts[i];
        }

        dim3 block(EXPERT_BN / EXPERT_TN, EXPERT_BM / EXPERT_TM);
        dim3 grid(
            (MOE_INTERMEDIATE_SIZE + EXPERT_BN - 1) / EXPERT_BN,
            (max_tokens + EXPERT_BM - 1) / EXPERT_BM,
            NUM_EXPERTS_PER_GPU
        );

        grouped_expert_gemm_kernel<<<grid, block>>>(
            d_expert_input_, stacked_w3_.data(), d_up_buf_,
            d_expert_offsets_, NUM_EXPERTS_PER_GPU,
            HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE);
        CHECK_CUDA(cudaGetLastError());
    }

    // Step 4: Fused SiLU + Mul (single kernel for all tokens)
    {
        size_t total_elements = total_expert_tokens * MOE_INTERMEDIATE_SIZE;
        int silu_blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grouped_silu_mul_kernel<<<silu_blocks, BLOCK_SIZE>>>(
            d_gate_buf_, d_up_buf_, d_gate_buf_, total_elements);
        CHECK_CUDA(cudaGetLastError());
    }

    // Step 5: Down projection for all experts (grouped GEMM)
    // hidden @ w2^T: [total_tokens, intermediate] @ [experts, hidden, intermediate]^T
    {
        int max_tokens = 0;
        for (int i = 0; i < NUM_EXPERTS_PER_GPU; i++) {
            if (h_expert_counts[i] > max_tokens) max_tokens = h_expert_counts[i];
        }

        dim3 block(EXPERT_BN / EXPERT_TN, EXPERT_BM / EXPERT_TM);
        dim3 grid(
            (HIDDEN_SIZE + EXPERT_BN - 1) / EXPERT_BN,
            (max_tokens + EXPERT_BM - 1) / EXPERT_BM,
            NUM_EXPERTS_PER_GPU
        );

        grouped_expert_gemm_kernel<<<grid, block>>>(
            d_gate_buf_, stacked_w2_.data(), d_expert_output_,
            d_expert_offsets_, NUM_EXPERTS_PER_GPU,
            MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE);
        CHECK_CUDA(cudaGetLastError());
    }

    // Step 6: Scatter weighted outputs back to y (single kernel)
    {
        int scatter_blocks = (total_expert_tokens * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grouped_scatter_accumulate_kernel<<<scatter_blocks, BLOCK_SIZE>>>(
            y.data(), d_expert_output_, d_sorted_indices_, d_sorted_weights_,
            total_expert_tokens, hidden_size);
        CHECK_CUDA(cudaGetLastError());
    }

allreduce:
    // Synchronize and reduce across GPUs within the node
    y.sync_to_host_async(g_parallel_ctx.d2h_stream);
    CHECK_CUDA(cudaStreamSynchronize(g_parallel_ctx.d2h_stream));
    y.mark_host_valid();
    float* y_host = y.host_data();

    // MPI_Allreduce within node to combine expert outputs
    MPI_Allreduce(MPI_IN_PLACE, y_host, num_tokens * hidden_size,
                  MPI_FLOAT, MPI_SUM, g_parallel_ctx.node_comm);

    // Async copy back to device
    CHECK_CUDA(cudaMemcpyAsync(y.data(), y_host, num_tokens * hidden_size * sizeof(float),
                               cudaMemcpyHostToDevice, g_parallel_ctx.h2d_stream));
    CHECK_CUDA(cudaStreamSynchronize(g_parallel_ctx.h2d_stream));
    y.mark_device_valid();
}

// ============================================================================
// Attention Implementation
// ============================================================================

Attention::Attention(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_q, ss_k, ss_v, ss_o, ss_q_ln, ss_k_ln;
    ss_q << "layers." << layer_idx << ".self_attn.q_proj.weight";
    ss_k << "layers." << layer_idx << ".self_attn.k_proj.weight";
    ss_v << "layers." << layer_idx << ".self_attn.v_proj.weight";
    ss_o << "layers." << layer_idx << ".self_attn.out_proj.weight";
    ss_q_ln << "layers." << layer_idx << ".self_attn.q_layernorm.weight";
    ss_k_ln << "layers." << layer_idx << ".self_attn.k_layernorm.weight";

    q_proj_ = Tensor::load_from_file(ss_q.str());
    k_proj_ = Tensor::load_from_file(ss_k.str());
    v_proj_ = Tensor::load_from_file(ss_v.str());
    o_proj_ = Tensor::load_from_file(ss_o.str());

    q_layernorm_ = std::make_unique<RMSNorm>(ss_q_ln.str());
    k_layernorm_ = std::make_unique<RMSNorm>(ss_k_ln.str());
}

void Attention::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                       const Tensor* attention_mask, Tensor& output) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    Tensor q_proj_out({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    Tensor k_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    Tensor v_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});

    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out);

    Tensor q_reshaped({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});

    size_t total_q = batch * seq_len * NUM_ATTENTION_HEADS * HEAD_DIM;
    size_t total_kv = batch * seq_len * NUM_KEY_VALUE_HEADS * HEAD_DIM;
    int blocks_q = (total_q + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_kv = (total_kv + BLOCK_SIZE - 1) / BLOCK_SIZE;

    reshape_proj_output_kernel<<<blocks_q, BLOCK_SIZE>>>(
        q_proj_out.data(), q_reshaped.data(), batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM);
    reshape_proj_output_kernel<<<blocks_kv, BLOCK_SIZE>>>(
        k_proj_out.data(), k_reshaped.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    reshape_proj_output_kernel<<<blocks_kv, BLOCK_SIZE>>>(
        v_proj_out.data(), v_reshaped.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    CHECK_CUDA(cudaGetLastError());

    Tensor q_normed({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_normed({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_reshaped, q_normed);
    k_layernorm_->forward(k_reshaped, k_normed);

    Tensor q({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor k({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    Tensor v({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});

    transpose_0213_kernel<<<blocks_q, BLOCK_SIZE>>>(
        q_normed.data(), q.data(), batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM);
    transpose_0213_kernel<<<blocks_kv, BLOCK_SIZE>>>(
        k_normed.data(), k.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    transpose_0213_kernel<<<blocks_kv, BLOCK_SIZE>>>(
        v_reshaped.data(), v.data(), batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM);
    CHECK_CUDA(cudaGetLastError());

    tensor_ops::apply_rotary_pos_emb(q, k, cos, sin);

    Tensor k_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor v_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k, NUM_KEY_VALUE_GROUPS, k_repeated);
    tensor_ops::repeat_kv(v, NUM_KEY_VALUE_GROUPS, v_repeated);

    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    Tensor scores({batch, NUM_ATTENTION_HEADS, seq_len, seq_len});

    size_t total_scores = batch * NUM_ATTENTION_HEADS * seq_len * seq_len;
    int blocks_scores = (total_scores + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batched_attn_scores_kernel<<<blocks_scores, BLOCK_SIZE>>>(
        q.data(), k_repeated.data(), scores.data(),
        batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM, scale);
    CHECK_CUDA(cudaGetLastError());

    apply_causal_mask_kernel<<<blocks_scores, BLOCK_SIZE>>>(
        scores.data(), batch, NUM_ATTENTION_HEADS, seq_len);
    CHECK_CUDA(cudaGetLastError());

    dim3 softmax_grid(batch * NUM_ATTENTION_HEADS, seq_len);
    int softmax_threads = std::min((int)seq_len, BLOCK_SIZE);
    size_t shared_mem = softmax_threads * sizeof(float);
    batched_softmax_kernel<<<softmax_grid, softmax_threads, shared_mem>>>(
        scores.data(), batch, NUM_ATTENTION_HEADS, seq_len);
    CHECK_CUDA(cudaGetLastError());

    Tensor attn_output({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    batched_attn_output_kernel<<<blocks_q, BLOCK_SIZE>>>(
        scores.data(), v_repeated.data(), attn_output.data(),
        batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM);
    CHECK_CUDA(cudaGetLastError());

    Tensor attn_flat({batch * seq_len, hidden_size});
    flatten_attn_output_kernel<<<blocks_q, BLOCK_SIZE>>>(
        attn_output.data(), attn_flat.data(), batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM);
    CHECK_CUDA(cudaGetLastError());

    Tensor output_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(attn_flat, o_proj_, output_flat);

    output_flat.reshape({batch, seq_len, hidden_size});

    if (output.size() == 0) {
        output = Tensor({batch, seq_len, hidden_size});
    }
    CHECK_CUDA(cudaMemcpy(output.data(), output_flat.data(), output.size() * sizeof(float), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// ShortConv Implementation
// ============================================================================

ShortConv::ShortConv(int layer_idx) : layer_idx_(layer_idx) {
    std::stringstream ss_conv, ss_in, ss_out;
    ss_conv << "layers." << layer_idx << ".conv.conv.weight";
    ss_in << "layers." << layer_idx << ".conv.in_proj.weight";
    ss_out << "layers." << layer_idx << ".conv.out_proj.weight";

    conv_weight_ = Tensor::load_from_file(ss_conv.str());
    in_proj_weight_ = Tensor::load_from_file(ss_in.str());
    out_proj_weight_ = Tensor::load_from_file(ss_out.str());

    if (USE_CONV_BIAS) {
        std::stringstream ss_conv_bias, ss_in_bias, ss_out_bias;
        ss_conv_bias << "layers." << layer_idx << ".conv.conv.bias";
        ss_in_bias << "layers." << layer_idx << ".conv.in_proj.bias";
        ss_out_bias << "layers." << layer_idx << ".conv.out_proj.bias";

        if (g_model_loader->has_tensor(ss_conv_bias.str())) {
            conv_bias_ = Tensor::load_from_file(ss_conv_bias.str());
        }
        if (g_model_loader->has_tensor(ss_in_bias.str())) {
            in_proj_bias_ = Tensor::load_from_file(ss_in_bias.str());
        }
        if (g_model_loader->has_tensor(ss_out_bias.str())) {
            out_proj_bias_ = Tensor::load_from_file(ss_out_bias.str());
        }
    }
}

void ShortConv::forward(const Tensor& x, Tensor& y) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    Tensor in_proj_out({batch * seq_len, 3 * hidden_size});
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out);

    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        int blocks = (batch * seq_len * 3 * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<<<blocks, BLOCK_SIZE>>>(in_proj_out.data(), in_proj_bias_.data(),
                                                  batch * seq_len, 3 * hidden_size);
        CHECK_CUDA(cudaGetLastError());
    }

    Tensor BCx({batch, 3 * hidden_size, seq_len});
    size_t total_bcx = batch * 3 * hidden_size * seq_len;
    int blocks_bcx = (total_bcx + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_conv_kernel<<<blocks_bcx, BLOCK_SIZE>>>(
        in_proj_out.data(), BCx.data(), batch, seq_len, 3 * hidden_size);
    CHECK_CUDA(cudaGetLastError());

    Tensor B({batch, hidden_size, seq_len});
    Tensor C({batch, hidden_size, seq_len});
    Tensor x_gate({batch, hidden_size, seq_len});

    size_t total_split = batch * hidden_size * seq_len;
    int blocks_split = (total_split + BLOCK_SIZE - 1) / BLOCK_SIZE;
    split_bcx_kernel<<<blocks_split, BLOCK_SIZE>>>(
        BCx.data(), B.data(), C.data(), x_gate.data(), batch, hidden_size, seq_len);
    CHECK_CUDA(cudaGetLastError());

    Tensor Bx({batch, hidden_size, seq_len});
    tensor_ops::mul(B, x_gate, Bx);

    Tensor conv_out({batch, hidden_size, seq_len});
    tensor_ops::causal_conv1d(Bx, conv_weight_, USE_CONV_BIAS ? &conv_bias_ : nullptr, conv_out);

    Tensor y_pre({batch, hidden_size, seq_len});
    tensor_ops::mul(C, conv_out, y_pre);

    Tensor y_pre_transposed({batch, seq_len, hidden_size});
    size_t total_trans = batch * seq_len * hidden_size;
    int blocks_trans = (total_trans + BLOCK_SIZE - 1) / BLOCK_SIZE;
    transpose_conv_reverse_kernel<<<blocks_trans, BLOCK_SIZE>>>(
        y_pre.data(), y_pre_transposed.data(), batch, hidden_size, seq_len);
    CHECK_CUDA(cudaGetLastError());

    Tensor y_pre_flat = y_pre_transposed.view({batch * seq_len, hidden_size});
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(y_pre_flat, out_proj_weight_, y_flat);

    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        int blocks = (batch * seq_len * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<<<blocks, BLOCK_SIZE>>>(y_flat.data(), out_proj_bias_.data(),
                                                  batch * seq_len, hidden_size);
        CHECK_CUDA(cudaGetLastError());
    }

    y_flat.reshape({batch, seq_len, hidden_size});

    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    CHECK_CUDA(cudaMemcpy(y.data(), y_flat.data(), y.size() * sizeof(float), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// DecoderLayer Implementation
// ============================================================================

DecoderLayer::DecoderLayer(int layer_idx, bool is_attention_layer)
    : layer_idx_(layer_idx), is_attention_layer_(is_attention_layer) {

    std::stringstream ss_norm1, ss_norm2;
    ss_norm1 << "layers." << layer_idx << ".operator_norm.weight";
    ss_norm2 << "layers." << layer_idx << ".ffn_norm.weight";

    input_layernorm_ = std::make_unique<RMSNorm>(ss_norm1.str());
    post_attention_layernorm_ = std::make_unique<RMSNorm>(ss_norm2.str());

    if (is_attention_layer) {
        self_attn_ = std::make_unique<Attention>(layer_idx);
    } else {
        short_conv_ = std::make_unique<ShortConv>(layer_idx);
    }

    if (static_cast<size_t>(layer_idx) >= NUM_DENSE_LAYERS) {
        moe_block_ = std::make_unique<SparseMoeBlock>(layer_idx);
    } else {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.w3.weight";
        dense_mlp_ = std::make_unique<MLP>(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }
}

void DecoderLayer::forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                          const Tensor* attention_mask, Tensor& output) {
    Tensor normed_input(x.shape());
    input_layernorm_->forward(x, normed_input);

    Tensor attn_output(x.shape());
    if (is_attention_layer_) {
        self_attn_->forward(normed_input, cos, sin, attention_mask, attn_output);
    } else {
        short_conv_->forward(normed_input, attn_output);
    }

    Tensor hidden_states(x.shape());
    tensor_ops::add(x, attn_output, hidden_states);

    Tensor normed_hidden(x.shape());
    post_attention_layernorm_->forward(hidden_states, normed_hidden);

    Tensor ffn_output;
    if (moe_block_) {
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits);
    } else {
        dense_mlp_->forward(normed_hidden, ffn_output);
    }

    tensor_ops::add(hidden_states, ffn_output, output);
}

// ============================================================================
// LFM2Model Implementation
// ============================================================================

LFM2Model::LFM2Model(const std::string& model_file) {
    if (g_parallel_ctx.world_rank == 0) {
        std::cout << "Loading LFM2-8B-A1B model from " << model_file << std::endl;
    }

    g_model_loader = std::make_unique<ModelLoader>(model_file);

    // Note: Using lazy GPU caching instead of preload_all_tensors() because:
    // - Expert parallelism: each GPU only needs its local experts (not all 32)
    // - Preloading all 2302 tensors would cause OOM
    // - Lazy loading with GPU cache still eliminates repeated H2D transfers
    // - Pinned memory in ModelLoader provides ~14 GB/s transfer speed

    load_embeddings();
    load_layers();
    load_output_layers();

    rotary_emb_ = std::make_unique<RotaryEmbedding>();

    if (g_parallel_ctx.world_rank == 0) {
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "GPU memory used for weights: "
                  << (g_model_loader->get_cached_memory_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
    }
}

void LFM2Model::load_embeddings() {
    if (g_parallel_ctx.world_rank == 0) {
        std::cout << "Loading embeddings..." << std::endl;
    }
    embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
    if (g_parallel_ctx.world_rank == 0) {
        std::cout << "  Embeddings shape: " << embed_tokens_.size(0) << " x " << embed_tokens_.size(1) << std::endl;
    }
}

void LFM2Model::load_layers() {
    if (g_parallel_ctx.world_rank == 0) {
        std::cout << "Loading " << NUM_HIDDEN_LAYERS << " decoder layers..." << std::endl;
    }

    layers_.reserve(NUM_HIDDEN_LAYERS);
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attention = (LAYER_TYPES[i] == 0);
        if (g_parallel_ctx.world_rank == 0) {
            std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
        }
        layers_.push_back(std::make_unique<DecoderLayer>(i, is_attention));
    }
}

void LFM2Model::load_output_layers() {
    if (g_parallel_ctx.world_rank == 0) {
        std::cout << "Loading output layers..." << std::endl;
    }

    norm_ = std::make_unique<RMSNorm>("embedding_norm.weight");

    if (g_model_loader->has_tensor("lm_head.weight")) {
        lm_head_ = Tensor::load_from_file("lm_head.weight");
    } else {
        lm_head_ = embed_tokens_;
        if (g_parallel_ctx.world_rank == 0) {
            std::cout << "  Using tied weights for LM head" << std::endl;
        }
    }
}

void LFM2Model::forward(const std::vector<int>& input_ids, Tensor& logits) {
    size_t batch = 1;
    size_t seq_len = input_ids.size();

    int* d_input_ids;
    CHECK_CUDA(cudaMalloc(&d_input_ids, seq_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input_ids, input_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));

    Tensor hidden_states({batch, seq_len, HIDDEN_SIZE});
    size_t total = seq_len * HIDDEN_SIZE;
    int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    embedding_lookup_kernel<<<blocks, BLOCK_SIZE>>>(
        embed_tokens_.data(), d_input_ids, hidden_states.data(), seq_len, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_input_ids));

    Tensor cos({seq_len, HEAD_DIM});
    Tensor sin({seq_len, HEAD_DIM});
    rotary_emb_->forward(seq_len, cos, sin);

    Tensor* attention_mask = nullptr;

    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        Tensor output({batch, seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = std::move(output);
    }

    Tensor normed_output({batch, seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output);

    Tensor last_hidden({batch, 1, HIDDEN_SIZE});
    int blocks_copy = (HIDDEN_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copy_last_token_kernel<<<blocks_copy, BLOCK_SIZE>>>(
        normed_output.data(), last_hidden.data(), seq_len, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    Tensor last_hidden_flat = last_hidden.view({batch, HIDDEN_SIZE});
    logits = Tensor({batch, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden_flat, lm_head_, logits);
}

void LFM2Model::forward_batch(const int* input_ids, size_t batch_size, size_t seq_len, Tensor& logits) {
    // Allocate and copy input_ids to GPU
    int* d_input_ids;
    size_t input_size = batch_size * seq_len;
    CHECK_CUDA(cudaMalloc(&d_input_ids, input_size * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_input_ids, input_ids, input_size * sizeof(int), cudaMemcpyHostToDevice));

    // Embedding lookup: [batch_size, seq_len, hidden_size]
    Tensor hidden_states({batch_size, seq_len, HIDDEN_SIZE});
    size_t total_embed = batch_size * seq_len * HIDDEN_SIZE;
    int blocks_embed = (total_embed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    embedding_lookup_batched_kernel<<<blocks_embed, BLOCK_SIZE>>>(
        embed_tokens_.data(), d_input_ids, hidden_states.data(),
        batch_size, seq_len, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaFree(d_input_ids));

    // Compute RoPE embeddings (same for all batches, depends only on seq_len)
    Tensor cos({seq_len, HEAD_DIM});
    Tensor sin({seq_len, HEAD_DIM});
    rotary_emb_->forward(seq_len, cos, sin);

    Tensor* attention_mask = nullptr;

    // Process through all layers
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        Tensor output({batch_size, seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = std::move(output);
    }

    // Final normalization
    Tensor normed_output({batch_size, seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output);

    // Extract last token from each batch: [batch_size, hidden_size]
    Tensor last_hidden({batch_size, HIDDEN_SIZE});
    size_t total_last = batch_size * HIDDEN_SIZE;
    int blocks_last = (total_last + BLOCK_SIZE - 1) / BLOCK_SIZE;
    copy_last_token_batched_kernel<<<blocks_last, BLOCK_SIZE>>>(
        normed_output.data(), last_hidden.data(),
        batch_size, seq_len, HIDDEN_SIZE);
    CHECK_CUDA(cudaGetLastError());

    // LM head projection: [batch_size, hidden_size] @ [vocab_size, hidden_size]^T -> [batch_size, vocab_size]
    logits = Tensor({batch_size, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden, lm_head_, logits);
}
