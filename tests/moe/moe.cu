#include "moe.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *gate_gpu, *expert_bias_gpu, *output_gpu;
static float **expert_w1_gpu, **expert_w2_gpu, **expert_w3_gpu;
static float **expert_w1_gpu_ptrs, **expert_w2_gpu_ptrs, **expert_w3_gpu_ptrs;
static int g_num_experts = 0;

// Intermediate buffers
static float *router_logits_gpu;
static float *routing_weights_gpu;
static int *top_k_indices_gpu;
static float *top_k_weights_gpu;
static float *w1_out_gpu, *w3_out_gpu;

// MoE configuration flags (match src/model.cu behavior)
static const float ROUTED_SCALING_FACTOR = 1.0f;
static const bool NORM_TOPK_PROB = true;
static const bool USE_EXPERT_BIAS = true;

// CUDA Kernels

// Kernel 1: Compute router logits and apply sigmoid
// router_logits[t, e] = sigmoid(x[t] @ gate[e]^T)
__global__ void compute_router_logits_kernel(
    const float* __restrict__ x,           // [num_tokens, hidden_size]
    const float* __restrict__ gate,        // [num_experts, hidden_size]
    float* __restrict__ router_logits,     // [num_tokens, num_experts]
    float* __restrict__ routing_weights,   // [num_tokens, num_experts]
    int num_tokens, int num_experts, int hidden_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = tid / num_experts;
    int expert_idx = tid % num_experts;

    if (token_idx >= num_tokens || expert_idx >= num_experts) return;

    // Compute dot product: x[token_idx] @ gate[expert_idx]
    float sum = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        sum += x[token_idx * hidden_size + h] * gate[expert_idx * hidden_size + h];
    }

    router_logits[token_idx * num_experts + expert_idx] = sum;

    // Apply sigmoid: 1 / (1 + exp(-x))
    float sigmoid = 1.0f / (1.0f + expf(-sum));
    routing_weights[token_idx * num_experts + expert_idx] = sigmoid;
}

// Kernel 2: Select top-k experts per token
__global__ void select_topk_kernel(
    const float* __restrict__ routing_weights,  // [num_tokens, num_experts]
    const float* __restrict__ expert_bias,      // [num_experts]
    int* __restrict__ top_k_indices,            // [num_tokens, num_experts_per_tok]
    float* __restrict__ top_k_weights,          // [num_tokens, num_experts_per_tok]
    int num_tokens, int num_experts, int num_experts_per_tok, bool use_bias) {

    int token_idx = blockIdx.x;
    if (token_idx >= num_tokens) return;

    // Use shared memory for scores
    extern __shared__ float shared_mem[];
    float* scores = shared_mem;
    int* indices = (int*)(shared_mem + num_experts);

    // Load routing weights and compute scores (with bias if enabled)
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        float weight = routing_weights[token_idx * num_experts + e];
        scores[e] = use_bias ? (weight + expert_bias[e]) : weight;
        indices[e] = e;
    }
    __syncthreads();

    // Selection sort for top-k (simple but effective for small k)
    for (int k = 0; k < num_experts_per_tok; k++) {
        if (threadIdx.x == 0) {
            // Find maximum
            int max_idx = k;
            float max_score = scores[k];
            for (int i = k + 1; i < num_experts; i++) {
                if (scores[i] > max_score) {
                    max_score = scores[i];
                    max_idx = i;
                }
            }
            // Swap
            if (max_idx != k) {
                float tmp_score = scores[k];
                scores[k] = scores[max_idx];
                scores[max_idx] = tmp_score;

                int tmp_idx = indices[k];
                indices[k] = indices[max_idx];
                indices[max_idx] = tmp_idx;
            }
        }
        __syncthreads();
    }

    // Normalize top-k weights (using original sigmoid weights, not scores)
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int k = 0; k < num_experts_per_tok; k++) {
            int expert_idx = indices[k];
            float weight = routing_weights[token_idx * num_experts + expert_idx];
            sum += weight;
        }

        // Normalize and apply scaling factor
        float norm_factor = (sum > 1e-6f) ? (ROUTED_SCALING_FACTOR / sum) : ROUTED_SCALING_FACTOR;

        for (int k = 0; k < num_experts_per_tok; k++) {
            int expert_idx = indices[k];
            top_k_indices[token_idx * num_experts_per_tok + k] = expert_idx;
            float weight = routing_weights[token_idx * num_experts + expert_idx];
            top_k_weights[token_idx * num_experts_per_tok + k] = NORM_TOPK_PROB ? (weight * norm_factor) : (weight * ROUTED_SCALING_FACTOR);
        }
    }
}

// Optimized kernel to compute w1 @ x and w3 @ x
__global__ void compute_w1w3_kernel(
    const float* __restrict__ x,                // [num_tokens, hidden_size]
    float** __restrict__ expert_w1,             // [num_experts][expert_hidden_size, hidden_size]
    float** __restrict__ expert_w3,             // [num_experts][expert_hidden_size, hidden_size]
    const int* __restrict__ top_k_indices,      // [num_tokens, num_experts_per_tok]
    float* __restrict__ w1_out,                 // [num_tokens * num_experts_per_tok, expert_hidden_size]
    float* __restrict__ w3_out,                 // [num_tokens * num_experts_per_tok, expert_hidden_size]
    int num_tokens, int num_experts_per_tok, int hidden_size, int expert_hidden_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * num_experts_per_tok * expert_hidden_size;

    if (tid >= total_elements) return;

    int h_inter = tid % expert_hidden_size;
    int pair_idx = tid / expert_hidden_size;

    int token_idx = pair_idx / num_experts_per_tok;
    int expert_idx = top_k_indices[pair_idx];

    const float* x_token = x + token_idx * hidden_size;
    const float* w1 = expert_w1[expert_idx];
    const float* w3 = expert_w3[expert_idx];

    // Compute w1[h_inter] @ x
    float sum1 = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        sum1 += w1[h_inter * hidden_size + h] * x_token[h];
    }
    w1_out[pair_idx * expert_hidden_size + h_inter] = sum1;

    // Compute w3[h_inter] @ x
    float sum3 = 0.0f;
    for (int h = 0; h < hidden_size; h++) {
        sum3 += w3[h_inter * hidden_size + h] * x_token[h];
    }
    w3_out[pair_idx * expert_hidden_size + h_inter] = sum3;
}

// Optimized kernel to compute final output: w2 @ (silu(w1_out) * w3_out)
__global__ void compute_w2_kernel(
    float** __restrict__ expert_w2,             // [num_experts][hidden_size, expert_hidden_size]
    const int* __restrict__ top_k_indices,      // [num_tokens, num_experts_per_tok]
    const float* __restrict__ top_k_weights,    // [num_tokens, num_experts_per_tok]
    const float* __restrict__ w1_out,           // [num_tokens * num_experts_per_tok, expert_hidden_size]
    const float* __restrict__ w3_out,           // [num_tokens * num_experts_per_tok, expert_hidden_size]
    float* __restrict__ output,                 // [num_tokens, hidden_size]
    int num_tokens, int num_experts_per_tok, int hidden_size, int expert_hidden_size) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = num_tokens * num_experts_per_tok * hidden_size;

    if (tid >= total_elements) return;

    int h_out = tid % hidden_size;
    int pair_idx = tid / hidden_size;

    int token_idx = pair_idx / num_experts_per_tok;
    int expert_idx = top_k_indices[pair_idx];
    float weight = top_k_weights[pair_idx];

    const float* w2 = expert_w2[expert_idx];
    const float* w1_vec = w1_out + pair_idx * expert_hidden_size;
    const float* w3_vec = w3_out + pair_idx * expert_hidden_size;

    // Compute w2[h_out] @ (silu(w1_out) * w3_out)
    float sum = 0.0f;
    for (int h_inter = 0; h_inter < expert_hidden_size; h_inter++) {
        float w1_val = w1_vec[h_inter];
        float silu_w1 = w1_val / (1.0f + expf(-w1_val));
        float gate_val = silu_w1 * w3_vec[h_inter];
        sum += w2[h_out * expert_hidden_size + h_inter] * gate_val;
    }

    atomicAdd(&output[token_idx * hidden_size + h_out], weight * sum);
}

void moe_initialize(int batch, int seq_len, int hidden_size, int num_experts,
                   int num_experts_per_tok, int expert_hidden_size,
                   float *gate, float **expert_w1, float **expert_w2, float **expert_w3, float *expert_bias) {
    g_num_experts = num_experts;

    int num_tokens = batch * seq_len;

    CHECK_CUDA(cudaMalloc(&x_gpu, num_tokens * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gate_gpu, num_experts * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&expert_bias_gpu, num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, num_tokens * hidden_size * sizeof(float)));

    // Allocate intermediate buffers
    CHECK_CUDA(cudaMalloc(&router_logits_gpu, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&routing_weights_gpu, num_tokens * num_experts * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&top_k_indices_gpu, num_tokens * num_experts_per_tok * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&top_k_weights_gpu, num_tokens * num_experts_per_tok * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w1_out_gpu, num_tokens * num_experts_per_tok * expert_hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&w3_out_gpu, num_tokens * num_experts_per_tok * expert_hidden_size * sizeof(float)));

    // Allocate expert weights
    expert_w1_gpu = (float**)malloc(num_experts * sizeof(float*));
    expert_w2_gpu = (float**)malloc(num_experts * sizeof(float*));
    expert_w3_gpu = (float**)malloc(num_experts * sizeof(float*));

    for (int i = 0; i < num_experts; i++) {
        CHECK_CUDA(cudaMalloc(&expert_w1_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&expert_w2_gpu[i], hidden_size * expert_hidden_size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&expert_w3_gpu[i], expert_hidden_size * hidden_size * sizeof(float)));
    }

    // Allocate device array of pointers
    CHECK_CUDA(cudaMalloc(&expert_w1_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&expert_w2_gpu_ptrs, num_experts * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&expert_w3_gpu_ptrs, num_experts * sizeof(float*)));

    CHECK_CUDA(cudaMemcpy(expert_w1_gpu_ptrs, expert_w1_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_w2_gpu_ptrs, expert_w2_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_w3_gpu_ptrs, expert_w3_gpu, num_experts * sizeof(float*), cudaMemcpyHostToDevice));

    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(gate_gpu, gate, num_experts * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(expert_bias_gpu, expert_bias, num_experts * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_experts; i++) {
        CHECK_CUDA(cudaMemcpy(expert_w1_gpu[i], expert_w1[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(expert_w2_gpu[i], expert_w2[i], hidden_size * expert_hidden_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(expert_w3_gpu[i], expert_w3[i], expert_hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    }
}

void moe(float *x, float *gate, float **expert_w1, float **expert_w2, float **expert_w3,
         float *expert_bias, float *output, int batch, int seq_len, int hidden_size,
         int num_experts, int num_experts_per_tok, int expert_hidden_size) {

    int num_tokens = batch * seq_len;

    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, num_tokens * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(output_gpu, 0, num_tokens * hidden_size * sizeof(float)));

    // Step 1: Compute router logits and apply sigmoid
    int total_router_elements = num_tokens * num_experts;
    int block_size = 256;
    int num_blocks = (total_router_elements + block_size - 1) / block_size;

    compute_router_logits_kernel<<<num_blocks, block_size>>>(
        x_gpu, gate_gpu, router_logits_gpu, routing_weights_gpu,
        num_tokens, num_experts, hidden_size
    );
    CHECK_CUDA(cudaGetLastError());

    // Step 2: Select top-k experts per token
    int shared_mem_size = num_experts * (sizeof(float) + sizeof(int));
    select_topk_kernel<<<num_tokens, 256, shared_mem_size>>>(
        routing_weights_gpu, expert_bias_gpu, top_k_indices_gpu, top_k_weights_gpu,
        num_tokens, num_experts, num_experts_per_tok, USE_EXPERT_BIAS
    );
    CHECK_CUDA(cudaGetLastError());

    // Step 3: Compute w1 @ x and w3 @ x
    int total_expert_pairs = num_tokens * num_experts_per_tok;
    int w1w3_threads = (expert_hidden_size > 1024) ? 1024 : expert_hidden_size;
    int w1w3_blocks = (total_expert_pairs * expert_hidden_size + w1w3_threads - 1) / w1w3_threads;
    compute_w1w3_kernel<<<w1w3_blocks, w1w3_threads>>>(
        x_gpu, expert_w1_gpu_ptrs, expert_w3_gpu_ptrs, top_k_indices_gpu,
        w1_out_gpu, w3_out_gpu,
        num_tokens, num_experts_per_tok, hidden_size, expert_hidden_size
    );
    CHECK_CUDA(cudaGetLastError());

    // Step 4: Compute w2 @ (silu(w1_out) * w3_out)
    int w2_threads = (hidden_size > 1024) ? 1024 : hidden_size;
    int w2_blocks = (total_expert_pairs * hidden_size + w2_threads - 1) / w2_threads;
    compute_w2_kernel<<<w2_blocks, w2_threads>>>(
        expert_w2_gpu_ptrs, top_k_indices_gpu, top_k_weights_gpu,
        w1_out_gpu, w3_out_gpu, output_gpu,
        num_tokens, num_experts_per_tok, hidden_size, expert_hidden_size
    );
    CHECK_CUDA(cudaGetLastError());

    // Copy output back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, num_tokens * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void moe_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(gate_gpu));
    CHECK_CUDA(cudaFree(expert_bias_gpu));
    CHECK_CUDA(cudaFree(output_gpu));

    // Free intermediate buffers
    CHECK_CUDA(cudaFree(router_logits_gpu));
    CHECK_CUDA(cudaFree(routing_weights_gpu));
    CHECK_CUDA(cudaFree(top_k_indices_gpu));
    CHECK_CUDA(cudaFree(top_k_weights_gpu));
    CHECK_CUDA(cudaFree(w1_out_gpu));
    CHECK_CUDA(cudaFree(w3_out_gpu));

    for (int i = 0; i < g_num_experts; i++) {
        CHECK_CUDA(cudaFree(expert_w1_gpu[i]));
        CHECK_CUDA(cudaFree(expert_w2_gpu[i]));
        CHECK_CUDA(cudaFree(expert_w3_gpu[i]));
    }

    CHECK_CUDA(cudaFree(expert_w1_gpu_ptrs));
    CHECK_CUDA(cudaFree(expert_w2_gpu_ptrs));
    CHECK_CUDA(cudaFree(expert_w3_gpu_ptrs));

    free(expert_w1_gpu);
    free(expert_w2_gpu);
    free(expert_w3_gpu);
}
