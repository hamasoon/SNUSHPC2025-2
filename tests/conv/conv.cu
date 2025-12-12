#include "conv.h"
#include "util.h"

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static float *x_gpu, *conv_weight_gpu, *in_proj_weight_gpu, *out_proj_weight_gpu, *output_gpu;
static float *in_proj_out_gpu, *B_gpu, *C_gpu, *x_gate_gpu;
static float *conv_out_gpu, *y_pre_transposed_gpu;

// Optimized matrix multiplication kernel with shared memory tiling
// Computes: C = A * B^T where A is (M, K), B is (N, K), C is (M, N)
#define TILE_SIZE 32
__global__ void matmul_kernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[col * K + t * TILE_SIZE + threadIdx.y];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Fused kernel: Transpose + Split into B, C, x_gate
__global__ void transpose_and_split_kernel(float *in, float *B, float *C, float *x_gate,
                                            int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;

    if (idx < total) {
        int s = idx % seq_len;
        int h = (idx / seq_len) % hidden_size;
        int b = idx / (hidden_size * seq_len);

        int base_idx = b * seq_len * 3 * hidden_size + s * 3 * hidden_size;
        B[idx] = in[base_idx + h];
        C[idx] = in[base_idx + h + hidden_size];
        x_gate[idx] = in[base_idx + h + 2 * hidden_size];
    }
}

// Fused kernel: Element-wise multiply B * x_gate and apply causal conv
__global__ void fused_mul_and_conv_kernel(float *B, float *x_gate, float *conv_weight,
                                           float *conv_out, int batch, int seq_len,
                                           int hidden_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;

    if (idx < total) {
        int s = idx % seq_len;
        int c = (idx / seq_len) % hidden_size;
        int b = idx / (hidden_size * seq_len);

        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernel_size; k++) {
            int input_pos = s - (kernel_size - 1) + k;
            if (input_pos >= 0) {
                int input_idx = b * hidden_size * seq_len + c * seq_len + input_pos;
                float bx_val = B[input_idx] * x_gate[input_idx];
                sum += bx_val * conv_weight[c * kernel_size + k];
            }
        }
        conv_out[idx] = sum;
    }
}

// Fused kernel: Element-wise multiply C * conv_out and transpose back
__global__ void fused_mul_and_transpose_kernel(float *C, float *conv_out, float *output,
                                                int batch, int seq_len, int hidden_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_size;

    if (idx < total) {
        int h = idx % hidden_size;
        int s = (idx / hidden_size) % seq_len;
        int b = idx / (seq_len * hidden_size);

        int input_idx = b * hidden_size * seq_len + h * seq_len + s;
        output[idx] = C[input_idx] * conv_out[input_idx];
    }
}

void conv_initialize(int batch, int seq_len, int hidden_size, int kernel_size,
                     float *conv_weight, float *in_proj_weight, float *out_proj_weight) {
    CHECK_CUDA(cudaMalloc(&x_gpu, batch * seq_len * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_weight_gpu, hidden_size * kernel_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&in_proj_weight_gpu, 3 * hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&out_proj_weight_gpu, hidden_size * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, batch * seq_len * hidden_size * sizeof(float)));

    // Allocate intermediate buffers
    CHECK_CUDA(cudaMalloc(&in_proj_out_gpu, batch * seq_len * 3 * hidden_size * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&x_gate_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&conv_out_gpu, batch * hidden_size * seq_len * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&y_pre_transposed_gpu, batch * seq_len * hidden_size * sizeof(float)));

    // Copy static weights to GPU
    CHECK_CUDA(cudaMemcpy(conv_weight_gpu, conv_weight, hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(in_proj_weight_gpu, in_proj_weight, 3 * hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(out_proj_weight_gpu, out_proj_weight, hidden_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice));
}

void conv(float *x, float *conv_weight, float *in_proj_weight, float *out_proj_weight,
          float *output, int batch, int seq_len, int hidden_size, int kernel_size) {

    // Copy input data to GPU
    CHECK_CUDA(cudaMemcpy(x_gpu, x, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice));

    // Step 1: in_proj matrix multiplication
    // x: (batch*seq_len, hidden_size) * weight^T: (hidden_size, 3*hidden_size)
    // output: (batch*seq_len, 3*hidden_size)
    dim3 block_matmul(TILE_SIZE, TILE_SIZE);
    dim3 grid_in_proj((3 * hidden_size + TILE_SIZE - 1) / TILE_SIZE,
                      (batch * seq_len + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid_in_proj, block_matmul>>>(x_gpu, in_proj_weight_gpu, in_proj_out_gpu,
                                                   batch * seq_len, 3 * hidden_size, hidden_size);

    // Step 2+3: Fused transpose and split
    int block_size = 256;
    int total_split = batch * hidden_size * seq_len;
    int grid_split = (total_split + block_size - 1) / block_size;
    transpose_and_split_kernel<<<grid_split, block_size>>>(in_proj_out_gpu, B_gpu, C_gpu, x_gate_gpu,
                                                             batch, seq_len, hidden_size);

    // Step 4+5: Fused element-wise multiply and causal conv1d
    int total_conv = batch * hidden_size * seq_len;
    int grid_conv = (total_conv + block_size - 1) / block_size;
    fused_mul_and_conv_kernel<<<grid_conv, block_size>>>(B_gpu, x_gate_gpu, conv_weight_gpu,
                                                           conv_out_gpu, batch, seq_len,
                                                           hidden_size, kernel_size);

    // Step 6+7: Fused element-wise multiply and transpose back
    int total_transpose_back = batch * seq_len * hidden_size;
    int grid_transpose_back = (total_transpose_back + block_size - 1) / block_size;
    fused_mul_and_transpose_kernel<<<grid_transpose_back, block_size>>>(C_gpu, conv_out_gpu,
                                                                          y_pre_transposed_gpu,
                                                                          batch, seq_len, hidden_size);

    // Step 8: out_proj matrix multiplication
    // y_pre_transposed: (batch*seq_len, hidden_size) * weight^T: (hidden_size, hidden_size)
    // output: (batch*seq_len, hidden_size)
    dim3 grid_out_proj((hidden_size + TILE_SIZE - 1) / TILE_SIZE,
                       (batch * seq_len + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid_out_proj, block_matmul>>>(y_pre_transposed_gpu, out_proj_weight_gpu, output_gpu,
                                                     batch * seq_len, hidden_size, hidden_size);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, output_gpu, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost));
}

void conv_finalize() {
    CHECK_CUDA(cudaFree(x_gpu));
    CHECK_CUDA(cudaFree(conv_weight_gpu));
    CHECK_CUDA(cudaFree(in_proj_weight_gpu));
    CHECK_CUDA(cudaFree(out_proj_weight_gpu));
    CHECK_CUDA(cudaFree(output_gpu));

    // Free intermediate buffers
    CHECK_CUDA(cudaFree(in_proj_out_gpu));
    CHECK_CUDA(cudaFree(B_gpu));
    CHECK_CUDA(cudaFree(C_gpu));
    CHECK_CUDA(cudaFree(x_gate_gpu));
    CHECK_CUDA(cudaFree(conv_out_gpu));
    CHECK_CUDA(cudaFree(y_pre_transposed_gpu));
}
