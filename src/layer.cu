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

// Optimized matmul configuration
// BM, BN: Block tile size for M and N dimensions
// BK: Block tile size for K dimension
// TM, TN: Register tile size per thread (each thread computes TM x TN elements)
#define BM 128
#define BN 64
#define BK 8
#define TM 8
#define TN 4

// Warp configuration
#define WARP_SIZE 32

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

// Fused SiLU + Mul kernel: y = silu(gate) * up
__global__ void silu_mul_kernel(const float* gate, const float* up, float* y, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float inv_sigmoid = 1.0f / (1.0f + expf(-g));
        float silu_g = g * inv_sigmoid;
        y[idx] = silu_g * up[idx];
    }
}

// ============================================================================
// Warp-level reduction utilities
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// ============================================================================
// Softmax kernel (optimized with warp-level reduction)
// ============================================================================

__global__ void softmax_kernel(const float* x, float* y, size_t outer_size, size_t inner_size) {
    extern __shared__ float shared[];

    size_t row = blockIdx.x;
    if (row >= outer_size) return;

    const float* x_row = x + row * inner_size;
    float* y_row = y + row * inner_size;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Find max for numerical stability
    float max_val = -INFINITY;
    for (size_t i = tid; i < inner_size; i += blockDim.x) {
        max_val = fmaxf(max_val, x_row[i]);
    }

    // Warp-level reduction for max
    max_val = warp_reduce_max(max_val);

    // Store warp results in shared memory
    if (lane_id == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        max_val = (lane_id < num_warps) ? shared[lane_id] : -INFINITY;
        max_val = warp_reduce_max(max_val);
        if (lane_id == 0) {
            shared[0] = max_val;
        }
    }
    __syncthreads();
    max_val = shared[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = tid; i < inner_size; i += blockDim.x) {
        float exp_val = expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum += exp_val;
    }

    // Warp-level reduction for sum
    sum = warp_reduce_sum(sum);

    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane_id == 0) {
            shared[0] = sum;
        }
    }
    __syncthreads();
    sum = shared[0];

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t i = tid; i < inner_size; i += blockDim.x) {
        y_row[i] *= inv_sum;
    }
}

// ============================================================================
// RMSNorm kernel (optimized with warp-level reduction)
// ============================================================================

__global__ void rms_norm_kernel(const float* x, const float* weight, float eps, float* y,
                                 size_t outer_size, size_t hidden_size, float inv_hidden_size) {
    extern __shared__ float shared[];

    size_t row = blockIdx.x;
    if (row >= outer_size) return;

    const float* x_row = x + row * hidden_size;
    float* y_row = y + row * hidden_size;

    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    // Warp-level reduction
    sum_sq = warp_reduce_sum(sum_sq);

    if (lane_id == 0) {
        shared[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces across all warps
    if (warp_id == 0) {
        sum_sq = (lane_id < num_warps) ? shared[lane_id] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
        if (lane_id == 0) {
            shared[0] = sum_sq;
        }
    }
    __syncthreads();

    float rms = 1.0f / sqrtf(shared[0] * inv_hidden_size + eps);

    // Normalize and scale
    for (size_t i = tid; i < hidden_size; i += blockDim.x) {
        y_row[i] = x_row[i] * rms * weight[i];
    }
}

// ============================================================================
// Matrix multiplication kernels
// ============================================================================

// Optimized matmul kernel with:
// 1. Register Tiling: Each thread computes TM x TN output elements
// 2. Double Buffering: Overlap shared memory loads with computation
// 3. Warp-level parallelism: Coalesced memory access patterns
// a: (m, k), b: (k, n), c: (m, n)
__global__ void matmul_kernel(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c, size_t m, size_t k, size_t n) {
    // Thread block computes BM x BN tile of C
    // Each thread computes TM x TN elements
    // Block has (BM/TM) x (BN/TN) = 16 x 16 = 256 threads

    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15

    // Position of this thread's TM x TN tile in the output block
    const int thread_row = ty * TM;  // Starting row within block tile
    const int thread_col = tx * TN;  // Starting col within block tile

    // Global position of the block tile
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Double buffered shared memory
    __shared__ float As[2][BK][BM];  // Transposed for coalesced access
    __shared__ float Bs[2][BK][BN];

    // Register tile for accumulation (TM x TN per thread)
    float reg_c[TM][TN] = {0.0f};

    // Register tiles for A and B fragments
    float reg_a[TM];
    float reg_b[TN];

    // Number of threads per block
    const int num_threads = (BM / TM) * (BN / TN);  // 256
    const int tid = ty * (BN / TN) + tx;  // Linear thread ID

    // Calculate how many elements each thread loads for A and B
    // A tile: BM x BK = 128 x 8 = 1024 elements, 256 threads -> 4 elements per thread
    // B tile: BK x BN = 8 x 128 = 1024 elements, 256 threads -> 4 elements per thread
    const int a_tile_elements = BM * BK;
    const int b_tile_elements = BK * BN;
    const int a_loads_per_thread = (a_tile_elements + num_threads - 1) / num_threads;
    const int b_loads_per_thread = (b_tile_elements + num_threads - 1) / num_threads;

    // Number of K tiles
    const int num_k_tiles = (k + BK - 1) / BK;

    // Current buffer index for double buffering
    int buf_idx = 0;

    // Preload first tile into shared memory (buffer 0)
    #pragma unroll
    for (int i = 0; i < a_loads_per_thread; i++) {
        int elem_idx = tid + i * num_threads;
        if (elem_idx < a_tile_elements) {
            int a_row = elem_idx / BK;
            int a_col = elem_idx % BK;
            int global_row = block_row + a_row;
            int global_col = a_col;
            if (global_row < m && global_col < k) {
                As[0][a_col][a_row] = a[global_row * k + global_col];
            } else {
                As[0][a_col][a_row] = 0.0f;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < b_loads_per_thread; i++) {
        int elem_idx = tid + i * num_threads;
        if (elem_idx < b_tile_elements) {
            int b_row = elem_idx / BN;
            int b_col = elem_idx % BN;
            int global_row = b_row;
            int global_col = block_col + b_col;
            if (global_row < k && global_col < n) {
                Bs[0][b_row][b_col] = b[global_row * n + global_col];
            } else {
                Bs[0][b_row][b_col] = 0.0f;
            }
        }
    }

    __syncthreads();

    // Main loop with double buffering
    for (int tile = 0; tile < num_k_tiles; tile++) {
        int next_buf = 1 - buf_idx;
        int next_tile = tile + 1;

        // Prefetch next tile into alternate buffer (if not last tile)
        if (next_tile < num_k_tiles) {
            #pragma unroll
            for (int i = 0; i < a_loads_per_thread; i++) {
                int elem_idx = tid + i * num_threads;
                if (elem_idx < a_tile_elements) {
                    int a_row = elem_idx / BK;
                    int a_col = elem_idx % BK;
                    int global_row = block_row + a_row;
                    int global_col = next_tile * BK + a_col;
                    if (global_row < m && global_col < k) {
                        As[next_buf][a_col][a_row] = a[global_row * k + global_col];
                    } else {
                        As[next_buf][a_col][a_row] = 0.0f;
                    }
                }
            }

            #pragma unroll
            for (int i = 0; i < b_loads_per_thread; i++) {
                int elem_idx = tid + i * num_threads;
                if (elem_idx < b_tile_elements) {
                    int b_row = elem_idx / BN;
                    int b_col = elem_idx % BN;
                    int global_row = next_tile * BK + b_row;
                    int global_col = block_col + b_col;
                    if (global_row < k && global_col < n) {
                        Bs[next_buf][b_row][b_col] = b[global_row * n + global_col];
                    } else {
                        Bs[next_buf][b_row][b_col] = 0.0f;
                    }
                }
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Load A fragment into registers
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[buf_idx][kk][thread_row + i];
            }

            // Load B fragment into registers
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_b[j] = Bs[buf_idx][kk][thread_col + j];
            }

            // Compute outer product
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        buf_idx = next_buf;
        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int global_row = block_row + thread_row + i;
        if (global_row < m) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                int global_col = block_col + thread_col + j;
                if (global_col < n) {
                    c[global_row * n + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// Fallback kernel for small matrices
__global__ void matmul_kernel_simple(const float* a, const float* b, float* c,
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

// Optimized transposed matmul: c = a @ b^T
// a: (m, k), b: (n, k), c: (m, n)
// With Register Tiling, Double Buffering, and Warp-level parallelism
// TITAN RTX (Turing) optimizations:
// - Shared memory padding (+4) to avoid bank conflicts (32 banks, 4 bytes each)
// - Vectorized float4 loads for 128-bit memory transactions
// - Coalesced global memory access pattern for 128-byte cache lines
// - __launch_bounds__ to optimize register allocation for better occupancy
__global__ __launch_bounds__(256, 4)
void matmul_transposed_kernel(const float* __restrict__ a, const float* __restrict__ b,
                              float* __restrict__ c, size_t m, size_t k, size_t n) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int thread_row = ty * TM;
    const int thread_col = tx * TN;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // Double buffered shared memory with padding to avoid bank conflicts
    // TITAN RTX has 32 banks, 4 bytes per bank
    // Adding +4 padding ensures different rows map to different banks
    __shared__ float As[2][BK][BM + 4];  // +4 padding for bank conflict avoidance
    __shared__ float Bs[2][BK][BN + 4];  // +4 padding for bank conflict avoidance

    float reg_c[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    const int num_threads = (BM / TM) * (BN / TN);  // 256 threads
    const int tid = ty * (BN / TN) + tx;

    // Vectorized loading constants
    // A tile: BM x BK = 128 x 8 = 1024 elements
    // B tile: BN x BK = 64 x 8 = 512 elements
    // With float4 (4 elements): A needs 256 float4 loads, B needs 128 float4 loads
    const int a_tile_float4 = (BM * BK) / 4;  // 256
    const int b_tile_float4 = (BN * BK) / 4;  // 128

    const int num_k_tiles = (k + BK - 1) / BK;

    int buf_idx = 0;

    // Preload first tile with float4 vectorized loads
    // A: Each thread loads consecutive elements along k dimension for better coalescing
    for (int i = tid; i < a_tile_float4; i += num_threads) {
        // Each float4 covers 4 consecutive k elements for one row
        int row_idx = i / (BK / 4);     // Which row (0 to BM-1)
        int k_group = i % (BK / 4);     // Which group of 4 in k dimension (0 or 1 for BK=8)
        int global_row = block_row + row_idx;
        int global_k_base = k_group * 4;

        if (global_row < m && global_k_base + 3 < k) {
            // Aligned float4 load
            float4 tmp = *reinterpret_cast<const float4*>(&a[global_row * k + global_k_base]);
            As[0][global_k_base + 0][row_idx] = tmp.x;
            As[0][global_k_base + 1][row_idx] = tmp.y;
            As[0][global_k_base + 2][row_idx] = tmp.z;
            As[0][global_k_base + 3][row_idx] = tmp.w;
        } else {
            // Boundary handling - scalar loads
            for (int kk = 0; kk < 4; kk++) {
                int gk = global_k_base + kk;
                As[0][gk][row_idx] = (global_row < m && gk < k) ? a[global_row * k + gk] : 0.0f;
            }
        }
    }

    // B (transposed): b is (n, k), each row of B is k elements
    for (int i = tid; i < b_tile_float4; i += num_threads) {
        int col_idx = i / (BK / 4);     // Which column (0 to BN-1)
        int k_group = i % (BK / 4);     // Which group of 4 in k dimension
        int global_col = block_col + col_idx;
        int global_k_base = k_group * 4;

        if (global_col < n && global_k_base + 3 < k) {
            // Aligned float4 load
            float4 tmp = *reinterpret_cast<const float4*>(&b[global_col * k + global_k_base]);
            Bs[0][global_k_base + 0][col_idx] = tmp.x;
            Bs[0][global_k_base + 1][col_idx] = tmp.y;
            Bs[0][global_k_base + 2][col_idx] = tmp.z;
            Bs[0][global_k_base + 3][col_idx] = tmp.w;
        } else {
            // Boundary handling - scalar loads
            for (int kk = 0; kk < 4; kk++) {
                int gk = global_k_base + kk;
                Bs[0][gk][col_idx] = (global_col < n && gk < k) ? b[global_col * k + gk] : 0.0f;
            }
        }
    }

    __syncthreads();

    // Main loop with double buffering
    for (int tile = 0; tile < num_k_tiles; tile++) {
        int next_buf = 1 - buf_idx;
        int next_tile = tile + 1;

        if (next_tile < num_k_tiles) {
            // Prefetch next A tile with float4
            for (int i = tid; i < a_tile_float4; i += num_threads) {
                int row_idx = i / (BK / 4);
                int k_group = i % (BK / 4);
                int global_row = block_row + row_idx;
                int global_k_base = next_tile * BK + k_group * 4;

                if (global_row < m && global_k_base + 3 < k) {
                    float4 tmp = *reinterpret_cast<const float4*>(&a[global_row * k + global_k_base]);
                    As[next_buf][k_group * 4 + 0][row_idx] = tmp.x;
                    As[next_buf][k_group * 4 + 1][row_idx] = tmp.y;
                    As[next_buf][k_group * 4 + 2][row_idx] = tmp.z;
                    As[next_buf][k_group * 4 + 3][row_idx] = tmp.w;
                } else {
                    for (int kk = 0; kk < 4; kk++) {
                        int gk = global_k_base + kk;
                        int local_k = k_group * 4 + kk;
                        As[next_buf][local_k][row_idx] = (global_row < m && gk < k) ? a[global_row * k + gk] : 0.0f;
                    }
                }
            }

            // Prefetch next B tile with float4
            for (int i = tid; i < b_tile_float4; i += num_threads) {
                int col_idx = i / (BK / 4);
                int k_group = i % (BK / 4);
                int global_col = block_col + col_idx;
                int global_k_base = next_tile * BK + k_group * 4;

                if (global_col < n && global_k_base + 3 < k) {
                    float4 tmp = *reinterpret_cast<const float4*>(&b[global_col * k + global_k_base]);
                    Bs[next_buf][k_group * 4 + 0][col_idx] = tmp.x;
                    Bs[next_buf][k_group * 4 + 1][col_idx] = tmp.y;
                    Bs[next_buf][k_group * 4 + 2][col_idx] = tmp.z;
                    Bs[next_buf][k_group * 4 + 3][col_idx] = tmp.w;
                } else {
                    for (int kk = 0; kk < 4; kk++) {
                        int gk = global_k_base + kk;
                        int local_k = k_group * 4 + kk;
                        Bs[next_buf][local_k][col_idx] = (global_col < n && gk < k) ? b[global_col * k + gk] : 0.0f;
                    }
                }
            }
        }

        // Compute on current buffer
        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[buf_idx][kk][thread_row + i];
            }

            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_b[j] = Bs[buf_idx][kk][thread_col + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        buf_idx = next_buf;
        __syncthreads();
    }

    // Write results to global memory - vectorized float4 store when possible
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int global_row = block_row + thread_row + i;
        if (global_row < m) {
            int global_col_base = block_col + thread_col;
            // TN=4, so we can use float4 store if aligned and within bounds
            if (global_col_base + 3 < n) {
                float4 out_val = make_float4(reg_c[i][0], reg_c[i][1], reg_c[i][2], reg_c[i][3]);
                *reinterpret_cast<float4*>(&c[global_row * n + global_col_base]) = out_val;
            } else {
                // Boundary handling - scalar stores
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    int global_col = global_col_base + j;
                    if (global_col < n) {
                        c[global_row * n + global_col] = reg_c[i][j];
                    }
                }
            }
        }
    }
}

// Fallback transposed kernel for very small matrices (k < BK=8)
// __launch_bounds__ optimizes register allocation for 256 threads
__global__ __launch_bounds__(256, 4)
void matmul_transposed_kernel_simple(const float* a, const float* b, float* c,
                                     size_t m, size_t k, size_t n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (size_t t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        size_t a_col = t * TILE_SIZE + threadIdx.x;
        size_t b_col = t * TILE_SIZE + threadIdx.y;

        if (row < m && a_col < k) {
            As[threadIdx.y][threadIdx.x] = a[row * k + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

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

    // Use optimized kernel for large matrices, simple kernel for small ones
    if (m >= BM && n >= BN && k >= BK) {
        // Optimized kernel: 16x16 threads, each computing 8x8 elements
        dim3 block(BN / TN, BM / TM);  // 16x16 = 256 threads
        dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
        matmul_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, k, n);
    } else {
        // Fallback to simple kernel for small matrices
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel_simple<<<grid, block>>>(a.data(), b.data(), c.data(), m, k, n);
    }
    CHECK_CUDA(cudaGetLastError());
}

void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c) {
    size_t m = a.size(0);
    size_t k = a.size(1);
    size_t n = b.size(0);

    // Always use optimized kernel - it has proper boundary checks
    // The kernel handles any m,k,n with boundary conditions
    // Only fall back to simple kernel for very small k (< BK=8)
    if (k >= BK) {
        // Full optimized kernel: 16x16 threads, each computing 8x4 elements
        dim3 block(BN / TN, BM / TM);  // 16x16 = 256 threads
        dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
        matmul_transposed_kernel<<<grid, block>>>(a.data(), b.data(), c.data(), m, k, n);
    } else {
        // Fallback to simple kernel only for very small k dimension
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
        matmul_transposed_kernel_simple<<<grid, block>>>(a.data(), b.data(), c.data(), m, k, n);
    }
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

// Fused silu + mul: y = silu(gate) * up
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& y) {
    size_t n = gate.size();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_mul_kernel<<<blocks, BLOCK_SIZE>>>(gate.data(), up.data(), y.data(), n);
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
