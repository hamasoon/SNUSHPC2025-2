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
#include <omp.h>

// Global model loader (definition)
std::unique_ptr<ModelLoader> g_model_loader;

// MPI context
static int g_mpi_rank = 0;
static int g_mpi_size = 1;

// ============================================================================
// GPU Memory Management and Multi-GPU Support
// ============================================================================

#define NUM_GPUS 4
#define EXPERTS_PER_GPU (NUM_EXPERTS / NUM_GPUS)
#define TILE_SIZE 32
#define BLOCK_SIZE 256

// GPU weight storage
struct GPUWeights {
    float* data;
    size_t size;
    int gpu_id;
};

// Per-GPU context
struct GPUContext {
    cudaStream_t stream;
    float* workspace;
    size_t workspace_size;
};

static GPUContext g_gpu_ctx[NUM_GPUS];
static bool g_multi_gpu_initialized = false;

// Initialize multi-GPU setup with MPI awareness
void init_multi_gpu() {
    if (g_multi_gpu_initialized) return;

    // Get MPI rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);

    int device_count;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < NUM_GPUS && i < device_count; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&g_gpu_ctx[i].stream);

        // Allocate workspace (enough for largest intermediate tensor)
        g_gpu_ctx[i].workspace_size = 256 * 1024 * 1024;  // 256MB per GPU
        cudaMalloc(&g_gpu_ctx[i].workspace, g_gpu_ctx[i].workspace_size);
    }

    // Enable peer access for direct GPU-to-GPU communication
    for (int i = 0; i < NUM_GPUS && i < device_count; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < NUM_GPUS && j < device_count; j++) {
            if (i != j) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                }
            }
        }
    }

    cudaSetDevice(0);
    g_multi_gpu_initialized = true;
}

// ============================================================================
// CUDA Kernels (static to avoid linker conflicts with layer.cu)
// ============================================================================

// Tiled matrix multiplication: C = A @ B^T
static __global__ void matmul_kernel(const float* __restrict__ A,
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

        tile_A[threadIdx.y][threadIdx.x] = (row < M && tile_col < K) ? A[row * K + tile_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (col < N && tile_row < K) ? B[col * K + tile_row] : 0.0f;

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
// SiLU activation kernel
static __global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x * (1.0f / (1.0f + expf(-x)));
    }
}

// Element-wise multiply kernel
static __global__ void mul_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// Attention score computation: scores[i,j] = sum_d Q[i,d] * K[j,d]
static __global__ void attention_scores_kernel(const float* __restrict__ Q,
                                        const float* __restrict__ K,
                                        float* __restrict__ scores,
                                        int seq_len, int head_dim, float scale) {
    int i = blockIdx.y;
    int j = blockIdx.x;

    if (i >= seq_len || j >= seq_len) return;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        sum += Q[i * head_dim + d] * K[j * head_dim + d];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x == 0) {
        scores[i * seq_len + j] = sum * scale;
    }
}

// Causal softmax kernel
static __global__ void causal_softmax_kernel(float* __restrict__ scores,
                                      int seq_len) {
    int i = blockIdx.x;
    if (i >= seq_len) return;

    float* row = scores + i * seq_len;

    // Apply causal mask and find max
    float max_val = -INFINITY;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        max_val = fmaxf(max_val, row[j]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max;
    if (threadIdx.x == 0) shared_max = max_val;
    __syncthreads();
    max_val = shared_max;

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = threadIdx.x; j <= i; j += blockDim.x) {
        float exp_val = expf(row[j] - max_val);
        row[j] = exp_val;
        sum += exp_val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float shared_sum;
    if (threadIdx.x == 0) shared_sum = sum;
    __syncthreads();
    sum = shared_sum;

    // Normalize
    float sum_inv = 1.0f / sum;
    for (int j = threadIdx.x; j < seq_len; j += blockDim.x) {
        row[j] = (j <= i) ? (row[j] * sum_inv) : 0.0f;
    }
}

// Attention output: out[i,d] = sum_j scores[i,j] * V[j,d]
static __global__ void attention_output_kernel(const float* __restrict__ scores,
                                        const float* __restrict__ V,
                                        float* __restrict__ output,
                                        int seq_len, int head_dim) {
    int i = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= seq_len || d >= head_dim) return;

    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        sum += scores[i * seq_len + j] * V[j * head_dim + d];
    }
    output[i * head_dim + d] = sum;
}

// ============================================================================
// OPTIMIZATION 2.1: Fused ShortConv Kernels
// ============================================================================

// Fused kernel: transpose (batch, seq_len, 3*hidden) -> (batch, 3*hidden, seq_len)
// and split into B, C, x_gate
static __global__ void fused_transpose_split_kernel(
    const float* __restrict__ in_proj_out,  // (batch*seq_len, 3*hidden_size)
    float* __restrict__ B,                   // (batch, hidden_size, seq_len)
    float* __restrict__ C,                   // (batch, hidden_size, seq_len)
    float* __restrict__ x_gate,              // (batch, hidden_size, seq_len)
    int batch, int seq_len, int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;

    if (idx >= total) return;

    int s = idx % seq_len;
    int h = (idx / seq_len) % hidden_size;
    int b = idx / (hidden_size * seq_len);

    // Input layout: (batch*seq_len, 3*hidden_size)
    int in_base = (b * seq_len + s) * 3 * hidden_size;

    // Output layout: (batch, hidden_size, seq_len)
    int out_idx = (b * hidden_size + h) * seq_len + s;

    B[out_idx] = in_proj_out[in_base + h];
    C[out_idx] = in_proj_out[in_base + h + hidden_size];
    x_gate[out_idx] = in_proj_out[in_base + h + 2 * hidden_size];
}

// Fused kernel: Bx = B * x_gate, then causal conv1d
static __global__ void fused_mul_conv1d_kernel(
    const float* __restrict__ B,
    const float* __restrict__ x_gate,
    const float* __restrict__ conv_weight,  // (hidden_size, kernel_size)
    float* __restrict__ conv_out,
    int batch, int hidden_size, int seq_len, int kernel_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * hidden_size * seq_len;

    if (idx >= total) return;

    int s = idx % seq_len;
    int h = (idx / seq_len) % hidden_size;
    int b = idx / (hidden_size * seq_len);

    float sum = 0.0f;
    for (int k = 0; k < kernel_size; k++) {
        int input_pos = s - (kernel_size - 1) + k;
        if (input_pos >= 0) {
            int input_idx = (b * hidden_size + h) * seq_len + input_pos;
            // Fused: B * x_gate then conv
            float bx_val = B[input_idx] * x_gate[input_idx];
            sum += bx_val * conv_weight[h * kernel_size + k];
        }
    }
    conv_out[idx] = sum;
}

// Fused kernel: y_pre = C * conv_out, then transpose to (batch, seq_len, hidden)
static __global__ void fused_mul_transpose_kernel(
    const float* __restrict__ C,
    const float* __restrict__ conv_out,
    float* __restrict__ output,  // (batch, seq_len, hidden_size)
    int batch, int seq_len, int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_size;

    if (idx >= total) return;

    int h = idx % hidden_size;
    int s = (idx / hidden_size) % seq_len;
    int b = idx / (seq_len * hidden_size);

    // Input layout: (batch, hidden_size, seq_len)
    int input_idx = (b * hidden_size + h) * seq_len + s;

    // Output layout: (batch, seq_len, hidden_size)
    output[idx] = C[input_idx] * conv_out[input_idx];
}

// Add bias kernel
static __global__ void add_bias_kernel(
    float* __restrict__ data,
    const float* __restrict__ bias,
    int rows, int cols) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;

    int col = idx % cols;
    data[idx] += bias[col];
}

// ============================================================================
// OPTIMIZATION 2.3: Embedding Lookup GPU Kernel
// ============================================================================

static __global__ void embedding_lookup_kernel(
    const int* __restrict__ tokens,
    const float* __restrict__ embed_table,
    float* __restrict__ output,
    int seq_len, int hidden_size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * hidden_size;

    if (idx >= total) return;

    int s = idx / hidden_size;
    int d = idx % hidden_size;

    int token_id = tokens[s];
    output[idx] = embed_table[token_id * hidden_size + d];
}

// ============================================================================
// OPTIMIZATION 2.2: Persistent GPU Buffer Manager
// ============================================================================

class PersistentGPUBuffers {
public:
    // Singleton instance
    static PersistentGPUBuffers& instance() {
        static PersistentGPUBuffers inst;
        return inst;
    }

    // Pre-allocated buffers for different operations
    float* shortconv_workspace = nullptr;
    size_t shortconv_workspace_size = 0;

    float* attention_workspace = nullptr;
    size_t attention_workspace_size = 0;

    float* embedding_d_tokens = nullptr;
    float* embedding_d_table = nullptr;
    float* embedding_d_output = nullptr;
    size_t embedding_table_size = 0;
    size_t embedding_max_seq = 0;

    bool initialized = false;

    void init(size_t max_batch, size_t max_seq_len, size_t hidden_size,
              size_t vocab_size, size_t num_heads, size_t head_dim) {
        if (initialized) return;

        // ShortConv workspace: B, C, x_gate, conv_out (each batch*hidden*seq)
        // Plus in_proj_out (batch*seq*3*hidden) and y_pre (batch*seq*hidden)
        size_t shortconv_size = max_batch * max_seq_len * hidden_size * 4 +  // B, C, x_gate, conv_out
                                max_batch * max_seq_len * 3 * hidden_size +   // in_proj_out
                                max_batch * max_seq_len * hidden_size;        // y_pre

        if (shortconv_size > shortconv_workspace_size) {
            if (shortconv_workspace) cudaFree(shortconv_workspace);
            cudaMalloc(&shortconv_workspace, shortconv_size * sizeof(float));
            shortconv_workspace_size = shortconv_size;
        }

        // Attention workspace: Q, K, V, scores, output
        size_t attn_qkv_size = max_batch * num_heads * max_seq_len * head_dim;
        size_t attn_scores_size = max_batch * num_heads * max_seq_len * max_seq_len;
        size_t attn_size = attn_qkv_size * 3 + attn_scores_size + attn_qkv_size;

        if (attn_size > attention_workspace_size) {
            if (attention_workspace) cudaFree(attention_workspace);
            cudaMalloc(&attention_workspace, attn_size * sizeof(float));
            attention_workspace_size = attn_size;
        }

        // Embedding buffers
        if (vocab_size * hidden_size > embedding_table_size) {
            if (embedding_d_table) cudaFree(embedding_d_table);
            cudaMalloc(&embedding_d_table, vocab_size * hidden_size * sizeof(float));
            embedding_table_size = vocab_size * hidden_size;
        }

        if (max_seq_len > embedding_max_seq) {
            if (embedding_d_tokens) cudaFree(embedding_d_tokens);
            if (embedding_d_output) cudaFree(embedding_d_output);
            cudaMalloc(&embedding_d_tokens, max_seq_len * sizeof(int));
            cudaMalloc(&embedding_d_output, max_batch * max_seq_len * hidden_size * sizeof(float));
            embedding_max_seq = max_seq_len;
        }

        initialized = true;
    }

    ~PersistentGPUBuffers() {
        if (shortconv_workspace) cudaFree(shortconv_workspace);
        if (attention_workspace) cudaFree(attention_workspace);
        if (embedding_d_tokens) cudaFree(embedding_d_tokens);
        if (embedding_d_table) cudaFree(embedding_d_table);
        if (embedding_d_output) cudaFree(embedding_d_output);
    }

private:
    PersistentGPUBuffers() = default;
    PersistentGPUBuffers(const PersistentGPUBuffers&) = delete;
    PersistentGPUBuffers& operator=(const PersistentGPUBuffers&) = delete;
};

// Global flag for embedding table upload
static bool g_embedding_uploaded = false;

// ============================================================================
// OPTIMIZATION 3.3: CUDA Stream Manager for Async Memory Transfers
// ============================================================================

class CUDAStreamManager {
public:
    static CUDAStreamManager& instance() {
        static CUDAStreamManager inst;
        return inst;
    }

    // Streams for different operations
    cudaStream_t compute_stream;
    cudaStream_t h2d_stream;  // Host to Device
    cudaStream_t d2h_stream;  // Device to Host

    bool initialized = false;

    void init() {
        if (initialized) return;

        cudaStreamCreate(&compute_stream);
        cudaStreamCreate(&h2d_stream);
        cudaStreamCreate(&d2h_stream);

        initialized = true;
    }

    ~CUDAStreamManager() {
        if (initialized) {
            cudaStreamDestroy(compute_stream);
            cudaStreamDestroy(h2d_stream);
            cudaStreamDestroy(d2h_stream);
        }
    }

    // Async H2D transfer
    void async_h2d(void* dst, const void* src, size_t size) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, h2d_stream);
    }

    // Async D2H transfer
    void async_d2h(void* dst, const void* src, size_t size) {
        cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, d2h_stream);
    }

    // Wait for H2D transfers to complete
    void sync_h2d() {
        cudaStreamSynchronize(h2d_stream);
    }

    // Wait for D2H transfers to complete
    void sync_d2h() {
        cudaStreamSynchronize(d2h_stream);
    }

    // Wait for compute to complete
    void sync_compute() {
        cudaStreamSynchronize(compute_stream);
    }

    // Wait for all streams
    void sync_all() {
        cudaStreamSynchronize(h2d_stream);
        cudaStreamSynchronize(d2h_stream);
        cudaStreamSynchronize(compute_stream);
    }

private:
    CUDAStreamManager() = default;
    CUDAStreamManager(const CUDAStreamManager&) = delete;
    CUDAStreamManager& operator=(const CUDAStreamManager&) = delete;
};

// Pinned memory allocator for faster H2D/D2H transfers
class PinnedMemoryPool {
public:
    static PinnedMemoryPool& instance() {
        static PinnedMemoryPool inst;
        return inst;
    }

    // Allocate pinned memory
    void* allocate(size_t size) {
        void* ptr;
        cudaMallocHost(&ptr, size);
        allocations_.push_back(ptr);
        return ptr;
    }

    // Free all allocations (call at program end)
    void free_all() {
        for (void* ptr : allocations_) {
            cudaFreeHost(ptr);
        }
        allocations_.clear();
    }

    ~PinnedMemoryPool() {
        free_all();
    }

private:
    std::vector<void*> allocations_;
    PinnedMemoryPool() = default;
    PinnedMemoryPool(const PinnedMemoryPool&) = delete;
    PinnedMemoryPool& operator=(const PinnedMemoryPool&) = delete;
};

// ============================================================================
// GPU Tensor Class for persistent GPU storage
// ============================================================================

class GPUTensor {
public:
    float* data_;
    std::vector<size_t> shape_;
    size_t size_;
    int device_id_;
    bool owns_data_;

    GPUTensor() : data_(nullptr), size_(0), device_id_(0), owns_data_(false) {}

    GPUTensor(const std::vector<size_t>& shape, int device_id = 0)
        : shape_(shape), device_id_(device_id), owns_data_(true) {
        size_ = 1;
        for (auto s : shape_) size_ *= s;

        cudaSetDevice(device_id_);
        cudaMalloc(&data_, size_ * sizeof(float));
    }

    ~GPUTensor() {
        if (owns_data_ && data_) {
            cudaSetDevice(device_id_);
            cudaFree(data_);
        }
    }

    // Move constructor
    GPUTensor(GPUTensor&& other) noexcept
        : data_(other.data_), shape_(std::move(other.shape_)),
          size_(other.size_), device_id_(other.device_id_), owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.owns_data_ = false;
    }

    GPUTensor& operator=(GPUTensor&& other) noexcept {
        if (this != &other) {
            if (owns_data_ && data_) {
                cudaSetDevice(device_id_);
                cudaFree(data_);
            }
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            size_ = other.size_;
            device_id_ = other.device_id_;
            owns_data_ = other.owns_data_;
            other.data_ = nullptr;
            other.owns_data_ = false;
        }
        return *this;
    }

    void upload(const Tensor& cpu_tensor) {
        cudaSetDevice(device_id_);
        cudaMemcpy(data_, cpu_tensor.data(), size_ * sizeof(float), cudaMemcpyHostToDevice);
    }

    void download(Tensor& cpu_tensor) const {
        cudaSetDevice(device_id_);
        cudaMemcpy(cpu_tensor.data(), data_, size_ * sizeof(float), cudaMemcpyDeviceToHost);
    }

    size_t size(int dim) const {
        if (dim < 0) dim += shape_.size();
        return shape_[dim];
    }
};

// ============================================================================
// GPU-accelerated Layer Implementations
// ============================================================================

// GPU MLP implementation
class GPUMLP {
public:
    GPUTensor w1_, w2_, w3_;
    int device_id_;

    GPUMLP(const std::string& w1_file, const std::string& w2_file,
           const std::string& w3_file, int device_id = 0) : device_id_(device_id) {
        Tensor w1_cpu = Tensor::load_from_file(w1_file);
        Tensor w2_cpu = Tensor::load_from_file(w2_file);
        Tensor w3_cpu = Tensor::load_from_file(w3_file);

        w1_ = GPUTensor(w1_cpu.shape(), device_id_);
        w2_ = GPUTensor(w2_cpu.shape(), device_id_);
        w3_ = GPUTensor(w3_cpu.shape(), device_id_);

        w1_.upload(w1_cpu);
        w2_.upload(w2_cpu);
        w3_.upload(w3_cpu);
    }

    void forward(float* d_input, float* d_output, float* workspace,
                 size_t batch_seq, size_t hidden_size, cudaStream_t stream) {
        size_t intermediate_size = w1_.size(0);

        // Workspace layout:
        // gate: batch_seq * intermediate_size
        // gate_silu: batch_seq * intermediate_size
        // up: batch_seq * intermediate_size
        // hidden: batch_seq * intermediate_size

        float* d_gate = workspace;
        float* d_gate_silu = workspace + batch_seq * intermediate_size;
        float* d_up = d_gate_silu + batch_seq * intermediate_size;
        float* d_hidden = d_up + batch_seq * intermediate_size;

        // gate = input @ w1^T
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid_gate((intermediate_size + TILE_SIZE - 1) / TILE_SIZE,
                       (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel<<<grid_gate, block, 0, stream>>>(
            d_input, w1_.data_, d_gate, batch_seq, intermediate_size, hidden_size);

        // gate_silu = silu(gate)
        int silu_blocks = (batch_seq * intermediate_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        silu_kernel<<<silu_blocks, BLOCK_SIZE, 0, stream>>>(
            d_gate, d_gate_silu, batch_seq * intermediate_size);

        // up = input @ w3^T
        matmul_kernel<<<grid_gate, block, 0, stream>>>(
            d_input, w3_.data_, d_up, batch_seq, intermediate_size, hidden_size);

        // hidden = gate_silu * up
        mul_kernel<<<silu_blocks, BLOCK_SIZE, 0, stream>>>(
            d_gate_silu, d_up, d_hidden, batch_seq * intermediate_size);

        // output = hidden @ w2^T
        dim3 grid_out((hidden_size + TILE_SIZE - 1) / TILE_SIZE,
                      (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel<<<grid_out, block, 0, stream>>>(
            d_hidden, w2_.data_, d_output, batch_seq, hidden_size, intermediate_size);
    }
};

// ============================================================================
// Large Block Implementations - Complex layers and modules
// ============================================================================

// MLP (Feed-Forward Network) implementation - allocate per-call for correctness
MLP::MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file) {
    w1_ = Tensor::load_from_file(w1_file);
    w2_ = Tensor::load_from_file(w2_file);
    w3_ = Tensor::load_from_file(w3_file);
    // GPU pointers not used - set to nullptr
    d_w1_ = nullptr;
    d_w2_ = nullptr;
    d_w3_ = nullptr;
}

MLP::~MLP() {
    // No GPU memory to free in per-call mode
}

// Move constructor
MLP::MLP(MLP&& other) noexcept
    : w1_(std::move(other.w1_)), w3_(std::move(other.w3_)), w2_(std::move(other.w2_)),
      d_w1_(nullptr), d_w2_(nullptr), d_w3_(nullptr) {
}

// Move assignment operator
MLP& MLP::operator=(MLP&& other) noexcept {
    if (this != &other) {
        w1_ = std::move(other.w1_);
        w2_ = std::move(other.w2_);
        w3_ = std::move(other.w3_);
    }
    return *this;
}

void MLP::forward(const Tensor& x, Tensor& y) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t intermediate_size = w1_.size(0);
    size_t batch_seq = batch * seq_len;

    // Allocate GPU memory
    float *d_input, *d_w1, *d_w2, *d_w3, *d_output;
    float *d_gate, *d_gate_silu, *d_up, *d_hidden;

    cudaMalloc(&d_input, batch_seq * hidden_size * sizeof(float));
    cudaMalloc(&d_w1, intermediate_size * hidden_size * sizeof(float));
    cudaMalloc(&d_w2, hidden_size * intermediate_size * sizeof(float));
    cudaMalloc(&d_w3, intermediate_size * hidden_size * sizeof(float));
    cudaMalloc(&d_output, batch_seq * hidden_size * sizeof(float));
    cudaMalloc(&d_gate, batch_seq * intermediate_size * sizeof(float));
    cudaMalloc(&d_gate_silu, batch_seq * intermediate_size * sizeof(float));
    cudaMalloc(&d_up, batch_seq * intermediate_size * sizeof(float));
    cudaMalloc(&d_hidden, batch_seq * intermediate_size * sizeof(float));

    // Upload data
    cudaMemcpy(d_input, x.data(), batch_seq * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w1, w1_.data(), intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w2, w2_.data(), hidden_size * intermediate_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w3, w3_.data(), intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);

    // gate = input @ w1^T
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid_gate((intermediate_size + TILE_SIZE - 1) / TILE_SIZE,
                   (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid_gate, block>>>(d_input, d_w1, d_gate, batch_seq, intermediate_size, hidden_size);

    // gate_silu = silu(gate)
    int silu_blocks = (batch_seq * intermediate_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_kernel<<<silu_blocks, BLOCK_SIZE>>>(d_gate, d_gate_silu, batch_seq * intermediate_size);

    // up = input @ w3^T
    matmul_kernel<<<grid_gate, block>>>(d_input, d_w3, d_up, batch_seq, intermediate_size, hidden_size);

    // hidden = gate_silu * up
    mul_kernel<<<silu_blocks, BLOCK_SIZE>>>(d_gate_silu, d_up, d_hidden, batch_seq * intermediate_size);

    // output = hidden @ w2^T
    dim3 grid_out((hidden_size + TILE_SIZE - 1) / TILE_SIZE,
                  (batch_seq + TILE_SIZE - 1) / TILE_SIZE);
    matmul_kernel<<<grid_out, block>>>(d_hidden, d_w2, d_output, batch_seq, hidden_size, intermediate_size);

    // Download result
    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    cudaMemcpy(y.data(), d_output, batch_seq * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_w3);
    cudaFree(d_output);
    cudaFree(d_gate);
    cudaFree(d_gate_silu);
    cudaFree(d_up);
    cudaFree(d_hidden);
}

// SparseMoeBlock implementation
SparseMoeBlock::SparseMoeBlock(int layer_idx) {
    std::stringstream ss;
    ss << "layers." << layer_idx << ".feed_forward.gate.weight";
    gate_ = Tensor::load_from_file(ss.str());

    experts_.reserve(NUM_EXPERTS);
    for (size_t i = 0; i < NUM_EXPERTS; i++) {
        std::stringstream ss_w1, ss_w2, ss_w3;
        ss_w1 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w1.weight";
        ss_w2 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w2.weight";
        ss_w3 << "layers." << layer_idx << ".feed_forward.experts." << i << ".w3.weight";

        experts_.emplace_back(ss_w1.str(), ss_w2.str(), ss_w3.str());
    }

    if (USE_EXPERT_BIAS) {
        std::stringstream ss_bias;
        ss_bias << "layers." << layer_idx << ".feed_forward.expert_bias";
        expert_bias_ = Tensor::load_from_file(ss_bias.str());
    }
}

void SparseMoeBlock::route_tokens(const Tensor& router_logits,
                                   std::vector<int>& top_k_indices,
                                   std::vector<float>& top_k_weights) {
    size_t num_tokens = router_logits.size(0);

    top_k_indices.resize(num_tokens * NUM_EXPERTS_PER_TOK);
    top_k_weights.resize(num_tokens * NUM_EXPERTS_PER_TOK);

    #pragma omp parallel for
    for (size_t t = 0; t < num_tokens; t++) {
        std::vector<float> routing_weights(NUM_EXPERTS);
        for (size_t e = 0; e < NUM_EXPERTS; e++) {
            float logit = router_logits.at(t, e);
            routing_weights[e] = 1.0f * (1.0f / (1.0f + std::exp(-logit)));
        }

        std::vector<std::pair<float, int>> scores(NUM_EXPERTS);
        if (USE_EXPERT_BIAS) {
            for (size_t e = 0; e < NUM_EXPERTS; e++) {
                scores[e] = {routing_weights[e] + expert_bias_[e], e};
            }
        } else {
            for (size_t e = 0; e < NUM_EXPERTS; e++) {
                scores[e] = {routing_weights[e], e};
            }
        }

        std::partial_sort(scores.begin(), scores.begin() + NUM_EXPERTS_PER_TOK, scores.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });

        std::vector<float> selected_weights(NUM_EXPERTS_PER_TOK);
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            int expert_idx = scores[k].second;
            top_k_indices[t * NUM_EXPERTS_PER_TOK + k] = expert_idx;
            selected_weights[k] = routing_weights[expert_idx];
        }

        if (NORM_TOPK_PROB) {
            float sum = 0.0f;
            for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
                sum += selected_weights[k];
            }
            if (sum > 1e-6f) {
                for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
                    selected_weights[k] /= sum;
                }
            }
        }

        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            top_k_weights[t * NUM_EXPERTS_PER_TOK + k] = selected_weights[k] * ROUTED_SCALING_FACTOR;
        }
    }
}

void SparseMoeBlock::forward(const Tensor& x, Tensor& y, Tensor& router_logits) {
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t num_tokens = batch * seq_len;
    size_t intermediate_size = MOE_INTERMEDIATE_SIZE;

    Tensor x_flat = x.view({num_tokens, hidden_size});

    router_logits = Tensor({num_tokens, NUM_EXPERTS});
    tensor_ops::matmul_transposed(x_flat, gate_, router_logits);

    std::vector<int> top_k_indices;
    std::vector<float> top_k_weights;
    route_tokens(router_logits, top_k_indices, top_k_weights);

    y = Tensor({batch, seq_len, hidden_size});
    y.zero();

    // =========================================================================
    // OPTIMIZATION 1.1: Batched Multi-GPU Expert Processing
    // =========================================================================

    // Build token-to-expert assignment matrix for batch processing
    // expert_tokens[e] = list of (token_idx, weight) pairs for expert e
    std::vector<std::vector<std::pair<size_t, float>>> expert_tokens(NUM_EXPERTS);

    for (size_t t = 0; t < num_tokens; t++) {
        for (size_t k = 0; k < NUM_EXPERTS_PER_TOK; k++) {
            int expert_idx = top_k_indices[t * NUM_EXPERTS_PER_TOK + k];
            float weight = top_k_weights[t * NUM_EXPERTS_PER_TOK + k];
            expert_tokens[expert_idx].push_back({t, weight});
        }
    }

    // Get number of available GPUs
    int device_count = 1;
    cudaGetDeviceCount(&device_count);
    int num_gpus = std::min(device_count, (int)NUM_GPUS);

    // Allocate output accumulator on CPU (thread-safe accumulation)
    std::vector<float> y_accum(num_tokens * hidden_size, 0.0f);

    // Process experts in parallel across multiple GPUs
    // Each GPU processes a subset of experts concurrently
    #pragma omp parallel for schedule(dynamic)
    for (size_t e = 0; e < NUM_EXPERTS; e++) {
        if (expert_tokens[e].empty()) continue;

        // Assign expert to GPU based on expert index (round-robin)
        int gpu_id = e % num_gpus;
        cudaSetDevice(gpu_id);

        size_t batch_size = expert_tokens[e].size();

        // Allocate GPU memory for this expert batch
        float *d_input, *d_output;
        float *d_w1, *d_w2, *d_w3;
        float *d_gate, *d_gate_silu, *d_up, *d_hidden;

        cudaMalloc(&d_input, batch_size * hidden_size * sizeof(float));
        cudaMalloc(&d_output, batch_size * hidden_size * sizeof(float));
        cudaMalloc(&d_w1, intermediate_size * hidden_size * sizeof(float));
        cudaMalloc(&d_w2, hidden_size * intermediate_size * sizeof(float));
        cudaMalloc(&d_w3, intermediate_size * hidden_size * sizeof(float));
        cudaMalloc(&d_gate, batch_size * intermediate_size * sizeof(float));
        cudaMalloc(&d_gate_silu, batch_size * intermediate_size * sizeof(float));
        cudaMalloc(&d_up, batch_size * intermediate_size * sizeof(float));
        cudaMalloc(&d_hidden, batch_size * intermediate_size * sizeof(float));

        // Create batched input tensor on CPU
        std::vector<float> batch_in_data(batch_size * hidden_size);
        for (size_t i = 0; i < batch_size; i++) {
            size_t t = expert_tokens[e][i].first;
            for (size_t h = 0; h < hidden_size; h++) {
                batch_in_data[i * hidden_size + h] = x_flat.at(t, h);
            }
        }

        // Upload input and weights to GPU
        cudaMemcpy(d_input, batch_in_data.data(), batch_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w1, experts_[e].w1().data(), intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w2, experts_[e].w2().data(), hidden_size * intermediate_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_w3, experts_[e].w3().data(), intermediate_size * hidden_size * sizeof(float), cudaMemcpyHostToDevice);

        // Run MLP forward pass on GPU
        // gate = input @ w1^T
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid_gate((intermediate_size + TILE_SIZE - 1) / TILE_SIZE,
                       (batch_size + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel<<<grid_gate, block>>>(d_input, d_w1, d_gate, batch_size, intermediate_size, hidden_size);

        // gate_silu = silu(gate)
        int silu_blocks = (batch_size * intermediate_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        silu_kernel<<<silu_blocks, BLOCK_SIZE>>>(d_gate, d_gate_silu, batch_size * intermediate_size);

        // up = input @ w3^T
        matmul_kernel<<<grid_gate, block>>>(d_input, d_w3, d_up, batch_size, intermediate_size, hidden_size);

        // hidden = gate_silu * up
        mul_kernel<<<silu_blocks, BLOCK_SIZE>>>(d_gate_silu, d_up, d_hidden, batch_size * intermediate_size);

        // output = hidden @ w2^T
        dim3 grid_out((hidden_size + TILE_SIZE - 1) / TILE_SIZE,
                      (batch_size + TILE_SIZE - 1) / TILE_SIZE);
        matmul_kernel<<<grid_out, block>>>(d_hidden, d_w2, d_output, batch_size, hidden_size, intermediate_size);

        // Download result
        std::vector<float> batch_out_data(batch_size * hidden_size);
        cudaMemcpy(batch_out_data.data(), d_output, batch_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_w1);
        cudaFree(d_w2);
        cudaFree(d_w3);
        cudaFree(d_gate);
        cudaFree(d_gate_silu);
        cudaFree(d_up);
        cudaFree(d_hidden);

        // Accumulate weighted outputs (thread-safe with atomic or critical section)
        for (size_t i = 0; i < batch_size; i++) {
            size_t t = expert_tokens[e][i].first;
            float weight = expert_tokens[e][i].second;

            for (size_t h = 0; h < hidden_size; h++) {
                #pragma omp atomic
                y_accum[t * hidden_size + h] += weight * batch_out_data[i * hidden_size + h];
            }
        }
    }

    // Copy accumulated results to output tensor
    for (size_t t = 0; t < num_tokens; t++) {
        size_t b = t / seq_len;
        size_t s = t % seq_len;
        for (size_t h = 0; h < hidden_size; h++) {
            y.at(b, s, h) = y_accum[t * hidden_size + h];
        }
    }

    // Reset to default GPU
    cudaSetDevice(0);
}

// Attention implementation
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
    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);

    // Flatten
    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // Project Q, K, V (using GPU-accelerated matmul from tensor_ops)
    Tensor q_proj_out({batch * seq_len, NUM_ATTENTION_HEADS * HEAD_DIM});
    Tensor k_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});
    Tensor v_proj_out({batch * seq_len, NUM_KEY_VALUE_HEADS * HEAD_DIM});

    tensor_ops::matmul_transposed(x_flat, q_proj_, q_proj_out);
    tensor_ops::matmul_transposed(x_flat, k_proj_, k_proj_out);
    tensor_ops::matmul_transposed(x_flat, v_proj_, v_proj_out);

    // Reshape to (batch, seq_len, num_heads, head_dim) for layernorm
    Tensor q_reshaped({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    Tensor v_reshaped({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    q_reshaped.at(b, s, h, d) = q_proj_out.at(b * seq_len + s, h * HEAD_DIM + d);
                }
            }
            for (size_t h = 0; h < NUM_KEY_VALUE_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    k_reshaped.at(b, s, h, d) = k_proj_out.at(b * seq_len + s, h * HEAD_DIM + d);
                    v_reshaped.at(b, s, h, d) = v_proj_out.at(b * seq_len + s, h * HEAD_DIM + d);
                }
            }
        }
    }

    // Apply layernorm to Q and K (normalizes over last dim = head_dim)
    Tensor q_normed({batch, seq_len, NUM_ATTENTION_HEADS, HEAD_DIM});
    Tensor k_normed({batch, seq_len, NUM_KEY_VALUE_HEADS, HEAD_DIM});
    q_layernorm_->forward(q_reshaped, q_normed);
    k_layernorm_->forward(k_reshaped, k_normed);

    // Transpose to (batch, num_heads, seq_len, head_dim) for attention
    Tensor q({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor k({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});
    Tensor v({batch, NUM_KEY_VALUE_HEADS, seq_len, HEAD_DIM});

    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    q.at(b, h, s, d) = q_normed.at(b, s, h, d);
                }
            }
            for (size_t h = 0; h < NUM_KEY_VALUE_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    k.at(b, h, s, d) = k_normed.at(b, s, h, d);
                    v.at(b, h, s, d) = v_reshaped.at(b, s, h, d);
                }
            }
        }
    }

    // Apply RoPE
    tensor_ops::apply_rotary_pos_emb(q, k, cos, sin);

    // Repeat K, V for GQA
    Tensor k_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    Tensor v_repeated({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});
    tensor_ops::repeat_kv(k, NUM_KEY_VALUE_GROUPS, k_repeated);
    tensor_ops::repeat_kv(v, NUM_KEY_VALUE_GROUPS, v_repeated);

    // =========================================================================
    // OPTIMIZATION 1.2: GPU Attention Kernels
    // Replace CPU attention loops with GPU kernel calls
    // Uses attn_scores_kernel, causal_softmax_kernel, attn_output_kernel
    // =========================================================================

    float scale = 1.0f * (1.0f / std::sqrt((float)HEAD_DIM));
    Tensor attn_output({batch, NUM_ATTENTION_HEADS, seq_len, HEAD_DIM});

    // Allocate GPU memory for attention computation
    size_t qkv_size = batch * NUM_ATTENTION_HEADS * seq_len * HEAD_DIM;
    size_t scores_size = batch * NUM_ATTENTION_HEADS * seq_len * seq_len;

    float *d_q, *d_k, *d_v, *d_scores, *d_attn_output;

    cudaMalloc(&d_q, qkv_size * sizeof(float));
    cudaMalloc(&d_k, qkv_size * sizeof(float));
    cudaMalloc(&d_v, qkv_size * sizeof(float));
    cudaMalloc(&d_scores, scores_size * sizeof(float));
    cudaMalloc(&d_attn_output, qkv_size * sizeof(float));

    // Upload Q, K, V to GPU (already in (batch, num_heads, seq_len, head_dim) format)
    cudaMemcpy(d_q, q.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_repeated.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_repeated.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch attention kernels for all batches and heads
    // Grid: (seq_len, batch * num_heads) for scores computation
    // Each block handles one query position

    // Compute attention scores: Q @ K^T with scaling
    // Using the attention_scores_kernel from layer.cu interface
    dim3 scores_grid(seq_len, batch * NUM_ATTENTION_HEADS);
    dim3 scores_block(std::min((int)seq_len, 256));

    // Process each (batch, head) pair
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
            size_t bh_offset = (b * NUM_ATTENTION_HEADS + h);
            float* q_ptr = d_q + bh_offset * seq_len * HEAD_DIM;
            float* k_ptr = d_k + bh_offset * seq_len * HEAD_DIM;
            float* scores_ptr = d_scores + bh_offset * seq_len * seq_len;
            float* v_ptr = d_v + bh_offset * seq_len * HEAD_DIM;
            float* out_ptr = d_attn_output + bh_offset * seq_len * HEAD_DIM;

            // Compute attention scores: scores[i,j] = sum_d Q[i,d] * K[j,d] * scale
            dim3 attn_scores_grid(seq_len, seq_len);
            attention_scores_kernel<<<attn_scores_grid, 32>>>(
                q_ptr, k_ptr, scores_ptr, seq_len, HEAD_DIM, scale);

            // Apply causal mask and softmax
            causal_softmax_kernel<<<seq_len, 32>>>(scores_ptr, seq_len);

            // Compute attention output: out[i,d] = sum_j scores[i,j] * V[j,d]
            dim3 attn_out_grid((HEAD_DIM + 31) / 32, seq_len);
            attention_output_kernel<<<attn_out_grid, 32>>>(
                scores_ptr, v_ptr, out_ptr, seq_len, HEAD_DIM);
        }
    }

    // Download attention output
    cudaMemcpy(attn_output.data(), d_attn_output, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_scores);
    cudaFree(d_attn_output);

    // Reshape and project output
    Tensor attn_flat({batch * seq_len, hidden_size});
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < seq_len; s++) {
            for (size_t h = 0; h < NUM_ATTENTION_HEADS; h++) {
                for (size_t d = 0; d < HEAD_DIM; d++) {
                    attn_flat.at(b * seq_len + s, h * HEAD_DIM + d) = attn_output.at(b, h, s, d);
                }
            }
        }
    }

    Tensor output_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(attn_flat, o_proj_, output_flat);

    output_flat.reshape({batch, seq_len, hidden_size});

    // Allocate output if needed
    if (output.size() == 0) {
        output = Tensor({batch, seq_len, hidden_size});
    }
    std::memcpy(output.data(), output_flat.data(), output.size() * sizeof(float));
}

// ShortConv implementation
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
    // =========================================================================
    // OPTIMIZATION 2.1: Fused ShortConv with GPU Kernels
    // Combines: in_proj -> transpose -> split -> B*x_gate -> conv1d -> C*conv -> transpose -> out_proj
    // =========================================================================

    // x: (batch, seq_len, hidden_size)
    size_t batch = x.size(0);
    size_t seq_len = x.size(1);
    size_t hidden_size = x.size(2);
    size_t kernel_size = CONV_KERNEL;

    // Flatten for matmul
    Tensor x_flat = x.view({batch * seq_len, hidden_size});

    // in_proj: (batch*seq_len, hidden_size) @ (3*hidden_size, hidden_size)^T -> (batch*seq_len, 3*hidden_size)
    Tensor in_proj_out({batch * seq_len, 3 * hidden_size});
    tensor_ops::matmul_transposed(x_flat, in_proj_weight_, in_proj_out);

    // Allocate GPU memory for fused operations
    size_t bhs = batch * hidden_size * seq_len;
    float *d_in_proj, *d_B, *d_C, *d_x_gate, *d_conv_out, *d_y_pre;
    float *d_conv_weight, *d_in_bias, *d_out_bias;

    cudaMalloc(&d_in_proj, batch * seq_len * 3 * hidden_size * sizeof(float));
    cudaMalloc(&d_B, bhs * sizeof(float));
    cudaMalloc(&d_C, bhs * sizeof(float));
    cudaMalloc(&d_x_gate, bhs * sizeof(float));
    cudaMalloc(&d_conv_out, bhs * sizeof(float));
    cudaMalloc(&d_y_pre, batch * seq_len * hidden_size * sizeof(float));
    cudaMalloc(&d_conv_weight, hidden_size * kernel_size * sizeof(float));

    // Upload data to GPU
    cudaMemcpy(d_in_proj, in_proj_out.data(), batch * seq_len * 3 * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_conv_weight, conv_weight_.data(), hidden_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    // Add in_proj bias on GPU if present
    if (USE_CONV_BIAS && in_proj_bias_.size() > 0) {
        cudaMalloc(&d_in_bias, 3 * hidden_size * sizeof(float));
        cudaMemcpy(d_in_bias, in_proj_bias_.data(), 3 * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        int bias_blocks = (batch * seq_len * 3 * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<<<bias_blocks, BLOCK_SIZE>>>(d_in_proj, d_in_bias, batch * seq_len, 3 * hidden_size);
        cudaFree(d_in_bias);
    }

    // Fused transpose and split: (batch*seq_len, 3*hidden) -> B, C, x_gate (each batch, hidden, seq)
    int split_blocks = (bhs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_transpose_split_kernel<<<split_blocks, BLOCK_SIZE>>>(
        d_in_proj, d_B, d_C, d_x_gate, batch, seq_len, hidden_size);

    // Fused B*x_gate and causal conv1d
    fused_mul_conv1d_kernel<<<split_blocks, BLOCK_SIZE>>>(
        d_B, d_x_gate, d_conv_weight, d_conv_out, batch, hidden_size, seq_len, kernel_size);

    // Fused C*conv_out and transpose to (batch, seq_len, hidden)
    int transpose_blocks = (batch * seq_len * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_mul_transpose_kernel<<<transpose_blocks, BLOCK_SIZE>>>(
        d_C, d_conv_out, d_y_pre, batch, seq_len, hidden_size);

    // Download y_pre for out_proj matmul
    Tensor y_pre_transposed({batch, seq_len, hidden_size});
    cudaMemcpy(y_pre_transposed.data(), d_y_pre, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_in_proj);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_x_gate);
    cudaFree(d_conv_out);
    cudaFree(d_y_pre);
    cudaFree(d_conv_weight);

    // out_proj: (batch*seq_len, hidden_size) @ (hidden_size, hidden_size)^T -> (batch*seq_len, hidden_size)
    Tensor y_pre_flat = y_pre_transposed.view({batch * seq_len, hidden_size});
    Tensor y_flat({batch * seq_len, hidden_size});
    tensor_ops::matmul_transposed(y_pre_flat, out_proj_weight_, y_flat);

    // Add out_proj bias if present (on GPU for efficiency)
    if (USE_CONV_BIAS && out_proj_bias_.size() > 0) {
        float *d_y_flat;
        cudaMalloc(&d_y_flat, batch * seq_len * hidden_size * sizeof(float));
        cudaMalloc(&d_out_bias, hidden_size * sizeof(float));
        cudaMemcpy(d_y_flat, y_flat.data(), batch * seq_len * hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_out_bias, out_proj_bias_.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
        int bias_blocks = (batch * seq_len * hidden_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_bias_kernel<<<bias_blocks, BLOCK_SIZE>>>(d_y_flat, d_out_bias, batch * seq_len, hidden_size);
        cudaMemcpy(y_flat.data(), d_y_flat, batch * seq_len * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_y_flat);
        cudaFree(d_out_bias);
    }

    // Reshape back to (batch, seq_len, hidden_size)
    y_flat.reshape({batch, seq_len, hidden_size});

    if (y.size() == 0) {
        y = Tensor({batch, seq_len, hidden_size});
    }
    std::memcpy(y.data(), y_flat.data(), y.size() * sizeof(float));
}

// DecoderLayer implementation
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
    // Input norm
    Tensor normed_input(x.shape());
    input_layernorm_->forward(x, normed_input);

    // Attention or Conv
    Tensor attn_output(x.shape());
    if (is_attention_layer_) {
        self_attn_->forward(normed_input, cos, sin, attention_mask, attn_output);
    } else {
        short_conv_->forward(normed_input, attn_output);
    }

    // Residual connection
    Tensor hidden_states(x.shape());
    tensor_ops::add(x, attn_output, hidden_states);

    // Post attention norm
    Tensor normed_hidden(x.shape());
    post_attention_layernorm_->forward(hidden_states, normed_hidden);

    // MoE block or dense MLP
    Tensor ffn_output;
    if (moe_block_) {
        // MoE layer (layers >= 2)
        Tensor router_logits;
        moe_block_->forward(normed_hidden, ffn_output, router_logits);
    } else {
        // Dense layer (layers 0-1)
        dense_mlp_->forward(normed_hidden, ffn_output);
    }

    // Residual connection
    tensor_ops::add(hidden_states, ffn_output, output);
}

// ============================================================================
// LFM2Model Implementation - Complete model
// ============================================================================

LFM2Model::LFM2Model(const std::string& model_file) {
    std::cout << "Loading LFM2-8B-A1B model from " << model_file << std::endl;

    // Initialize multi-GPU
    init_multi_gpu();

    // Initialize global model loader
    g_model_loader = std::make_unique<ModelLoader>(model_file);

    load_embeddings();
    load_layers();
    load_output_layers();

    rotary_emb_ = std::make_unique<RotaryEmbedding>();

    std::cout << "Model loaded successfully!" << std::endl;
}

void LFM2Model::load_embeddings() {
    std::cout << "Loading embeddings..." << std::endl;
    embed_tokens_ = Tensor::load_from_file("embed_tokens.weight");
    std::cout << "  Embeddings shape: " << embed_tokens_.size(0) << " x " << embed_tokens_.size(1) << std::endl;
}

void LFM2Model::load_layers() {
    std::cout << "Loading " << NUM_HIDDEN_LAYERS << " decoder layers..." << std::endl;

    layers_.reserve(NUM_HIDDEN_LAYERS);
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        bool is_attention = (LAYER_TYPES[i] == 0);
        std::cout << "  Layer " << i << ": " << (is_attention ? "Attention" : "Conv") << std::endl;
        layers_.push_back(std::make_unique<DecoderLayer>(i, is_attention));
    }
}

void LFM2Model::load_output_layers() {
    std::cout << "Loading output layers..." << std::endl;

    norm_ = std::make_unique<RMSNorm>("embedding_norm.weight");

    if (g_model_loader->has_tensor("lm_head.weight")) {
        lm_head_ = Tensor::load_from_file("lm_head.weight");
    } else {
        lm_head_ = embed_tokens_;
        std::cout << "  Using tied weights for LM head" << std::endl;
    }
}

void LFM2Model::forward(const std::vector<int>& input_ids, Tensor& logits) {
    size_t batch = 1;
    size_t seq_len = input_ids.size();

    // =========================================================================
    // OPTIMIZATION 2.3: GPU Embedding Lookup
    // =========================================================================

    Tensor hidden_states({batch, seq_len, HIDDEN_SIZE});

    // Allocate GPU memory for embedding lookup
    int* d_tokens;
    float *d_embed_table, *d_output;

    cudaMalloc(&d_tokens, seq_len * sizeof(int));
    cudaMalloc(&d_embed_table, VOCAB_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, seq_len * HIDDEN_SIZE * sizeof(float));

    // Upload tokens and embedding table to GPU
    cudaMemcpy(d_tokens, input_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice);

    // Upload embedding table (only once per model load would be better, but safe for now)
    if (!g_embedding_uploaded) {
        cudaMemcpy(d_embed_table, embed_tokens_.data(), VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        g_embedding_uploaded = true;
    } else {
        cudaMemcpy(d_embed_table, embed_tokens_.data(), VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Launch embedding lookup kernel
    int total_elements = seq_len * HIDDEN_SIZE;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    embedding_lookup_kernel<<<blocks, BLOCK_SIZE>>>(d_tokens, d_embed_table, d_output, seq_len, HIDDEN_SIZE);

    // Download result
    cudaMemcpy(hidden_states.data(), d_output, seq_len * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_tokens);
    cudaFree(d_embed_table);
    cudaFree(d_output);

    // Compute RoPE embeddings
    Tensor cos({seq_len, HEAD_DIM});
    Tensor sin({seq_len, HEAD_DIM});
    rotary_emb_->forward(seq_len, cos, sin);

    // Create causal attention mask (not strictly needed for CPU impl)
    Tensor* attention_mask = nullptr;

    // Pass through decoder layers
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        Tensor output({batch, seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = output;
    }

    // Final norm
    Tensor normed_output({batch, seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output);

    // LM head projection (only for last token in generation)
    Tensor last_hidden({batch, 1, HIDDEN_SIZE});
    for (size_t i = 0; i < HIDDEN_SIZE; i++) {
        last_hidden.at(0, 0, i) = normed_output.at(0, seq_len - 1, i);
    }

    Tensor last_hidden_flat = last_hidden.view({batch, HIDDEN_SIZE});
    logits = Tensor({batch, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden_flat, lm_head_, logits);
}

// ============================================================================
// OPTIMIZATION 3.2: Batched Forward Pass for Multiple Sequences
// ============================================================================

void LFM2Model::forward_batch(const std::vector<std::vector<int>>& input_ids_batch,
                               std::vector<Tensor>& logits_batch) {
    size_t num_sequences = input_ids_batch.size();
    if (num_sequences == 0) return;

    // Find max sequence length for padding
    size_t max_seq_len = 0;
    for (const auto& seq : input_ids_batch) {
        max_seq_len = std::max(max_seq_len, seq.size());
    }

    // Process sequences in true batch (batch > 1)
    size_t batch = num_sequences;

    // Create padded input tensor
    Tensor hidden_states({batch, max_seq_len, HIDDEN_SIZE});
    hidden_states.zero();

    // GPU-accelerated batched embedding lookup
    int* d_tokens;
    float *d_embed_table, *d_output;
    size_t total_tokens = batch * max_seq_len;

    cudaMalloc(&d_tokens, total_tokens * sizeof(int));
    cudaMalloc(&d_embed_table, VOCAB_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_output, total_tokens * HIDDEN_SIZE * sizeof(float));

    // Create padded token array (pad with 0s)
    std::vector<int> padded_tokens(total_tokens, 0);
    for (size_t b = 0; b < batch; b++) {
        for (size_t s = 0; s < input_ids_batch[b].size(); s++) {
            padded_tokens[b * max_seq_len + s] = input_ids_batch[b][s];
        }
    }

    // Upload tokens and embedding table
    cudaMemcpy(d_tokens, padded_tokens.data(), total_tokens * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embed_table, embed_tokens_.data(), VOCAB_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch batched embedding lookup
    int total_elements = total_tokens * HIDDEN_SIZE;
    int blocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    embedding_lookup_kernel<<<blocks, BLOCK_SIZE>>>(d_tokens, d_embed_table, d_output, total_tokens, HIDDEN_SIZE);

    // Download embeddings
    cudaMemcpy(hidden_states.data(), d_output, total_tokens * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_tokens);
    cudaFree(d_embed_table);
    cudaFree(d_output);

    // Compute RoPE embeddings
    Tensor cos({max_seq_len, HEAD_DIM});
    Tensor sin({max_seq_len, HEAD_DIM});
    rotary_emb_->forward(max_seq_len, cos, sin);

    // Pass through decoder layers
    Tensor* attention_mask = nullptr;
    for (size_t i = 0; i < NUM_HIDDEN_LAYERS; i++) {
        Tensor output({batch, max_seq_len, HIDDEN_SIZE});
        layers_[i]->forward(hidden_states, cos, sin, attention_mask, output);
        hidden_states = output;
    }

    // Final norm
    Tensor normed_output({batch, max_seq_len, HIDDEN_SIZE});
    norm_->forward(hidden_states, normed_output);

    // Extract last token for each sequence and compute logits
    logits_batch.resize(batch);

    // Get actual sequence lengths for each batch item
    std::vector<size_t> seq_lengths(batch);
    for (size_t b = 0; b < batch; b++) {
        seq_lengths[b] = input_ids_batch[b].size();
    }

    // Process all sequences together for LM head
    Tensor last_hidden({batch, 1, HIDDEN_SIZE});
    for (size_t b = 0; b < batch; b++) {
        size_t last_pos = seq_lengths[b] - 1;
        for (size_t i = 0; i < HIDDEN_SIZE; i++) {
            last_hidden.at(b, 0, i) = normed_output.at(b, last_pos, i);
        }
    }

    Tensor last_hidden_flat = last_hidden.view({batch, HIDDEN_SIZE});
    Tensor all_logits({batch, VOCAB_SIZE});
    tensor_ops::matmul_transposed(last_hidden_flat, lm_head_, all_logits);

    // Split logits into individual tensors
    for (size_t b = 0; b < batch; b++) {
        logits_batch[b] = Tensor({1, VOCAB_SIZE});
        for (size_t v = 0; v < VOCAB_SIZE; v++) {
            logits_batch[b].at(0, v) = all_logits.at(b, v);
        }
    }
}
