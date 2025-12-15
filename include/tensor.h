#pragma once

#include <cstddef>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

/**
 * GPU Memory Pool for reducing cudaMalloc/cudaFree overhead.
 * Uses size-bucketing strategy to reuse GPU memory buffers.
 */
class GPUMemoryPool {
public:
    static GPUMemoryPool& instance();
    float* allocate(size_t num_elements);
    void deallocate(float* ptr, size_t num_elements);
    void clear();

    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

private:
    GPUMemoryPool();
    ~GPUMemoryPool();
    size_t bucket_size(size_t num_elements) const;

    std::unordered_map<size_t, std::vector<float*>> free_buffers_;
    mutable std::mutex mutex_;
};

inline GPUMemoryPool& gpu_memory_pool() {
    return GPUMemoryPool::instance();
}

// Forward declaration
class ModelLoader;

class Tensor {
public:
    // Constructors
    Tensor();
    Tensor(const std::vector<size_t>& shape, bool on_gpu = true);
    Tensor(const std::vector<size_t>& shape, float* data, bool copy = true, bool on_gpu = false);
    ~Tensor();

    // Copy constructor and assignment
    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);

    // Move constructor and assignment
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Shape operations
    size_t ndim() const { return shape_.size(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size() const { return size_; }
    size_t size(int dim) const;

    // Data access - GPU data
    float* data() { return d_data_; }
    const float* data() const { return d_data_; }

    // CPU data access (for compatibility)
    float* host_data();
    const float* host_data() const;

    float& operator[](size_t idx);
    const float& operator[](size_t idx) const;

    // Element access (CPU side - syncs if needed)
    float& at(size_t i);
    float& at(size_t i, size_t j);
    float& at(size_t i, size_t j, size_t k);
    float& at(size_t i, size_t j, size_t k, size_t l);

    const float& at(size_t i) const;
    const float& at(size_t i, size_t j) const;
    const float& at(size_t i, size_t j, size_t k) const;
    const float& at(size_t i, size_t j, size_t k, size_t l) const;

    // Reshape
    void reshape(const std::vector<size_t>& new_shape);
    Tensor view(const std::vector<size_t>& new_shape) const;

    // IO operations
    static Tensor load_from_file(const std::string& filename, ModelLoader* loader = nullptr);
    void save_to_file(const std::string& filename) const;

    // Tensor operations
    Tensor transpose(int dim0, int dim1) const;
    Tensor slice(int dim, size_t start, size_t end) const;
    Tensor copy() const;

    // Fill operations
    void fill(float value);
    void zero();
    void ones();

    // GPU/CPU sync operations
    void to_gpu();
    void to_cpu();
    void sync_to_host() const;
    void sync_to_device();
    bool is_on_gpu() const { return on_gpu_; }

    // Async GPU/CPU sync operations with CUDA streams
    void sync_to_host_async(cudaStream_t stream) const;
    void sync_to_device_async(cudaStream_t stream);

    // Mark device data as valid (after manual cudaMemcpy)
    void mark_device_valid() { device_valid_ = true; host_valid_ = false; }
    void mark_host_valid() const { host_valid_ = true; }

private:
    std::vector<size_t> shape_;
    size_t size_;
    float* d_data_;           // GPU data
    mutable float* h_data_;   // CPU data (cached)
    bool owns_data_;
    bool on_gpu_;
    mutable bool host_valid_; // Is host cache valid?
    mutable bool device_valid_; // Is device data valid?
    mutable bool host_pinned_; // Is host memory pinned?

    void allocate();
    void deallocate();
    size_t compute_size() const;
    size_t compute_stride(int dim) const;
    void ensure_host_data() const;
};

// Tensor operations
namespace tensor_ops {
    // Matrix operations
    void matmul(const Tensor& a, const Tensor& b, Tensor& c);
    void matmul_transposed(const Tensor& a, const Tensor& b, Tensor& c); // c = a @ b^T
    
    // Element-wise operations
    void add(const Tensor& a, const Tensor& b, Tensor& c);
    void add_scalar(const Tensor& a, float b, Tensor& c);
    void mul(const Tensor& a, const Tensor& b, Tensor& c);
    void mul_scalar(const Tensor& a, float b, Tensor& c);
    
    // Activation functions
    void silu(const Tensor& x, Tensor& y); // SiLU(x) = x * sigmoid(x)
    void sigmoid(const Tensor& x, Tensor& y);
    void softmax(const Tensor& x, Tensor& y, int dim);
    
    // Normalization
    void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& y);
    
    // RoPE (Rotary Position Embedding)
    void apply_rotary_pos_emb(Tensor& q, Tensor& k, const Tensor& cos, const Tensor& sin);
    void compute_rope_embeddings(size_t head_dim, size_t max_seq_len, float theta, 
                                 Tensor& cos, Tensor& sin);
    
    // Repeat KV for GQA (Grouped Query Attention)
    void repeat_kv(const Tensor& x, size_t n_rep, Tensor& y);
    
    // Convolution
    void causal_conv1d(const Tensor& x, const Tensor& weight, const Tensor* bias,
                       Tensor& y);
}
