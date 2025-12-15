#include "tensor.h"
#include <cstring>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "model_loader.h"

// Global model loader is declared in model.h
extern std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// GPU Memory Pool Implementation
// ============================================================================

static constexpr size_t MIN_BUCKET_SIZE = 256;  // 1 KB minimum

GPUMemoryPool& GPUMemoryPool::instance() {
    static GPUMemoryPool pool;
    return pool;
}

GPUMemoryPool::GPUMemoryPool() {}

GPUMemoryPool::~GPUMemoryPool() {
    clear();
}

size_t GPUMemoryPool::bucket_size(size_t num_elements) const {
    if (num_elements == 0) return 0;
    if (num_elements < MIN_BUCKET_SIZE) return MIN_BUCKET_SIZE;

    // Round up to next power of 2 for sizes up to 1M elements
    if (num_elements <= 1024 * 1024) {
        size_t bucket = MIN_BUCKET_SIZE;
        while (bucket < num_elements) bucket *= 2;
        return bucket;
    }
    // For larger sizes, round up to nearest 1M elements
    size_t million = 1024 * 1024;
    return ((num_elements + million - 1) / million) * million;
}

float* GPUMemoryPool::allocate(size_t num_elements) {
    if (num_elements == 0) return nullptr;

    size_t bucket = bucket_size(num_elements);
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = free_buffers_.find(bucket);
    if (it != free_buffers_.end() && !it->second.empty()) {
        float* ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    float* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bucket * sizeof(float));
    if (err != cudaSuccess) {
        // Try to free pooled memory and retry
        for (auto& pair : free_buffers_) {
            for (float* buf : pair.second) cudaFree(buf);
            pair.second.clear();
        }
        err = cudaMalloc(&ptr, bucket * sizeof(float));
        if (err != cudaSuccess) return nullptr;
    }
    return ptr;
}

void GPUMemoryPool::deallocate(float* ptr, size_t num_elements) {
    if (ptr == nullptr || num_elements == 0) return;
    size_t bucket = bucket_size(num_elements);
    std::lock_guard<std::mutex> lock(mutex_);
    free_buffers_[bucket].push_back(ptr);
}

void GPUMemoryPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : free_buffers_) {
        for (float* ptr : pair.second) cudaFree(ptr);
        pair.second.clear();
    }
    free_buffers_.clear();
}

// Tensor class implementation with GPU support

// Tensor constructors and destructors
Tensor::Tensor() : size_(0), d_data_(nullptr), h_data_(nullptr), owns_data_(false),
                   on_gpu_(true), host_valid_(false), device_valid_(false), host_pinned_(false) {}

Tensor::Tensor(const std::vector<size_t>& shape, bool on_gpu)
    : shape_(shape), owns_data_(true), on_gpu_(on_gpu), host_valid_(false), device_valid_(true), host_pinned_(false) {
    size_ = compute_size();
    h_data_ = nullptr;
    allocate();
}

Tensor::Tensor(const std::vector<size_t>& shape, float* data, bool copy, bool on_gpu)
    : shape_(shape), owns_data_(copy), on_gpu_(on_gpu), host_pinned_(false) {
    size_ = compute_size();
    h_data_ = nullptr;
    if (copy) {
        allocate();
        if (on_gpu) {
            // data is on CPU, copy to GPU
            CHECK_CUDA(cudaMemcpy(d_data_, data, size_ * sizeof(float), cudaMemcpyHostToDevice));
            device_valid_ = true;
            host_valid_ = false;
        } else {
            // This shouldn't happen in our design, but handle it
            h_data_ = new float[size_];
            std::memcpy(h_data_, data, size_ * sizeof(float));
            CHECK_CUDA(cudaMemcpy(d_data_, data, size_ * sizeof(float), cudaMemcpyHostToDevice));
            device_valid_ = true;
            host_valid_ = true;
        }
    } else {
        // View - share data (assumes data is on GPU if on_gpu is true)
        if (on_gpu) {
            d_data_ = data;
            device_valid_ = true;
            host_valid_ = false;
        } else {
            h_data_ = data;
            d_data_ = nullptr;
            host_valid_ = true;
            device_valid_ = false;
        }
    }
}

Tensor::~Tensor() {
    deallocate();
}

// Copy constructor
Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), size_(other.size_), owns_data_(true),
      on_gpu_(other.on_gpu_), host_valid_(false), device_valid_(true), host_pinned_(false) {
    h_data_ = nullptr;
    if (other.size_ > 0) {
        allocate();
        if (other.device_valid_) {
            CHECK_CUDA(cudaMemcpy(d_data_, other.d_data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
        } else if (other.host_valid_) {
            CHECK_CUDA(cudaMemcpy(d_data_, other.h_data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
}

// Copy assignment
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        deallocate();
        shape_ = other.shape_;
        size_ = other.size_;
        owns_data_ = true;
        on_gpu_ = other.on_gpu_;
        host_valid_ = false;
        device_valid_ = true;
        host_pinned_ = false;
        h_data_ = nullptr;
        if (other.size_ > 0) {
            allocate();
            if (other.device_valid_) {
                CHECK_CUDA(cudaMemcpy(d_data_, other.d_data_, size_ * sizeof(float), cudaMemcpyDeviceToDevice));
            } else if (other.host_valid_) {
                CHECK_CUDA(cudaMemcpy(d_data_, other.h_data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }
    return *this;
}

// Move constructor
Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), size_(other.size_),
      d_data_(other.d_data_), h_data_(other.h_data_), owns_data_(other.owns_data_),
      on_gpu_(other.on_gpu_), host_valid_(other.host_valid_), device_valid_(other.device_valid_),
      host_pinned_(other.host_pinned_) {
    other.d_data_ = nullptr;
    other.h_data_ = nullptr;
    other.size_ = 0;
    other.owns_data_ = false;
    other.host_pinned_ = false;
}

// Move assignment
Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        deallocate();
        shape_ = std::move(other.shape_);
        size_ = other.size_;
        d_data_ = other.d_data_;
        h_data_ = other.h_data_;
        owns_data_ = other.owns_data_;
        on_gpu_ = other.on_gpu_;
        host_valid_ = other.host_valid_;
        device_valid_ = other.device_valid_;
        host_pinned_ = other.host_pinned_;

        other.d_data_ = nullptr;
        other.h_data_ = nullptr;
        other.size_ = 0;
        other.owns_data_ = false;
        other.host_pinned_ = false;
    }
    return *this;
}

void Tensor::allocate() {
    if (size_ > 0) {
        // Use memory pool instead of direct cudaMalloc
        d_data_ = gpu_memory_pool().allocate(size_);
        if (d_data_ == nullptr) {
            throw std::runtime_error("GPU memory allocation failed");
        }
    }
}

void Tensor::deallocate() {
    if (owns_data_) {
        if (d_data_ != nullptr) {
            // Return to memory pool instead of cudaFree
            gpu_memory_pool().deallocate(d_data_, size_);
            d_data_ = nullptr;
        }
        if (h_data_ != nullptr) {
            if (host_pinned_) {
                cudaFreeHost(h_data_);
            } else {
                delete[] h_data_;
            }
            h_data_ = nullptr;
            host_pinned_ = false;
        }
    }
}

size_t Tensor::compute_size() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
}

size_t Tensor::size(int dim) const {
    if (dim < 0) dim += shape_.size();
    if (dim < 0 || static_cast<size_t>(dim) >= shape_.size()) {
        throw std::out_of_range("Dimension out of range");
    }
    return shape_[dim];
}

size_t Tensor::compute_stride(int dim) const {
    size_t stride = 1;
    for (size_t i = dim + 1; i < shape_.size(); i++) {
        stride *= shape_[i];
    }
    return stride;
}

// Ensure host data is allocated and valid
void Tensor::ensure_host_data() const {
    if (h_data_ == nullptr && size_ > 0) {
        h_data_ = new float[size_];
    }
    if (!host_valid_ && device_valid_ && size_ > 0) {
        CHECK_CUDA(cudaMemcpy(h_data_, d_data_, size_ * sizeof(float), cudaMemcpyDeviceToHost));
        host_valid_ = true;
    }
}

// Host data access
float* Tensor::host_data() {
    ensure_host_data();
    device_valid_ = false; // Mark device as stale since host may be modified
    return h_data_;
}

const float* Tensor::host_data() const {
    ensure_host_data();
    return h_data_;
}

// Sync operations
void Tensor::sync_to_host() const {
    ensure_host_data();
}

void Tensor::sync_to_device() {
    if (!device_valid_ && host_valid_ && size_ > 0) {
        CHECK_CUDA(cudaMemcpy(d_data_, h_data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
        device_valid_ = true;
    }
}

// Async sync operations with CUDA streams
void Tensor::sync_to_host_async(cudaStream_t stream) const {
    if (h_data_ == nullptr && size_ > 0) {
        // Must use pinned memory for async transfers
        CHECK_CUDA(cudaMallocHost(&h_data_, size_ * sizeof(float)));
        host_pinned_ = true;
    }
    if (!host_valid_ && device_valid_ && size_ > 0) {
        CHECK_CUDA(cudaMemcpyAsync(h_data_, d_data_, size_ * sizeof(float),
                                    cudaMemcpyDeviceToHost, stream));
        // Note: host_valid_ will be set after stream synchronization
    }
}

void Tensor::sync_to_device_async(cudaStream_t stream) {
    if (!device_valid_ && host_valid_ && size_ > 0) {
        CHECK_CUDA(cudaMemcpyAsync(d_data_, h_data_, size_ * sizeof(float),
                                    cudaMemcpyHostToDevice, stream));
        // Note: device_valid_ will be set after stream synchronization
    }
}

void Tensor::to_gpu() {
    if (!device_valid_) {
        sync_to_device();
    }
    on_gpu_ = true;
}

void Tensor::to_cpu() {
    sync_to_host();
    on_gpu_ = false;
}

// Element access (requires sync to host)
float& Tensor::operator[](size_t idx) {
    ensure_host_data();
    device_valid_ = false;
    return h_data_[idx];
}

const float& Tensor::operator[](size_t idx) const {
    ensure_host_data();
    return h_data_[idx];
}

float& Tensor::at(size_t i) {
    ensure_host_data();
    device_valid_ = false;
    return h_data_[i];
}

float& Tensor::at(size_t i, size_t j) {
    ensure_host_data();
    device_valid_ = false;
    return h_data_[i * shape_[1] + j];
}

float& Tensor::at(size_t i, size_t j, size_t k) {
    ensure_host_data();
    device_valid_ = false;
    return h_data_[(i * shape_[1] + j) * shape_[2] + k];
}

float& Tensor::at(size_t i, size_t j, size_t k, size_t l) {
    ensure_host_data();
    device_valid_ = false;
    return h_data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

const float& Tensor::at(size_t i) const {
    ensure_host_data();
    return h_data_[i];
}

const float& Tensor::at(size_t i, size_t j) const {
    ensure_host_data();
    return h_data_[i * shape_[1] + j];
}

const float& Tensor::at(size_t i, size_t j, size_t k) const {
    ensure_host_data();
    return h_data_[(i * shape_[1] + j) * shape_[2] + k];
}

const float& Tensor::at(size_t i, size_t j, size_t k, size_t l) const {
    ensure_host_data();
    return h_data_[((i * shape_[1] + j) * shape_[2] + k) * shape_[3] + l];
}

// Reshape
void Tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same number of elements");
    }
    shape_ = new_shape;
}

Tensor Tensor::view(const std::vector<size_t>& new_shape) const {
    // Verify new shape has same number of elements
    size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(), 1ULL, std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("New shape must have same number of elements");
    }

    // Create a view that shares GPU data with this tensor (no copy)
    Tensor result;
    result.shape_ = new_shape;
    result.size_ = size_;
    result.d_data_ = d_data_;
    result.h_data_ = nullptr;
    result.owns_data_ = false;
    result.on_gpu_ = on_gpu_;
    result.host_valid_ = false;
    result.device_valid_ = device_valid_;
    return result;
}

// IO operations
Tensor Tensor::load_from_file(const std::string& filename, ModelLoader* loader) {
    // If a specific loader is provided, use it
    if (loader) {
        return loader->load_tensor(filename);
    }

    // Otherwise, if global model loader is available, use it
    if (g_model_loader) {
        return g_model_loader->load_tensor(filename);
    }

    // Fallback to individual file loading (if model.bin not used)
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Read number of dimensions
    uint32_t ndim;
    file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));

    // Read shape
    std::vector<size_t> shape(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
        shape[i] = dim;
    }

    // Create tensor
    Tensor tensor(shape);

    // Read data to host first, then copy to device
    size_t total_size = tensor.size();
    float* host_buffer = new float[total_size];
    file.read(reinterpret_cast<char*>(host_buffer), total_size * sizeof(float));
    CHECK_CUDA(cudaMemcpy(tensor.data(), host_buffer, total_size * sizeof(float), cudaMemcpyHostToDevice));
    delete[] host_buffer;

    file.close();
    return tensor;
}

void Tensor::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    // Write number of dimensions
    uint32_t ndim = shape_.size();
    file.write(reinterpret_cast<const char*>(&ndim), sizeof(uint32_t));

    // Write shape
    for (size_t dim : shape_) {
        uint32_t dim32 = dim;
        file.write(reinterpret_cast<const char*>(&dim32), sizeof(uint32_t));
    }

    // Sync to host and write data
    sync_to_host();
    file.write(reinterpret_cast<const char*>(h_data_), size_ * sizeof(float));

    file.close();
}

// Tensor operations
Tensor Tensor::copy() const {
    return Tensor(*this);
}

void Tensor::fill(float value) {
    if (size_ > 0) {
        // Allocate host buffer if needed
        if (h_data_ == nullptr) {
            h_data_ = new float[size_];
        }
        std::fill(h_data_, h_data_ + size_, value);
        CHECK_CUDA(cudaMemcpy(d_data_, h_data_, size_ * sizeof(float), cudaMemcpyHostToDevice));
        host_valid_ = true;
        device_valid_ = true;
    }
}

void Tensor::zero() {
    if (d_data_ != nullptr && size_ > 0) {
        CHECK_CUDA(cudaMemset(d_data_, 0, size_ * sizeof(float)));
        device_valid_ = true;
        host_valid_ = false;
    }
}

void Tensor::ones() {
    fill(1.0f);
}
