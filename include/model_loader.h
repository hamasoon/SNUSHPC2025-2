#pragma once

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>

// Model loader for single binary file format
// Supports persistent GPU weight storage to avoid repeated H2D transfers
class ModelLoader {
public:
    ModelLoader(const std::string& model_file);
    ~ModelLoader();

    // Load a tensor by name (returns cached GPU tensor if available)
    Tensor load_tensor(const std::string& name);

    // Check if tensor exists
    bool has_tensor(const std::string& name) const;

    // Get tensor info without loading
    std::vector<size_t> get_shape(const std::string& name) const;

    // Preload all tensors to GPU memory for persistent storage
    // Uses async transfers with the provided stream for better performance
    void preload_all_tensors(cudaStream_t stream = 0);

    // Check if a tensor is already cached on GPU
    bool is_cached(const std::string& name) const;

    // Get memory usage statistics
    size_t get_cached_memory_bytes() const;

private:
    struct TensorInfo {
        std::string name;
        std::vector<size_t> shape;
        uint64_t offset;
        uint64_t size;
    };

    std::string model_file_;
    std::unordered_map<std::string, TensorInfo> index_;

    // Persistent GPU tensor cache - weights stay on GPU
    std::unordered_map<std::string, Tensor> gpu_cache_;

    // Double-buffered pinned host memory for async H2D transfers
    // Allows overlapping file I/O with GPU transfers
    float* pinned_buffer_[2];
    size_t pinned_buffer_size_;

    void load_index();
    void ensure_pinned_buffers(size_t size);
};
