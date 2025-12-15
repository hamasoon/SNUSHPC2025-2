#include "model_loader.h"
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>

ModelLoader::ModelLoader(const std::string& model_file)
    : model_file_(model_file), pinned_buffer_(nullptr), pinned_buffer_size_(0) {
    load_index();
}

ModelLoader::~ModelLoader() {
    // Free pinned buffer
    if (pinned_buffer_ != nullptr) {
        cudaFreeHost(pinned_buffer_);
        pinned_buffer_ = nullptr;
        pinned_buffer_size_ = 0;
    }
    // GPU cache tensors will be freed by their destructors
}

void ModelLoader::ensure_pinned_buffer(size_t size) {
    if (pinned_buffer_size_ < size) {
        if (pinned_buffer_ != nullptr) {
            cudaFreeHost(pinned_buffer_);
        }
        CHECK_CUDA(cudaMallocHost(&pinned_buffer_, size));
        pinned_buffer_size_ = size;
    }
}

void ModelLoader::load_index() {
    std::ifstream file(model_file_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_file_);
    }
    
    // Read number of tensors
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(uint32_t));
    
    std::cout << "Loading model index from " << model_file_ << std::endl;
    std::cout << "  Number of tensors: " << num_tensors << std::endl;
    
    // Read index for each tensor
    for (uint32_t i = 0; i < num_tensors; i++) {
        TensorInfo info;
        
        // Read name length and name
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(uint32_t));
        std::vector<char> name_buf(name_len);
        file.read(name_buf.data(), name_len);
        info.name = std::string(name_buf.begin(), name_buf.end());
        
        // Read number of dimensions
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(uint32_t));
        
        // Read shape
        info.shape.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
            info.shape[j] = dim;
        }
        
        // Read offset and size
        file.read(reinterpret_cast<char*>(&info.offset), sizeof(uint64_t));
        file.read(reinterpret_cast<char*>(&info.size), sizeof(uint64_t));
        
        index_[info.name] = info;
    }
    
    file.close();
    std::cout << "  Index loaded successfully" << std::endl;
}

Tensor ModelLoader::load_tensor(const std::string& name) {
    // Check if tensor is already cached on GPU
    auto cache_it = gpu_cache_.find(name);
    if (cache_it != gpu_cache_.end()) {
        // Return a view of the cached tensor (no copy, shares GPU memory)
        return cache_it->second.view(cache_it->second.shape());
    }

    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }

    const TensorInfo& info = it->second;

    // Create tensor with shape (allocates on GPU)
    Tensor tensor(info.shape);

    // Open file and seek to tensor data
    std::ifstream file(model_file_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_file_);
    }

    // Use pinned memory for faster H2D transfer
    ensure_pinned_buffer(info.size);

    file.seekg(info.offset);
    file.read(reinterpret_cast<char*>(pinned_buffer_), info.size);
    file.close();

    // Copy to GPU using pinned memory (faster than pageable memory)
    CHECK_CUDA(cudaMemcpy(tensor.data(), pinned_buffer_, info.size, cudaMemcpyHostToDevice));

    // Cache the tensor on GPU for future use
    gpu_cache_[name] = std::move(tensor);

    // Return a view of the cached tensor
    return gpu_cache_[name].view(gpu_cache_[name].shape());
}

bool ModelLoader::has_tensor(const std::string& name) const {
    return index_.find(name) != index_.end();
}

std::vector<size_t> ModelLoader::get_shape(const std::string& name) const {
    auto it = index_.find(name);
    if (it == index_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second.shape;
}

bool ModelLoader::is_cached(const std::string& name) const {
    return gpu_cache_.find(name) != gpu_cache_.end();
}

size_t ModelLoader::get_cached_memory_bytes() const {
    size_t total = 0;
    for (const auto& pair : gpu_cache_) {
        total += pair.second.size() * sizeof(float);
    }
    return total;
}

void ModelLoader::preload_all_tensors() {
    std::cout << "Preloading all tensors to GPU..." << std::endl;

    size_t total_size = 0;
    size_t max_tensor_size = 0;

    // Find the maximum tensor size for pinned buffer allocation
    for (const auto& pair : index_) {
        total_size += pair.second.size;
        if (pair.second.size > max_tensor_size) {
            max_tensor_size = pair.second.size;
        }
    }

    // Pre-allocate pinned buffer for the largest tensor
    ensure_pinned_buffer(max_tensor_size);

    std::cout << "  Total tensors: " << index_.size() << std::endl;
    std::cout << "  Total size: " << (total_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    // Open file once for all reads
    std::ifstream file(model_file_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_file_);
    }

    size_t loaded = 0;
    for (const auto& pair : index_) {
        const std::string& name = pair.first;
        const TensorInfo& info = pair.second;

        // Skip if already cached
        if (is_cached(name)) {
            loaded++;
            continue;
        }

        // Create tensor on GPU
        Tensor tensor(info.shape);

        // Read data using pinned buffer
        file.seekg(info.offset);
        file.read(reinterpret_cast<char*>(pinned_buffer_), info.size);

        // Copy to GPU
        CHECK_CUDA(cudaMemcpy(tensor.data(), pinned_buffer_, info.size, cudaMemcpyHostToDevice));

        // Cache the tensor
        gpu_cache_[name] = std::move(tensor);

        loaded++;
        if (loaded % 100 == 0) {
            std::cout << "  Loaded " << loaded << "/" << index_.size() << " tensors" << std::endl;
        }
    }

    file.close();

    std::cout << "  Preload complete. GPU memory used: "
              << (get_cached_memory_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
}
