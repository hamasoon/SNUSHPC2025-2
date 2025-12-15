#include "model_loader.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

ModelLoader::ModelLoader(const std::string& model_file)
    : model_file_(model_file), pinned_buffer_size_(0) {
    pinned_buffer_[0] = nullptr;
    pinned_buffer_[1] = nullptr;
    load_index();
}

ModelLoader::~ModelLoader() {
    // Free pinned buffers (double-buffered)
    for (int i = 0; i < 2; i++) {
        if (pinned_buffer_[i] != nullptr) {
            cudaFreeHost(pinned_buffer_[i]);
            pinned_buffer_[i] = nullptr;
        }
    }
    pinned_buffer_size_ = 0;
    // GPU cache tensors will be freed by their destructors
}

void ModelLoader::ensure_pinned_buffers(size_t size) {
    if (pinned_buffer_size_ < size) {
        // Free old buffers
        for (int i = 0; i < 2; i++) {
            if (pinned_buffer_[i] != nullptr) {
                cudaFreeHost(pinned_buffer_[i]);
            }
            CHECK_CUDA(cudaMallocHost(&pinned_buffer_[i], size));
        }
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
    ensure_pinned_buffers(info.size);

    file.seekg(info.offset);
    file.read(reinterpret_cast<char*>(pinned_buffer_[0]), info.size);
    file.close();

    // Copy to GPU using pinned memory (faster than pageable memory)
    CHECK_CUDA(cudaMemcpy(tensor.data(), pinned_buffer_[0], info.size, cudaMemcpyHostToDevice));

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

void ModelLoader::preload_all_tensors(cudaStream_t stream) {
    std::cout << "Preloading all tensors to GPU with async transfers..." << std::endl;

    size_t total_size = 0;
    size_t max_tensor_size = 0;

    // Collect tensors to load (sorted by offset for sequential file access)
    std::vector<std::pair<std::string, TensorInfo>> tensors_to_load;
    for (const auto& pair : index_) {
        if (!is_cached(pair.first)) {
            tensors_to_load.push_back(pair);
            total_size += pair.second.size;
            if (pair.second.size > max_tensor_size) {
                max_tensor_size = pair.second.size;
            }
        }
    }

    // Sort by file offset for sequential access
    std::sort(tensors_to_load.begin(), tensors_to_load.end(),
              [](const auto& a, const auto& b) { return a.second.offset < b.second.offset; });

    // Pre-allocate double pinned buffers for the largest tensor
    ensure_pinned_buffers(max_tensor_size);

    std::cout << "  Tensors to load: " << tensors_to_load.size() << std::endl;
    std::cout << "  Total size: " << (total_size / (1024.0 * 1024.0)) << " MB" << std::endl;

    // Open file once for all reads
    std::ifstream file(model_file_, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open model file: " + model_file_);
    }

    // Double-buffered async loading:
    // While GPU transfers buffer[i], CPU reads into buffer[1-i]
    size_t loaded = 0;
    int current_buffer = 0;
    Tensor* pending_tensor = nullptr;
    std::string pending_name;

    for (size_t i = 0; i <= tensors_to_load.size(); i++) {
        // Wait for previous async transfer to complete (if any)
        if (pending_tensor != nullptr) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
            // Cache the tensor
            gpu_cache_[pending_name] = std::move(*pending_tensor);
            delete pending_tensor;
            pending_tensor = nullptr;
            loaded++;

            if (loaded % 100 == 0) {
                std::cout << "  Loaded " << loaded << "/" << tensors_to_load.size() << " tensors" << std::endl;
            }
        }

        // Start next tensor load (if there are more)
        if (i < tensors_to_load.size()) {
            const std::string& name = tensors_to_load[i].first;
            const TensorInfo& info = tensors_to_load[i].second;

            // Create tensor on GPU
            pending_tensor = new Tensor(info.shape);
            pending_name = name;

            // Read data into current pinned buffer
            file.seekg(info.offset);
            file.read(reinterpret_cast<char*>(pinned_buffer_[current_buffer]), info.size);

            // Start async copy to GPU
            CHECK_CUDA(cudaMemcpyAsync(pending_tensor->data(), pinned_buffer_[current_buffer],
                                        info.size, cudaMemcpyHostToDevice, stream));

            // Switch buffer for next iteration
            current_buffer = 1 - current_buffer;
        }
    }

    file.close();

    // Final synchronization
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::cout << "  Preload complete. GPU memory used: "
              << (get_cached_memory_bytes() / (1024.0 * 1024.0)) << " MB" << std::endl;
}
