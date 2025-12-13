#pragma once

#include "tensor.h"
#include "layer.h"
#include "config.h"
#include "model_loader.h"
#include <vector>
#include <memory>
#include <string>
#include <cuda_runtime.h>

// Global model loader (defined in model.cu)
extern std::unique_ptr<ModelLoader> g_model_loader;

// ============================================================================
// OPTIMIZATION 1.3: Memory Pool Allocator
// Pre-allocated GPU memory pool for intermediate buffers
// ============================================================================
class GPUMemoryPool {
public:
    static GPUMemoryPool& instance();

    void init(size_t pool_size);
    void* allocate(size_t size);
    void reset();  // Reset allocation offset (call between forward passes)
    void free_all();

    bool is_initialized() const { return initialized_; }

private:
    GPUMemoryPool() = default;
    ~GPUMemoryPool();
    GPUMemoryPool(const GPUMemoryPool&) = delete;
    GPUMemoryPool& operator=(const GPUMemoryPool&) = delete;

    float* pool_ = nullptr;
    size_t pool_size_ = 0;
    size_t current_offset_ = 0;
    bool initialized_ = false;
};

// ============================================================================
// OPTIMIZATION 1.1: Persistent GPU Weight Storage
// Holds all model weights on GPU for the entire model lifetime
// ============================================================================
struct PersistentLayerWeights {
    // Attention weights (for attention layers)
    float* q_proj = nullptr;
    float* k_proj = nullptr;
    float* v_proj = nullptr;
    float* o_proj = nullptr;
    float* q_ln_weight = nullptr;
    float* k_ln_weight = nullptr;

    // Conv weights (for conv layers)
    float* conv_weight = nullptr;
    float* in_proj_weight = nullptr;
    float* out_proj_weight = nullptr;
    float* conv_bias = nullptr;
    float* in_proj_bias = nullptr;
    float* out_proj_bias = nullptr;

    // Norm weights
    float* input_ln_weight = nullptr;
    float* post_attn_ln_weight = nullptr;

    // Dense MLP weights (for layers 0-1)
    float* mlp_w1 = nullptr;
    float* mlp_w2 = nullptr;
    float* mlp_w3 = nullptr;

    // MoE weights (for layers >= 2)
    float* gate = nullptr;
    float* expert_bias = nullptr;
    struct ExpertWeights {
        float* w1 = nullptr;
        float* w2 = nullptr;
        float* w3 = nullptr;
    };
    std::vector<ExpertWeights> experts;
};

class PersistentGPUWeights {
public:
    static PersistentGPUWeights& instance();

    void init(const Tensor& embed_tokens, const Tensor& lm_head,
              const std::vector<std::unique_ptr<DecoderLayer>>& layers,
              const RMSNorm& final_norm);

    bool is_initialized() const { return initialized_; }

    // Embeddings
    float* embed_tokens() const { return d_embed_tokens_; }
    float* lm_head() const { return d_lm_head_; }
    float* final_norm_weight() const { return d_final_norm_weight_; }

    // Per-layer weights
    const PersistentLayerWeights& layer(int idx) const { return layer_weights_[idx]; }

private:
    PersistentGPUWeights() = default;
    ~PersistentGPUWeights();
    PersistentGPUWeights(const PersistentGPUWeights&) = delete;
    PersistentGPUWeights& operator=(const PersistentGPUWeights&) = delete;

    float* d_embed_tokens_ = nullptr;
    float* d_lm_head_ = nullptr;
    float* d_final_norm_weight_ = nullptr;
    std::vector<PersistentLayerWeights> layer_weights_;
    bool initialized_ = false;
};

class LFM2Model {
public:
    LFM2Model(const std::string& model_file);
    ~LFM2Model();

    // Forward pass (single sequence)
    void forward(const std::vector<int>& input_ids, Tensor& logits);

    // Batched forward pass (multiple sequences) - OPTIMIZATION 3.2
    void forward_batch(const std::vector<std::vector<int>>& input_ids_batch,
                       std::vector<Tensor>& logits_batch);

    // OPTIMIZATION 3.3: Forward with CUDA graphs
    void forward_with_cuda_graph(const std::vector<int>& input_ids, Tensor& logits);

private:
    std::unique_ptr<ModelLoader> loader_;

    // Embeddings
    Tensor embed_tokens_;

    // Decoder layers
    std::vector<std::unique_ptr<DecoderLayer>> layers_;

    // Final norm
    std::unique_ptr<RMSNorm> norm_;

    // LM head (output projection)
    Tensor lm_head_;

    // RoPE
    std::unique_ptr<RotaryEmbedding> rotary_emb_;

    // OPTIMIZATION 1.1: Persistent GPU weights initialized
    bool gpu_weights_initialized_ = false;

    // OPTIMIZATION 3.3: CUDA graph for inference
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
    bool cuda_graph_captured_ = false;
    size_t captured_seq_len_ = 0;

    // Helper functions
    void load_embeddings();
    void load_layers();
    void load_output_layers();
    void init_persistent_gpu_weights();
    void capture_cuda_graph(size_t seq_len);
};
