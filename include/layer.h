#pragma once

#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>

// RMSNorm Layer
class RMSNorm {
public:
    RMSNorm(const std::string& weight_file);
    void forward(const Tensor& x, Tensor& y);
    
private:
    Tensor weight_;
};

// Rotary Position Embedding
class RotaryEmbedding {
public:
    RotaryEmbedding();
    void forward(size_t seq_len, Tensor& cos, Tensor& sin);
    
private:
    Tensor cos_cached_;
    Tensor sin_cached_;
    size_t max_seq_len_;
};

// MLP Layer (Feed-Forward Network)
class MLP {
public:
    MLP(const std::string& w1_file, const std::string& w2_file, const std::string& w3_file);
    void forward(const Tensor& x, Tensor& y);
    
private:
    Tensor w1_;  // up projection
    Tensor w3_;  // gate projection
    Tensor w2_;  // down projection
};

// Sparse MoE Block with Expert Parallelism
class SparseMoeBlock {
public:
    SparseMoeBlock(int layer_idx);
    void forward(const Tensor& x, Tensor& y, Tensor& router_logits);

private:
    Tensor gate_;  // router (replicated on all GPUs)
    std::vector<MLP> local_experts_;  // Only local experts for this GPU
    Tensor expert_bias_;  // optional (replicated)
    int layer_idx_;

    // Maps local expert index to global expert index
    std::vector<int> local_to_global_expert_;

    // Pre-allocated buffers to avoid cudaMalloc/cudaFree per call
    int* d_top_k_indices_;
    float* d_top_k_weights_;
    size_t routing_buffer_size_;  // max num_tokens supported

    // Pre-allocated buffers for gather/scatter operations
    int* d_gather_indices_;      // GPU buffer for token indices per expert
    float* d_scatter_weights_;   // GPU buffer for scatter weights per expert
    size_t gather_scatter_buffer_size_;

    // Buffers for fully GPU-based routing
    int* d_expert_counts_;       // Count per local expert
    int* d_expert_offsets_;      // Prefix sum offsets per expert
    int* d_expert_write_pos_;    // Current write position per expert (for building indices)
    int* d_sorted_indices_;      // Token indices sorted by expert
    float* d_sorted_weights_;    // Weights sorted by expert

    void route_tokens_gpu(const Tensor& router_logits, size_t num_tokens);
    void ensure_routing_buffers(size_t num_tokens);
    void ensure_gather_scatter_buffers(size_t max_tokens_per_expert);
};

// Multi-Head Attention
class Attention {
public:
    Attention(int layer_idx);
    void forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor* attention_mask, Tensor& output);
    
private:
    Tensor q_proj_;
    Tensor k_proj_;
    Tensor v_proj_;
    Tensor o_proj_;
    std::unique_ptr<RMSNorm> q_layernorm_;
    std::unique_ptr<RMSNorm> k_layernorm_;
    int layer_idx_;
};

// Short Convolution (Mamba-style)
class ShortConv {
public:
    ShortConv(int layer_idx);
    void forward(const Tensor& x, Tensor& y);
    
private:
    Tensor conv_weight_;
    Tensor conv_bias_;
    Tensor in_proj_weight_;
    Tensor in_proj_bias_;
    Tensor out_proj_weight_;
    Tensor out_proj_bias_;
    int layer_idx_;
};

// Decoder Layer
class DecoderLayer {
public:
    DecoderLayer(int layer_idx, bool is_attention_layer);
    void forward(const Tensor& x, const Tensor& cos, const Tensor& sin,
                 const Tensor* attention_mask, Tensor& output);
    
    bool is_attention_layer() const { return is_attention_layer_; }
    
private:
    int layer_idx_;
    bool is_attention_layer_;
    
    // Components
    std::unique_ptr<RMSNorm> input_layernorm_;
    std::unique_ptr<RMSNorm> post_attention_layernorm_;
    
    // Either attention or conv
    std::unique_ptr<Attention> self_attn_;
    std::unique_ptr<ShortConv> short_conv_;
    
    // Either MoE block (layers >= 2) or dense MLP (layers 0-1)
    std::unique_ptr<SparseMoeBlock> moe_block_;
    std::unique_ptr<MLP> dense_mlp_;
};
