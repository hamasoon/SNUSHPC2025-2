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

    // Getter for weight (needed for GPU kernels)
    const Tensor& weight() const { return weight_; }
    Tensor& weight() { return weight_; }

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
    ~MLP();

    // Delete copy (GPU pointers would be double-freed)
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;

    // Move constructor
    MLP(MLP&& other) noexcept;
    MLP& operator=(MLP&& other) noexcept;

    void forward(const Tensor& x, Tensor& y);

    // Getters for weights (needed for batched multi-GPU MoE processing)
    const Tensor& w1() const { return w1_; }
    const Tensor& w2() const { return w2_; }
    const Tensor& w3() const { return w3_; }

private:
    Tensor w1_;  // up projection
    Tensor w3_;  // gate projection
    Tensor w2_;  // down projection

    // Persistent GPU weights
    float* d_w1_ = nullptr;
    float* d_w2_ = nullptr;
    float* d_w3_ = nullptr;
};

// Sparse MoE Block
class SparseMoeBlock {
public:
    SparseMoeBlock(int layer_idx);
    void forward(const Tensor& x, Tensor& y, Tensor& router_logits);
    
private:
    Tensor gate_;  // router
    std::vector<MLP> experts_;
    Tensor expert_bias_;  // optional
    
    void route_tokens(const Tensor& router_logits, std::vector<int>& top_k_indices,
                     std::vector<float>& top_k_weights);
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
