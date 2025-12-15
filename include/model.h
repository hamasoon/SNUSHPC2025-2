#pragma once

#include "tensor.h"
#include "layer.h"
#include "config.h"
#include "model_loader.h"
#include <vector>
#include <memory>
#include <string>
#include <mpi.h>

// Global model loader (defined in model.cu)
extern std::unique_ptr<ModelLoader> g_model_loader;

// Parallelism context
struct ParallelContext {
    int world_rank;       // Global MPI rank
    int world_size;       // Total MPI processes
    int local_rank;       // Rank within node (0-3 for 4 GPUs)
    int node_rank;        // Which node this process is on
    int num_nodes;        // Total number of nodes
    MPI_Comm node_comm;   // Communicator for processes within same node

    // Expert assignment
    int expert_start;     // First expert index for this GPU
    int expert_end;       // Last expert index (exclusive) for this GPU

    // CUDA streams for async operations
    cudaStream_t compute_stream;    // Main compute stream
    cudaStream_t h2d_stream;        // Host to device transfer stream
    cudaStream_t d2h_stream;        // Device to host transfer stream

    void init(int rank, int size);
    void finalize();
    bool is_expert_local(int expert_idx) const {
        return expert_idx >= expert_start && expert_idx < expert_end;
    }
    int expert_to_rank(int expert_idx) const {
        return (expert_idx / NUM_EXPERTS_PER_GPU) % NUM_GPUS_PER_NODE;
    }
};

extern ParallelContext g_parallel_ctx;

class LFM2Model {
public:
    LFM2Model(const std::string& model_file);

    // Forward pass (single sample)
    void forward(const std::vector<int>& input_ids, Tensor& logits);

    // Batched forward pass (multiple samples)
    // input_ids: flattened array of [batch_size * seq_len]
    // logits: output tensor of shape [batch_size, vocab_size]
    void forward_batch(const int* input_ids, size_t batch_size, size_t seq_len, Tensor& logits);
    
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
    
    // Helper functions
    void load_embeddings();
    void load_layers();
    void load_output_layers();
};
