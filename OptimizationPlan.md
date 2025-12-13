Model Optimization Plan for LFM2-8B-A1B
Based on my thorough analysis of the codebase, here is a comprehensive optimization plan:
Codebase Summary
Component	File	Purpose
Config	config.h	Model hyperparameters (24 layers, 32 experts, 65K vocab)
Tensors	tensor.h, tensor.cu	N-dimensional array operations
Layers	layer.h, layer.cu	CUDA kernels + CPU tensor operations
Model	model.h, model.cu	Forward pass, MoE routing, layer orchestration
Entry Point	main.cpp	MPI initialization, sample processing loop
Priority 1: Critical Optimizations (Expected 3-5× speedup)
1.1 MoE Expert Batching — model.cu:745-798
Problem: Experts process tokens individually in sequential loop
Solution:
Batch tokens destined for same expert before GPU launch
Use parallel expert execution across multiple GPUs
Implement token-to-expert assignment matrix for batch processing
1.2 GPU Attention Kernels — model.cu:963-1000
Problem: Attention computed on CPU despite CUDA kernels existing in layer.cu:252-350
Solution:
Replace CPU attention loops with GPU kernel calls
Use attention_score_kernel, causal_softmax_kernel, attention_output_kernel
Keep Q, K, V tensors on GPU throughout attention computation

Priority 2: High-Impact Optimizations (Expected 2-3× speedup)
2.1 ShortConv Kernel Fusion
Problem: 8 separate operations with CPU tensor manipulations
Solution:
Create fused CUDA kernel combining: in_proj → reshape → split → multiply → conv1d → multiply → transpose → out_proj
Eliminate intermediate CPU-side tensor operations
2.2 Persistent GPU Buffers — model.cu:656-720
Problem: Repeated cudaMalloc/cudaFree per operation
Solution:
Pre-allocate workspace buffers for all intermediate tensors
Extend GPUContext::workspace to handle all layer computations
Use workspace pooling with offset-based allocation
2.3 Embedding Lookup GPU Kernel — model.cu:1269-1275
Problem: CPU nested loops for embedding lookup
Solution:
__global__ void embedding_lookup_kernel(
    const int* tokens, const float* embed_table, 
    float* output, int seq_len, int hidden_size);

Priority 3: Medium-Impact Optimizations (Expected 1.5-2× speedup)
3.1 RMSNorm GPU Implementation — Currently CPU fallback
Problem: 48 normalization calls per forward pass on CPU
Solution: Use existing rms_norm_kernel from layer.cu:65-115
3.2 Sample Batching — main.cpp:223
Problem: Samples processed sequentially per MPI rank
Solution:
Implement batched forward pass supporting multiple sequences
Pad sequences to common length with attention masking
3.3 Memory Transfer Optimization
Problem: Synchronous H2D/D2H transfers blocking GPU
Solution:
Use CUDA streams for async transfers
Overlap computation with data transfer (double-buffering)
Use pinned memory consistently (cudaMallocHost)

Priority 4: Lower-Impact Optimizations
4.1 MPI Weight Loading — main.cpp
Single rank loads model, broadcast to others (reduces I/O contention)
4.2 Output Reduction
Gather only non-zero outputs instead of full vocab logits
4.3 Multi-GPU Expert Distribution
Currently: 8 experts per GPU (32 ÷ 4)
Optimize: Load balance based on expert utilization patterns
Implementation Roadmap
Phase	Optimizations	Target Files
Phase 1	GPU Attention + RMSNorm	model.cu, layer.cu
Phase 2	MoE Expert Batching	model.cu
Phase 3	ShortConv Fusion + Embedding Kernel	model.cu, layer.cu
Phase 4	Memory Management + Async Transfers	model.cu, tensor.cu
Phase 5	Sample Batching + MPI Optimization	main.cpp, model.cu
Key Metrics to Track
Tokens/second: Primary throughput metric
GPU utilization: Target >80% via nvidia-smi
Memory bandwidth: Monitor PCIe transfer bottlenecks
MPI scaling efficiency: Measure speedup vs. node count
Architecture Diagram
┌─────────────────────────────────────────────────────────────┐
│                    MPI Data Parallelism                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Node 0  │  │ Node 1  │  │ Node 2  │  │ Node 3  │        │
│  │ GPU×4   │  │ GPU×4   │  │ GPU×4   │  │ GPU×4   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                         │                                    │
│                  MPI_Reduce(outputs)                         │
└─────────────────────────────────────────────────────────────┘

Per-Node GPU Layout (Model Parallelism for MoE):
┌─────────────────────────────────────────────────────────────┐
│  GPU 0: Experts 0-7   │  GPU 1: Experts 8-15                │
│  GPU 2: Experts 16-23 │  GPU 3: Experts 24-31               │
│                                                              │
│  All GPUs: Shared Attention + Dense Layers                  │
└─────────────────────────────────────────────────────────────┘