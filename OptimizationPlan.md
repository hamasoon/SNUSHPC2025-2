# LFM2-8B-A1B Model Optimization Plan

## Executive Summary

This document outlines a comprehensive optimization strategy for the LFM2-8B-A1B inference implementation. The model is a 8B parameter Mixture of Experts (MoE) transformer with 24 decoder layers, featuring hybrid attention/convolution layers and sparse expert routing.

---

## Current Architecture Analysis

### Model Configuration
- **Vocabulary Size**: 65,536 tokens
- **Hidden Size**: 2,048
- **Intermediate Size**: 7,168 (dense), 1,792 (MoE)
- **Layers**: 24 total (2 dense + 22 MoE layers)
- **Attention Heads**: 32 (Q) / 8 (KV) - Grouped Query Attention
- **Experts**: 32 experts, 4 activated per token
- **Layer Pattern**: 6 attention layers (2,6,10,14,18,21), 18 convolution layers

### Current Optimizations Already Implemented
1. **MoE Expert Batching** (model.cu:887-1009) - Batched multi-GPU expert processing
2. **GPU Attention Kernels** (model.cu:1110-1178) - Tiled attention computation
3. **Fused ShortConv Kernels** (model.cu:231-328) - Kernel fusion for conv layers
4. **Persistent GPU Buffers** (model.cu:356-436) - Memory reuse across calls
5. **GPU Embedding Lookup** (model.cu:334-350) - Parallel token embedding
6. **Async Stream Manager** (model.cu:445-513) - Multi-stream support
7. **GPU RMSNorm** (layer.cu:623-658) - Parallel normalization
8. **Tiled MatMul** (layer.cu:19-63) - 32x32 tile-based matrix multiplication

---

## Optimization Categories

### Category 1: Memory Bandwidth Optimizations

#### 1.1 Persistent Weight Storage on GPU
**Current Issue**: Weights are uploaded to GPU per forward pass (model.cu:746-749, 948-951)
**Solution**: Pre-upload all model weights to GPU during initialization
**Expected Speedup**: 2-3x reduction in data transfer overhead
**Implementation**:
- Extend `GPUTensor` class to hold all layer weights
- Upload during `LFM2Model` constructor
- Eliminate `cudaMemcpy` calls in forward passes

#### 1.2 Unified Memory with Prefetching
**Current Issue**: Explicit H2D/D2H copies create synchronization points
**Solution**: Use `cudaMallocManaged` with explicit prefetch hints
**Expected Speedup**: 10-20% reduction in memory stalls
**Files**: model.cu, layer.cu

#### 1.3 Memory Pool Allocator
**Current Issue**: `cudaMalloc`/`cudaFree` called per layer (model.cu:735-743)
**Solution**: Pre-allocate large memory pool, use offset-based allocation
**Expected Speedup**: Eliminate allocation overhead (~5-10%)

---

### Category 2: Compute Optimizations

#### 2.1 Flash Attention Implementation
**Current Issue**: Naive attention with O(N�) memory (model.cu:1119-1178)
**Solution**: Implement Flash Attention with tiling and online softmax
**Expected Speedup**: 2-4x with reduced memory footprint
**Files**: model.cu attention kernels

#### 2.2 Tensor Core Utilization (FP16/TF32)
**Current Issue**: All computations in FP32
**Solution**: Use mixed-precision with FP16 storage, TF32 compute
**Expected Speedup**: 2x on Ampere/Ada GPUs
**Requires**: Add FP16 weight conversion, accumulation in FP32

#### 2.3 Fused Multi-Head Attention Kernel
**Current Issue**: Separate kernels for scores, softmax, output (model.cu:1157-1166)
**Solution**: Single fused kernel with shared memory QKV
**Expected Speedup**: 30-50% reduction in kernel launch overhead

---

### Category 3: Parallelism Optimizations

#### 3.1 Multi-GPU Pipeline Parallelism
**Current Issue**: Experts distributed but layers processed sequentially
**Solution**: Pipeline layers across GPUs with micro-batching
**Expected Speedup**: Near-linear scaling with GPU count
**Files**: model.cu `forward()`, `DecoderLayer::forward()`

#### 3.2 Improved MPI Data Distribution
**Current Issue**: Simple sample-based distribution (main.cpp:217-221)
**Solution**: Dynamic load balancing with work stealing
**Expected Speedup**: Better utilization with variable sequence lengths

#### 3.3 CUDA Graph Capture
**Current Issue**: Kernel launch overhead per layer
**Solution**: Capture forward pass as CUDA graph for single-shot execution
**Expected Speedup**: 10-30% reduction in CPU overhead
**Applicable to**: Entire decoder layer forward passes

#### 3.4 Asynchronous Expert Computation
**Current Issue**: Experts processed with `cudaMemcpy` synchronization
**Solution**: Use streams per expert with event-based synchronization
**Expected Speedup**: Overlap compute with memory transfers

---

### Category 4: Algorithm Optimizations

#### 4.1 KV Cache Implementation
**Current Issue**: No KV cache - recomputes full sequence each token
**Solution**: Store and reuse K,V projections for autoregressive generation
**Expected Speedup**: O(N) per token instead of O(N�) cumulative
**Critical for**: Multi-token generation scenarios

#### 4.2 Speculative Decoding
**Current Issue**: Single token generation per forward pass
**Solution**: Draft model predicts multiple tokens, main model verifies
**Expected Speedup**: 2-3x token throughput
**Requires**: Smaller draft model or early-exit mechanism

#### 4.3 Expert Load Balancing
**Current Issue**: Static round-robin GPU assignment (model.cu:919)
**Solution**: Dynamic assignment based on expert activation frequency
**Expected Speedup**: Better GPU utilization with skewed distributions

#### 4.4 Sparse Attention Patterns
**Current Issue**: Full causal attention for all positions
**Solution**: Sliding window + global tokens for long sequences
**Expected Speedup**: O(N) instead of O(N�) for long contexts

---

### Category 5: I/O and Data Loading Optimizations

#### 5.1 Memory-Mapped Model Loading
**Current Issue**: Sequential file reads (model_loader.cpp:58-81)
**Solution**: `mmap()` model file for on-demand page loading
**Expected Speedup**: Faster startup, lazy weight loading

#### 5.2 Concurrent Input Processing
**Current Issue**: Single-threaded input reading (main.cpp:141-154)
**Solution**: Parallel input batching with pinned memory staging
**Expected Speedup**: Hide input loading latency

#### 5.3 Output Streaming
**Current Issue**: All outputs computed before any returned
**Solution**: Stream outputs as generated for real-time applications
**User Experience**: Lower perceived latency

---

## Implementation Priority Matrix

| Priority | Optimization | Effort | Impact | Dependencies |
|----------|-------------|--------|--------|--------------|
| **P0** | Persistent Weights (1.1) | Low | High | None |
| **P1** | Flash Attention (2.1) | High | High | None |
| **P1** | CUDA Graphs (3.3) | Medium | Medium | 1.1 |
| **P1** | Memory Pool (1.3) | Medium | Medium | None |
| **P2** | Tensor Cores (2.2) | High | High | 2.1 |
| **P2** | KV Cache (4.1) | High | Very High | 2.2 |
| **P2** | Pipeline Parallel (3.1) | High | High | 1.1 |
| **P3** | Speculative Decoding (4.2) | Very High | High | 4.1 |
| **P3** | Sparse Attention (4.4) | Medium | Medium | 2.2 |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (Estimated: 2-3x speedup)
1. Pre-upload all weights to GPU
. Implement memory pool allocator

### Phase 2: Core Optimizations (Estimated: additional 2-3x)
1. Flash Attention implementation
2. CUDA Graph capture for decoder layers
3. Fused attention kernels

### Phase 3: Advanced Features (Estimated: additional 1.5-2x)
1. Mixed-precision (FP16/TF32)
2. KV Cache for autoregressive generation
3. Multi-GPU pipeline parallelism

### Phase 4: Scaling Optimizations
1. Dynamic expert load balancing
2. Speculative decoding
3. Sparse attention for long contexts

---

## Profiling Recommendations

### Key Metrics to Track
1. **Memory Bandwidth Utilization**: Target >80% of theoretical
2. **SM Occupancy**: Target >50% for compute kernels
3. **Kernel Launch Overhead**: Should be <10% of total time
4. **Host-Device Transfer Time**: Should be <5% after optimizations

### Profiling Tools
- `nsys profile ./main` - Timeline analysis
- `ncu --set full ./main` - Kernel metrics
- `cuda-memcheck ./main` - Memory correctness

### Benchmark Targets
| Metric | Current (Estimated) | Target |
|--------|---------------------|--------|
| Tokens/sec (single GPU) | ~10 | ~50-100 |
| Tokens/sec (4 GPU) | ~30 | ~200-400 |
| First token latency | ~2s | ~500ms |
| Memory efficiency | ~30% | ~80% |

---

## Risk Assessment

### Technical Risks
1. **Flash Attention Complexity**: Requires careful implementation for correctness
2. **CUDA Graph Limitations**: Dynamic shapes may require multiple graphs

### Mitigation Strategies
1. Comprehensive validation against reference outputs
2. Incremental implementation with correctness checks
3. Fallback paths for unsupported configurations

---

## Code Locations Reference

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| Main matmul | layer.cu | 19-63 | Tiled matrix multiplication |
| RMSNorm GPU | layer.cu | 66-115 | Warp-reduced normalization |
| Attention kernels | model.cu | 144-229 | Score/softmax/output |
| MoE forward | model.cu | 868-1013 | Expert batching |
| ShortConv fused | model.cu | 237-328 | Transpose/split/conv fusion |
| Embedding lookup | model.cu | 334-350 | GPU token embedding |
| Stream manager | model.cu | 445-513 | Async transfer support |
| MLP forward | model.cu | 724-788 | Feed-forward network |
| Model forward | model.cu | 1450-1520 | Main inference loop |

---

## Conclusion

This optimization plan provides a structured approach to improving the LFM2-8B-A1B inference performance. The key focus areas are:

1. **Memory efficiency**: Eliminate redundant transfers
2. **Parallelism**: Better multi-GPU utilization
3. **Algorithmic**: KV cache and Flash Attention

Expected overall speedup: **5-10x** with full implementation of Phase 1-3 optimizations.
