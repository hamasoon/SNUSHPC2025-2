#!/bin/bash

# Nsight Compute profiling for GEMM kernels
# Conditions: 1 Node, 4 GPU, 256 Samples, 128 Batch

: ${NODES:=1}
: ${SAMPLES:=256}

# Profile GEMM kernels (matmul_transposed_kernel and grouped_expert_gemm_kernel)
# -k: kernel name filter (regex supported)
# -c: limit number of kernel launches to profile (reduce profiling time)
# --set full: collect all metrics including roofline
# -o: output file name

# Option 1: Profile all kernel instances (takes longer but comprehensive)
# TMPDIR=$HOME salloc -N $NODES --partition class1 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 4 --oversubscribe \
#     ncu -o ncu_gemm_full -f --set full \
#     -k "regex:matmul_transposed|grouped_expert_gemm" \
#     ./main -n $SAMPLES -b 128

# Option 2: Single GPU profiling (avoids MPI complexity and /tmp permission issues)
# Run on 1 node with 1 GPU, profiling only that single process
# This avoids the /tmp/nsight-compute-lock permission issue on compute nodes
mkdir -p $HOME/ncu_tmp
salloc -N 1 --partition class1 --exclusive --gres=gpu:1 \
    bash -c '
        cd '$PWD' && \
        TMPDIR='$HOME'/ncu_tmp \
        HOME='$HOME' \
        CUDA_VISIBLE_DEVICES=0 \
        ncu --set full \
        --replay-mode application \
        --app-replay-buffer memory \
        -k "regex:matmul_transposed|grouped_expert_gemm" \
        -c 10 \
        ./main -n '$SAMPLES' -b 128 2>&1 | tee ncu_gemm_output.txt
    '

# Note: -c 10 captures diverse kernel configurations
# Each kernel type has varying grid sizes (e.g., 96x32, 32x32, 28x4x8, etc.)

# Option 3: 4 GPU MPI profiling (if /tmp permission issue is resolved)
# Uncomment below and comment Option 2 if you need multi-GPU profiling
# mkdir -p $HOME/ncu_tmp
# salloc -N $NODES --partition class1 --exclusive --gres=gpu:4 \
#     mpirun --bind-to none -mca btl ^openib -npernode 4 --oversubscribe \
#     bash -c 'if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
#         TMPDIR='$HOME'/ncu_tmp \
#         HOME='$HOME' \
#         ncu --set full \
#         --replay-mode application \
#         --app-replay-buffer memory \
#         -k "regex:matmul_transposed|grouped_expert_gemm" \
#         -c 10 \
#         '$PWD'/main -n '$SAMPLES' -b 128 2>&1 | tee '$PWD'/ncu_gemm_output.txt
#     else
#         '$PWD'/main -n '$SAMPLES' -b 128
#     fi'
