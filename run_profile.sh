#!/bin/bash

# Profiling script with NSight Systems or NSight Compute
# Usage:
#   ./run_profile.sh              # NSight Systems (default)
#   ./run_profile.sh --ncu        # NSight Compute detailed profiling
#   ./run_profile.sh --ncu-kernel "matmul"  # Profile specific kernel

: ${NODES:=1}
: ${SAMPLES:=64}

# Parse arguments
USE_NCU=0
NCU_KERNEL=""
NCU_OPTS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ncu)
            USE_NCU=1
            shift
            ;;
        --ncu-kernel)
            USE_NCU=1
            NCU_KERNEL="$2"
            shift 2
            ;;
        --ncu-full)
            USE_NCU=1
            NCU_OPTS="--set full"
            shift
            ;;
        *)
            break
            ;;
    esac
done

if [ $USE_NCU -eq 1 ]; then
    echo "=== NSight Compute Profiling ==="
    echo "Nodes: $NODES, Samples: $SAMPLES"

    # Build ncu command
    NCU_CMD="ncu --target-processes all"

    # Kernel filter
    if [ -n "$NCU_KERNEL" ]; then
        NCU_CMD="$NCU_CMD --kernel-name-base demangled --kernel-name \"$NCU_KERNEL\""
        echo "Filtering kernel: $NCU_KERNEL"
    fi

    # Analysis sections
    if [ -n "$NCU_OPTS" ]; then
        NCU_CMD="$NCU_CMD $NCU_OPTS"
    else
        # Default: key metrics for optimization
        NCU_CMD="$NCU_CMD --section ComputeWorkloadAnalysis"
        NCU_CMD="$NCU_CMD --section MemoryWorkloadAnalysis"
        NCU_CMD="$NCU_CMD --section LaunchStats"
        NCU_CMD="$NCU_CMD --section Occupancy"
        NCU_CMD="$NCU_CMD --section SchedulerStats"
    fi

    # Output file
    NCU_CMD="$NCU_CMD -o ncu_report -f"

    # Limit kernel instances to avoid long profiling time
    NCU_CMD="$NCU_CMD --launch-skip 0 --launch-count 10"

    echo "Running: $NCU_CMD ./main -n $SAMPLES -b 64 $@"
    echo ""

    salloc -N $NODES --partition class1 --exclusive --gres=gpu:4 \
        mpirun --bind-to none -mca btl ^openib -npernode 1 \
            --oversubscribe -quiet \
            $NCU_CMD ./main -n $SAMPLES -b 64 $@

    echo ""
    echo "Report saved to: ncu_report.ncu-rep"
    echo "View with: ncu-ui ncu_report.ncu-rep"
    echo "Or export: ncu -i ncu_report.ncu-rep --csv > ncu_report.csv"
else
    echo "=== NSight Systems Profiling ==="
    salloc -N $NODES --partition class1 --exclusive --gres=gpu:4 \
        mpirun --bind-to none -mca btl ^openib -npernode 4 \
            --oversubscribe -quiet \
            nsys profile --cudabacktrace=all ./main -n 1024 -b 64 $@
fi
