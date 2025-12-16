#!/bin/bash

# Profiling script with NSight Systems or NSight Compute
# Usage:
#   ./run_profile.sh              # NSight Systems (default)
#   ./run_profile.sh --ncu        # NSight Compute detailed profiling
#   ./run_profile.sh --ncu-kernel "matmul"  # Profile specific kernel
#   ./run_profile.sh --roofline   # Roofline analysis
#   ./run_profile.sh --roofline --ncu-kernel "gemm"  # Roofline for specific kernel

: ${NODES:=1}
: ${SAMPLES:=64}

# Parse arguments
USE_NCU=0
USE_ROOFLINE=0
NCU_KERNEL=""
NCU_OPTS=""
LAUNCH_SKIP=0
LAUNCH_COUNT=10

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
        --roofline)
            USE_NCU=1
            USE_ROOFLINE=1
            shift
            ;;
        --skip)
            LAUNCH_SKIP="$2"
            shift 2
            ;;
        --count)
            LAUNCH_COUNT="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

if [ $USE_NCU -eq 1 ]; then
    if [ $USE_ROOFLINE -eq 1 ]; then
        echo "=== NSight Compute Roofline Analysis ==="
    else
        echo "=== NSight Compute Profiling ==="
    fi
    echo "Nodes: $NODES, Samples: $SAMPLES"

    # Build ncu command
    export TMPDIR=$HOME
    NCU_CMD="ncu --target-processes all"

    # Kernel filter
    if [ -n "$NCU_KERNEL" ]; then
        NCU_CMD="$NCU_CMD --kernel-name-base demangled --kernel-name \"$NCU_KERNEL\""
        echo "Filtering kernel: $NCU_KERNEL"
    fi

    # Analysis sections
    if [ $USE_ROOFLINE -eq 1 ]; then
        # Roofline analysis requires specific metrics
        NCU_CMD="$NCU_CMD --section SpeedOfLight_RooflineChart"
        NCU_CMD="$NCU_CMD --section SpeedOfLight"
        NCU_CMD="$NCU_CMD --section MemoryWorkloadAnalysis"
        NCU_CMD="$NCU_CMD --section ComputeWorkloadAnalysis"
        NCU_CMD="$NCU_CMD --section Occupancy"

        # Collect raw metrics for roofline calculation (comma-separated)
        NCU_CMD="$NCU_CMD --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes.sum,l2__throughput.avg.pct_of_peak_sustained_elapsed"

        echo "Roofline metrics enabled"
    elif [ -n "$NCU_OPTS" ]; then
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
    if [ $USE_ROOFLINE -eq 1 ]; then
        NCU_CMD="$NCU_CMD -o ncu_roofline -f"
    else
        NCU_CMD="$NCU_CMD -o ncu_report -f"
    fi

    # Limit kernel instances to avoid long profiling time
    NCU_CMD="$NCU_CMD --launch-skip $LAUNCH_SKIP --launch-count $LAUNCH_COUNT"
    echo "Launch skip: $LAUNCH_SKIP, Launch count: $LAUNCH_COUNT"

    echo "Running: $NCU_CMD ./main -n $SAMPLES -b 64 $@"
    echo ""

    salloc -N $NODES --partition class1 --exclusive --gres=gpu:4 \
        mpirun --bind-to none -mca btl ^openib -npernode 1 \
            --oversubscribe -quiet -x TMPDIR=$HOME \
            $NCU_CMD ./main -n $SAMPLES -b 64 $@

    echo ""
    if [ $USE_ROOFLINE -eq 1 ]; then
        echo "Report saved to: ncu_roofline.ncu-rep"
        echo ""
        echo "=== Roofline Analysis Guide ==="
        echo "View interactive roofline: ncu-ui ncu_roofline.ncu-rep"
        echo ""
        echo "Key metrics to check:"
        echo "  - Arithmetic Intensity (FLOP/Byte): Higher = compute-bound"
        echo "  - % of Peak Compute: How close to theoretical max FLOPS"
        echo "  - % of Peak Memory BW: How close to theoretical max bandwidth"
        echo ""
        echo "T4 GPU Theoretical Peaks:"
        echo "  - FP32 Peak: 8.1 TFLOPS"
        echo "  - Memory BW: 320 GB/s"
        echo "  - Ridge Point: ~25 FLOP/Byte"
        echo ""
        echo "Export CSV: ncu -i ncu_roofline.ncu-rep --csv > roofline.csv"
    else
        echo "Report saved to: ncu_report.ncu-rep"
        echo "View with: ncu-ui ncu_report.ncu-rep"
        echo "Or export: ncu -i ncu_report.ncu-rep --csv > ncu_report.csv"
    fi
else
    echo "=== NSight Systems Profiling ==="
    salloc -N $NODES --partition class1 --exclusive --gres=gpu:4 \
        mpirun --bind-to none -mca btl ^openib -npernode 4 \
            --oversubscribe -quiet \
            nsys profile --cudabacktrace=all ./main -n 1024 -b 64 $@
fi
