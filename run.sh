#!/bin/bash

# Use 4 nodes by default for optimal data parallelism
: ${NODES:=4}
: ${SAMPLES:=1024}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
		mpirun --bind-to none -mca btl ^openib -npernode 4 \
		--oversubscribe \
		./main -w -n $SAMPLES -b 128 $@
