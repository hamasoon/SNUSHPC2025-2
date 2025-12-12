#!/bin/bash

# Use 4 nodes by default for optimal data parallelism
: ${NODES:=4}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 1 \
		--oversubscribe -quiet \
		./main -n 16 $@
