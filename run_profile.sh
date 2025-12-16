#!/bin/bash

# Use 4 nodes by default for optimal data parallelism
: ${NODES:=4}

salloc -N $NODES --partition class1 --exclusive --gres=gpu:4   \
	mpirun --bind-to none -mca btl ^openib -npernode 4 \
		--oversubscribe -quiet \
		ncu -o ncu_report --set full ./main -n 1024 -b 64 $@
