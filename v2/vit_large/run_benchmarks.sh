#!/bin/bash
export PYTHONUSERBASE=$PSCRATCH/lejepa_tomography/env
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.12/site-packages:$PYTHONPATH
export IMAGE="nersc/pytorch:25.02.01"

echo "=== STARTING VIT-LARGE GPU MEMORY BENCHMARKS ==="
for bs in 2 4 8 12 16; do
    shifter --image=$IMAGE python test_batch_size.py $bs
done
echo "=== BENCHMARKS COMPLETE ==="
