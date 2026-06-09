#!/bin/bash
# Run SparseAlign with Wanda unstructured pruning
# Usage: bash sparsealign/run.sh

MODEL_PATH=${MODEL_PATH:-"models/Qwen2.5-1.5B-Instruct"}
SPARSITY=${SPARSITY:-0.5}
NSAMPLES=${NSAMPLES:-256}
DEVICE=${DEVICE:-"cuda:0"}

python -m sparsealign.main ${MODEL_PATH} wikitext2 \
    --nsamples ${NSAMPLES} \
    --seqlen 128 \
    --prune \
    --method wanda \
    --sparsity ${SPARSITY} \
    --compensate \
    --epochs 10 \
    --lr 5e-5 \
    --device ${DEVICE} \
    --run_name "wanda-${SPARSITY}" \
    --log_path ./logs \
    --checkpoint_dir ./checkpoints
