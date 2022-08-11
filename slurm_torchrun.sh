#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%j.%x.info.log
#SBATCH --error=logs/%j.%x.error.log

export OMP_NUM_THREADS=1
export UCX_TLS=rc
# export NCCL_DEBUG=INFO
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_MIN_NCHANNELS=16
export NCCL_BUFFSIZE=4194304

export MASTER_ADDR=`hostname -i | cut -d' ' -f1`
export MASTER_PORT=`comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`

echo MASTER_ADDR:$MASTER_ADDR
echo MASTER_PORT:$MASTER_PORT

srun -l pipenv run torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=42 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT finetune.py
