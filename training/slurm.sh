#!/bin/bash

#SBATCH --job-name=FootNet
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=1
#SBATCH -o training.out 
#SBATCH --cpus-per-task=32
#SBATCH --partition=HERMES
#SBATCH --mem=160000


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname -I|awk '{print $2}')

echo $HOST
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

world_size=15
exp=XNestedUNet24h_fullrun

srun torchrun \
--nnodes $world_size \
--nproc_per_node 1 \
--rdzv_id $exp \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:28100 XSTILT_multinode_training.py --experiment $exp  --batch_size 4 --world_size $world_size --num_workers 16 --backhours 24 --total_epochs 200 

