#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH -J misinfo
# #SBATCH -J $1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32000
#SBATCH --time=48:00:00
#SBATCH --exclude=icsnode05,icsnode06
# #SBATCH --nodelist=icsnode12
# # preffered nodes: 08,13 (11G), 05,06 (40GB) but cuda incompatibilities 

# activate conda env
. ~/miniconda3/etc/profile.d/conda.sh
# source activate $1
# conda activate py37ptpl
echo "[$1] Activating env...."
. env_activate.sh

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# on cluster might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=^docker0,lo

# might need the latest CUDA
# module load NCCL/2.4.7-1-cuda.10.0

# avoid hugging tokenizer warning ->"The current process just got forked, after parallelism has already been used."
export TOKENIZERS_PARALLELISM=false

# run script 
echo "Running script ...."
# srun python3 train.py
# srun python3 train_intent_cls_transformer.py --strategy=ddp --gpus=1 --data_dirpath=data/input/intents/ --data_filename_prefix=intent_compilation4_ --max_seq_length=250 --warmup_steps=5000
#--t10sec=1000
#--t10sec=False
# guild run -q -y stance:train_cls1 &
# guild run -y stance:train_cls1
# guild run -y $1
if [ -z "$2" ]
  then
    echo "New Run"
    guild run -y $1
  else
    echo "Restarting Run [$2]"
    guild run -y $1 --force-sourcecode --restart $2
fi


echo "---------- FINALIZED -------------"