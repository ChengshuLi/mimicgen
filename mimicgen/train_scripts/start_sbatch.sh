#!/bin/bash
#SBATCH --partition=viscam
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --account=viscam

# only use the following on partition with GPUs
# h100, a100
#SBATCH --gres=gpu:3090:1
# exclude low mem gpu
##SBATCH --exclude=viscam1,viscam7
#SBATCH --exclude=viscam1,viscam12

#SBATCH --job-name=b1k-datagen
#SBATCH --output=/svl/u/mengdixu/test_output/%j.out
#SBATCH --error=/svl/u/mengdixu/test_output/%j.err


# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

source /sailhome/mengdixu/.bashrc
conda activate mimicgen

# export WANDB_API_KEY=032b02330c1a745ffd2eae4ae84f36d81590dcac
# wandb login

# export PATH="/svl/u/weiyul/Research/kdm/Jacinle/bin:$PATH"
# export PYTHONPATH="/svl/u/weiyul/Research/kdm/knowledge-driven-manipulation/src:/svl/u/weiyul/Research/kdm/calvin/calvin_models:/svl/u/weiyul/Research/kdm/calvin/calvin_env:/svl/u/weiyul/Research/kdm/Jacinle:/svl/u/weiyul/Research/kdm/Concepts:$PYTHONPATH"

cd /svl/u/mengdixu/b1k-datagen/mimicgen

$1

# done
echo "Done"