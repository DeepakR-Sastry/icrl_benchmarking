#!/bin/bash
# Our VICRL on HalfCheetah-with-position, 20 seeds x 30 rounds x 2e5 steps,
# to compare with Liu's own runs (wandb project icrl-hc-icrl, group
# ours_vicrl). A copy of ours_gacl.sh with the config, time and group
# changed. Local smoke run: ~16 s per 20,480 steps -> ~1.5 h per run on one
# thread (docs/context.md); 6 h leaves room for a slower node.
# Setup (venv, clone, expert pile): docs/decisions.md, 2026-10-28; the
# expert pile expert_data/hc_liu.pkl is made with scripts/make_hc_liu_expert.py.
# Submit from ~/scratch/icrl_benchmarking:
#   mkdir -p logs && sbatch scripts/cluster/ours_vicrl.sh
# Another config and wandb group (e.g. the 60-round runs):
#   sbatch --export=ALL,CONFIG=configs/hc/vicrl_60.yaml,GROUP=ours_vicrl_60 \
#     scripts/cluster/ours_vicrl.sh
#SBATCH --account=aip-s2ganapa
#SBATCH --job-name=ours_vicrl
#SBATCH --array=0-19
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%A_%a.out

# Modules first, then the venv: loading modules resets some variables.
module load StdEnv/2023 python/3.13.2 mujoco/3.3.0
source $HOME/icrl_venv/bin/activate
# wandb offline (no internet on compute nodes); torch on the one CPU we asked for.
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Config paths (the expert pile) are relative to the repo root; runs go to
# runs/hc_vicrl_seed<k>/ (hc_vicrl_60_seed<k>/ for vicrl_60.yaml), wandb offline files to ./wandb.
cd $HOME/scratch/icrl_benchmarking

# The 30-round config and its group unless given at submission.
CONFIG=${CONFIG:-configs/hc/vicrl.yaml}
GROUP=${GROUP:-ours_vicrl}

python scripts/run_icrl.py --config $CONFIG \
    --seed $SLURM_ARRAY_TASK_ID \
    --wandb-project icrl-hc-icrl --wandb-group $GROUP
