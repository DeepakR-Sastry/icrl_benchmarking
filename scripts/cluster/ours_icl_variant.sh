#!/bin/bash
# Our ICL (Gaurav et al.) on HalfCheetah-Constrained, one config named by the
# first argument: configs/hc/<name>.yaml, wandb group ours_<name>, run folders
# runs/hc_<name>_seed<k>/. Written for the bug-off runs (docs/decisions.md
# 2026-11-10): icl_ray_no7, icl_ray_no8, icl_ray_no9, icl_ray_fixed,
# icl_fixed. Otherwise the same as ours_icl_ray.sh (seeds 1-20, 1 CPU, 6G,
# 8 h, compute[1-2] excluded: see that file for why).
# Submit from ~/scratch/icrl_benchmarking, one job per config (#SBATCH lines
# can't read $1, so the job name is given here):
#   mkdir -p logs
#   sbatch --job-name=icl_ray_no7 scripts/cluster/ours_icl_variant.sh icl_ray_no7
#SBATCH --account=aip-s2ganapa
#SBATCH --array=1-20
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=8:00:00
#SBATCH --exclude=compute[1-2]
#SBATCH --output=logs/%x_%A_%a.out

# Config paths (the expert pile) are relative to the repo root; runs go to
# runs/hc_<name>_seed<k>/, wandb offline files to ./wandb.
cd $HOME/scratch/icrl_benchmarking

# Refuse a missing or misspelled config before anything runs.
NAME=$1
if [ -z "$NAME" ] || [ ! -f "configs/hc/$NAME.yaml" ]; then
    echo "usage: sbatch --job-name=<name> $0 <name>   (configs/hc/<name>.yaml must exist)" >&2
    exit 1
fi

# Modules first, then the venv: loading modules resets some variables.
module load StdEnv/2023 python/3.13.2 mujoco/3.3.0
source $HOME/icrl_venv/bin/activate
# wandb offline (no internet on compute nodes); torch on the one CPU we asked for.
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/run_icrl.py --config configs/hc/$NAME.yaml \
    --seed $SLURM_ARRAY_TASK_ID \
    --wandb-project icrl-hc-icrl --wandb-group ours_$NAME
