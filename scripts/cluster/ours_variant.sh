#!/bin/bash
# Our code on HalfCheetah, one config named by the first argument:
# configs/hc/<name>.yaml, wandb group ours__<name> (run_icrl.py's rule), run folders
# runs/hc_<name>_seed<k>/, seeds 1-20 (array). Any loop (ICRL, GACL, ICL):
# scripts/run_icrl.py picks it from the config. Written for the ICL bug-off
# runs as ours_icl_variant.sh, renamed 2026-11-12 for every config-by-name
# job (docs/decisions.md). 1 CPU, 6G, 8 h, compute[1-2] excluded (slow:
# ~4x on compute2; docs/decisions.md 2026-11-09). Change the seeds at
# submission with --array (e.g. --array=0-19).
# Submit from ~/scratch/icrl_benchmarking, one job per config (#SBATCH lines
# can't read $1, so the job name is given here):
#   mkdir -p logs
#   sbatch --job-name=<name> scripts/cluster/ours_variant.sh <name>
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
    --wandb-project icrl-hc-icrl
