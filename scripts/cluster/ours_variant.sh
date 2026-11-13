#!/bin/bash
# Our code, one config named by the first argument as <folder>/<name>
# (e.g. hc/icl_ray, ant/icl_ray): configs/<folder>/<name>.yaml, wandb project
# icrl-<folder>-icrl, group ours__<folder>_<name> (run_icrl.py's rule), run
# folders runs/<folder>_<name>_seed<k>/, seeds 1-20 (array). Any loop (ICRL,
# GACL, ICL): scripts/run_icrl.py picks it from the config. Written for the
# ICL bug-off runs as ours_icl_variant.sh, renamed 2026-11-12 for every
# config-by-name job, <folder>/ added 2026-11-12 for Ant (docs/decisions.md).
# 1 CPU, 6G, 8 h, compute[1-2] excluded (slow: ~4x on compute2;
# docs/decisions.md 2026-11-09). Change the seeds at submission with --array
# (e.g. --array=0-19).
# Submit from ~/scratch/icrl_benchmarking, one job per config (#SBATCH lines
# can't read $1, so the job name is given here; no "/" in it, as the log
# path logs/<job name>_... is built from it):
#   mkdir -p logs
#   sbatch --job-name=<folder>_<name> scripts/cluster/ours_variant.sh <folder>/<name>
#SBATCH --account=aip-s2ganapa
#SBATCH --array=1-20
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=8:00:00
#SBATCH --exclude=compute[1-2]
#SBATCH --output=logs/%x_%A_%a.out

# Config paths (the expert pile) are relative to the repo root; runs go to
# runs/<folder>_<name>_seed<k>/, wandb offline files to ./wandb.
cd $HOME/scratch/icrl_benchmarking

# Refuse a missing or misspelled config before anything runs.
NAME=$1                # e.g. ant/icl_ray
FOLDER=${NAME%%/*}     # the part before the first "/": ant
if [ -z "$NAME" ] || [ "$FOLDER" = "$NAME" ] || [ ! -f "configs/$NAME.yaml" ]; then
    echo "usage: sbatch --job-name=<folder>_<name> $0 <folder>/<name>   (configs/<folder>/<name>.yaml must exist)" >&2
    exit 1
fi

# Modules first, then the venv: loading modules resets some variables.
module load StdEnv/2023 python/3.13.2 mujoco/3.3.0
source $HOME/icrl_venv/bin/activate
# wandb offline (no internet on compute nodes); torch on the one CPU we asked for.
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/run_icrl.py --config configs/$NAME.yaml \
    --seed $SLURM_ARRAY_TASK_ID \
    --wandb-project icrl-$FOLDER-icrl
