#!/bin/bash
# Our ICL (Gaurav et al.) on HalfCheetah-Constrained, 20 seeds, to compare
# with Gaurav's own runs (wandb project icrl-hc-icrl, group ours_icl). A copy
# of ours_vicrl.sh with the config, seeds, time, memory and group changed.
# Seeds 1-20: his hc.json seed is 1, and both local seed-1 runs used it
# (docs/results.md, step 17); his job uses 1-20 too.
# Local run: ~1.3 h on one thread (docs/context.md); 8 h leaves room for a
# slow node. compute1 and compute2 are excluded: compute2 ~4x slower
# (VICRL job 2104820; ICL test job 2220924 had no round done after 30 min),
# compute1 > 1.6x (test job 2221219: no round after 27 min; local 17 min);
# docs/decisions.md 2026-11-09. 6G: his run peaked at 4.3 GB; ours was not measured.
# Setup (venv + POT, clone, expert pile): docs/decisions.md, 2026-10-28 and
# 2026-11-09; the pile expert_data/hc_gaurav.pkl is made with
# scripts/make_gaurav_expert.py --env hc and copied over with scp.
# Submit from ~/scratch/icrl_benchmarking:
#   mkdir -p logs && sbatch scripts/cluster/ours_icl.sh
#SBATCH --account=aip-s2ganapa
#SBATCH --job-name=ours_icl
#SBATCH --array=1-20
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=8:00:00
#SBATCH --exclude=compute[1-2]
#SBATCH --output=logs/%x_%A_%a.out

# Modules first, then the venv: loading modules resets some variables.
module load StdEnv/2023 python/3.13.2 mujoco/3.3.0
source $HOME/icrl_venv/bin/activate
# wandb offline (no internet on compute nodes); torch on the one CPU we asked for.
export WANDB_MODE=offline WANDB_SILENT=true OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Config paths (the expert pile) are relative to the repo root; runs go to
# runs/hc_icl_seed<k>/, wandb offline files to ./wandb.
cd $HOME/scratch/icrl_benchmarking

python scripts/run_icrl.py --config configs/hc/icl.yaml \
    --seed $SLURM_ARRAY_TASK_ID \
    --wandb-project icrl-hc-icrl --wandb-group ours_icl
