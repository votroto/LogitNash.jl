#!/bin/bash
#SBATCH --job-name=gamut
#SBATCH --time=01:00:00
#SBATCH --partition=amd
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=1-22

SAMPLES=100

DIR="gamut.${SLURM_ARRAY_JOB_ID}"
mkdir -p "$DIR"

ml Java
ml Julia
ml GCC

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" generators.conf)
read -r NAME CMD <<< "$LINE"

#sh ./wrapper_gambit.sh "$SAMPLES" "$CMD" > "$DIR/${NAME}.dat"
julia ./wrapper_logitnash.jl "$SAMPLES" "$CMD" > "$DIR/${NAME}.dat"