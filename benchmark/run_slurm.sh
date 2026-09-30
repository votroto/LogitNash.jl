#!/bin/bash
#SBATCH --job-name=gamut
#SBATCH --time=04:00:00
#SBATCH --partition=amd
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --array=4-18

SAMPLES=100

DIR="gamut.${SLURM_ARRAY_JOB_ID}"
mkdir -p "$DIR"

ml Java
ml Julia
ml GCC



LINE="RandomGame java -jar $HOME/opt/gamut.jar -random_params -players 6 -actions 5 -normalize -min_payoff -1 -max_payoff 1 -output GambitOutput -f /dev/stdout -g RandomGame"
read -r NAME CMD <<< "$LINE"

sh ./wrapper_gambit.sh "$SAMPLES"  "$SLURM_ARRAY_TASK_ID" "$CMD" > "$DIR/${NAME}${SLURM_ARRAY_TASK_ID}.dat"
#julia ./wrapper_logitnash.jl "$SAMPLES" "$SLURM_ARRAY_TASK_ID" "$CMD" > "$DIR/${NAME}${SLURM_ARRAY_TASK_ID}.dat"
