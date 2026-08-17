#!/bin/bash
#SBATCH --job-name=gamutlogitnash
#SBATCH --partition=cpu
#SBATCH --array=0-21
#SBATCH --output=logitnash-%A_%a.out
#SBATCH --error=logitnash-%A_%a.err

DISTRIBUTIONS=(
    BertrandOligopoly
    BidirectionalLEG-CG
    BidirectionalLEG-RG
    BidirectionalLEG-SG
    CovariantGame
    CovariantGame-Pos
    CovariantGame-Zero
    DispersionGame
    GraphicalGame-RG
    GraphicalGame-Road
    GraphicalGame-SG
    GraphicalGame-SW
    MinimumEffortGame
    PolymatrixGame-CG
    PolymatrixGame-RG
    PolymatrixGame-Road
    PolymatrixGame-SW
    RandomGame
    TravelersDilemma
    UniformLEG-CG
    UniformLEG-RG
    UniformLEG-SG
)

DISTRIBUTION="${DISTRIBUTIONS[$SLURM_ARRAY_TASK_ID]}"

ml Java
ml Julia

GENERATOR="java -jar $HOME/opt/gamut.jar -random_params -players 6 -actions 5 -normalize -min_payoff 0 -max_payoff 1 -output GambitOutput -f /dev/stdout -g $DISTRIBUTION"

echo "$GENERATOR" | julia ./run_benchmark.jl 100
