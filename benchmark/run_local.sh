#!/bin/bash
SAMPLES=100

JOB_ID=$(date +%s)
DIR="gamut.${JOB_ID}"
mkdir -p "$DIR"

while read -r NAME CMD; do
    #sh ./wrapper_gambit.sh "$SAMPLES" "$CMD" > "$DIR/${NAME}.dat"
    julia ./wrapper_logitnash.jl "$SAMPLES" "$CMD" > "$DIR/${NAME}.dat"
done < "generators.conf"
