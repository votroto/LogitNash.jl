#!/usr/bin/env bash

SAMPLES="$1"
CMD="$2"
TIMEFORMAT="%U %S"

for ((i=1; i<=SAMPLES; i++)); do
    GAME=$(bash -c "$CMD")

    exec 3>&2
    TIME_OUT=$( { time gambit-logit -qe -l1000000 <<< "$GAME" >&3 2>&3; } 2>&1 )
    EXIT_CODE=$?
    exec 3>&-

    CPU_TIME=$(awk '{printf "%.6f", $1 + $2}' <<< "$TIME_OUT")
    echo "$CPU_TIME $EXIT_CODE"
done