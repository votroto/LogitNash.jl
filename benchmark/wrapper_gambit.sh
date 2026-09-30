#!/usr/bin/env bash

SAMPLES="$1"
CMD="$2"
TIMEFORMAT="%U %S"

# gambit-logit currently never admits failure in case of non-convergence.
# We take its word anyway as failure on the standard GAMUT slate is unlikely.

for ((i=1; i<=SAMPLES; i++)); do
    GAME=$(bash -c "$CMD")

    exec 3>&2
    TIME_OUT=$( { time gambit-logit -qe -l1000000 <<< "$GAME" >&3 2>&3; } 2>&1 )
    EXIT_CODE=$?
    exec 3>&-

    CPU_TIME=$(awk '{printf "%.6f", $1 + $2}' <<< "$TIME_OUT")
    echo "$CPU_TIME $EXIT_CODE"
done