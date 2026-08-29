#!/usr/bin/env bash

SAMPLES="$1"
CMD="$2"
TIMEFORMAT="%U %S"

# gambit-logit currently never admits failure in case of non-convergence.
# We take its word anyway as failure on the standard GAMUT slate is unlikely.

for ((i=1; i<=SAMPLES; i++)); do
    GAME=$(bash -c "$CMD")

    time gambit-logit -qe -l1000000 <<< "$GAME"
done