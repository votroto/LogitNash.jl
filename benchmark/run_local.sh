SAMPLES=3

JOB_ID=$(date +%s)
DIR="gamut.${JOB_ID}"
mkdir -p "$DIR"

while read -r NAME CMD; do
    julia ./wrapper_logitnash.jl "$SAMPLES" "$CMD" > "$DIR/${NAME}.dat"
done < "generators.conf"
