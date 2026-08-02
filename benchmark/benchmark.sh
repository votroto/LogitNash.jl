#!/usr/bin/env bash
set -euo pipefail

GAMUT="/home/votroto/Downloads/gamut/gamut.jar"
TMP_DIR=$(mktemp -d "bench_fifo_XXXXXX")

cleanup() {
    if [[ -n "${TMP_DIR}" && -d "${TMP_DIR}" ]]; then
        rm -rf "${TMP_DIR}"
    fi
    trap - EXIT
    kill -- -$$ 2>/dev/null
}
trap cleanup EXIT INT TERM HUP

FIFO_PATH="$TMP_DIR/data.fifo"
mkfifo "$FIFO_PATH"

( ulimit -c 0
echo "core limit: $(ulimit -c)"
  julia -L server.jl -e 'serve_solver(getaddrinfo("localhost"), 3000; stop_t=1e6, stop_eps=0.0)'
) &
sleep 1

java -jar "$GAMUT" -g UniformLEG-SG -random_params -players 6 -normalize -min_payoff -1 -max_payoff 1 -output GambitOutput -f "$FIFO_PATH" -actions 5 &
cat "$FIFO_PATH" | nc -N localhost 3000 | grep time | sed 's/.*= //' | paste -sd' '

for i in {0..10}; do
    java -jar "$GAMUT" -g UniformLEG-SG -random_params -players 6 -normalize -min_payoff -1 -max_payoff 1 -output GambitOutput -f "$FIFO_PATH" -actions 5 &
    cat "$FIFO_PATH" | nc -N localhost 3000 | grep time | sed 's/.*= //' | paste -sd' '
done


sleep 1
