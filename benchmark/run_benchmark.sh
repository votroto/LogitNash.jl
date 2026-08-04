julia ./run_benchmark.jl $1 < ./generators.txt > runtimes.dat
gnuplot ./plot.gp