# Usage:     this file    result folder (without trailing /)
# gnuplot -c plot.gnuplot gamut.1787679980

EXPERIMENTID=ARG1
OUTNAME = EXPERIMENTID . ".pdf"

dists = system("awk 'NF {print $1}' generators.conf")
cat_runtimes(NAME) = "< cat " . EXPERIMENTID . "/" . NAME . ".dat"
clamp(x) = (x >= 1) ? 1 : ((x <= 0.01) ? 0.01 : x)

set terminal pdfcairo size 8in,4in
set output OUTNAME

set logscale y
set ylabel "Time (s)"

unset grid
set grid xtics lw 1 lt 1 lc rgb "#cccccc"
set xtics rotate by -90 offset -2.2,0
set xtics
do for [i=1:words(dists)] {
    set xtics add (word(dists, i) i)
}

set key outside
set style fill transparent solid 0.35 noborder

set yrange [0.01:1]
set xrange [0:22]

plot for [i=1:words(dists)] \
    cat_runtimes(word(dists, i)) \
    using (-0.5+i-0.2*(rand(0)-0.5)):(clamp($1)) \
    with points pt 7 ps 0.5 lc rgb "#99cc1155" notitle