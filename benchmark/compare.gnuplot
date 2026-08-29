# Usage:       this file      baseline_folder    new_folder
# gnuplot -c plot.gnuplot       sota_results    my_results

DIR1 = ARG1
DIR2 = ARG2
OUTNAME = DIR1 . "_vs_" . DIR2 . ".pdf"






#set logscale x
set key tc rgb '#000000'
set tics textcolor rgb "#888888"
set border behind lc rgb "#555555"
set linestyle 1 dt 3 lc rgb "#222222"
set grid x,y ls 1
unset grid
#set grid y2tics lw 1 lt 1 lc rgb "#222222"




dists = system("awk 'NF {print $1}' generators.conf")

# Helper functions
clamp(x) = (x >= 100) ? 100 : ((x <= 0.01) ? 0.01 : x)
jitter(x, amount) = x + amount * (rand(0) - 0.5)

set terminal pdfcairo size 5in,10in background rgb "#dddddd"

set output OUTNAME

set logscale y
set ylabel "Time (s)"

unset grid
set grid xtics lw 1 lt 1 lc rgb "#cccccc"
set xtics rotate by -90 offset first 0.5,0
set xtics

# X-range scales dynamically based on the number of benchmarks
set xrange [0:words(dists)]

# Calculate statistics, generate labels, and set up xtics
do for [i=1:words(dists)] {
    name = word(dists, i)
print(name)
    # Place the tick label exactly at i
    set xtics add (word(dists, i) i-1)

    file1 = DIR1 . "/" . name . ".dat"
    file2 = DIR2 . "/" . name . ".dat"

    # Use the built-in stats command to extract the medians silently
    stats file1 using (($2==0) ? $1 : NaN) nooutput
    tim_A = exists("STATS_records") ? STATS_median : NaN

    stats file2 using (($2==0) ? $1 : NaN) nooutput
    tim_B = exists("STATS_records") ? STATS_median : NaN

    # Speedup: Baseline time / New time
    # (If B is twice as fast, speedup = 2.0)
    speedup = tim_A / tim_B

    # Print the speedup label at X = i, Y = 0.0065 (just inside the graph bottom)
    set label i sprintf("%.1f×", speedup) at (i-0.5), 0.01 offset 0,character 2 rotate by -90  center font ",9" front
}

#set yrange [0.01:50]

set key inside top right box maxrows 1
set style fill transparent solid 0.35 noborder

# Define distinct colors for the datasets
# Solid colors for the legend, alpha (transparent) versions for the scatter points
color1_solid = "#8888aa"
color1_alpha = "#aa8888aa"
color2_solid = "#ff000d"
color2_alpha = "#aaff000d"

# Plot the legend entries (NaN points) first, then loop through and plot the data
plot \
    NaN with points pt 7 ps 1 lc rgb color1_solid title DIR1, \
    NaN with points pt 7 ps 1 lc rgb color2_solid title "{/:Bold LogitNash.jl}", \
    for [i=1:words(dists)] DIR1 . "/" . word(dists, i) . ".dat" \
        using (jitter(i - 0.5, 0.2)):(($2==0) ? clamp($1) : NaN) \
        with points pt 7 ps 0.3 lc rgb color1_alpha notitle, \
    for [i=1:words(dists)] DIR2 . "/" . word(dists, i) . ".dat" \
        using (jitter(i - 0.5, 0.2)):(($2==0) ? clamp($1) : NaN) \
        with points pt 7 ps 0.7 lc rgb color2_alpha notitle