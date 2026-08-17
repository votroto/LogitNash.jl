reset
# set terminal pdfcairo size 6in,6in
# set output 'xeon.gold.6146.singlethread.racetolambda.pdf'
set terminal wxt size 800,450 background rgb "#121212"


clamp(v, mn, mx) = v < mn ? mn : (v > mx ? mx : v)


#set logscale x
set key tc rgb 0xffffff
set tics textcolor rgb "#888888"
set xlabel "Time (s)" tc rgb "#888888" offset graph 0.5,0
set border behind lc rgb "#555555"
set linestyle 1 dt 3 lc rgb "#222222"
set grid x,y ls 1
unset grid
set grid y2tics lw 1 lt 1 lc rgb "#222222"


# Transposed: ylabel becomes xlabel
#set xlabel "Time (s)"

# Transposed: xtics grid becomes y2tics grid (draws horizontal lines on the right axis tics)

# Transposed: moved labels to right axis (y2), removed rotation for natural reading
unset ytics
# Added a small negative Y offset to replicate your original `-1.0,0` alignment shift
set y2tics offset 0, second -0.5
set style fill transparent solid 0.35 noborder


# Transposed: Exact original xtics string moved to y2tics
set y2tics ( "BertrandOligopoly" 1, "BidirectionalLEG-CG" 2, "BidirectionalLEG-RG" 3, "BidirectionalLEG-SG" 4, "CovariantGame" 5, "CovariantGame-Pos" 6, "CovariantGame-Zero" 7, "DispersionGame" 8, "GraphicalGame-RG" 9, "GraphicalGame-Road" 10, "GraphicalGame-SG" 11, "GraphicalGame-SW" 12, "MinimumEffortGame" 13, "PolymatrixGame-CG" 14, "PolymatrixGame-RG" 15, "PolymatrixGame-Road" 16, "PolymatrixGame-SW" 17, "RandomGame" 18, "TravelersDilemma" 19, "UniformLEG-CG" 20, "UniformLEG-RG" 21, "UniformLEG-SG" 22 )

plot_time = strftime("%Y-%m-%d", time(0))


# --- NEW STATISTICS BLOCK ---
# Loop through all 22 classes to calculate average speedup
do for [i=0:21] {
    # 1. Get stats for LogitNash (assuming this is the new/optimized implementation)
    stats 'xeonlogitnash.dat' index i using ($1+$2) nooutput
    mean_logit = (STATS_records > 0) ? STATS_median : NaN

    # 2. Get stats for Gambit (ignoring invalid rows as in your plot command)
    stats 'xeongambit.dat' index i using ($5 == 0 ? $2+$3 : NaN) nooutput
    mean_gambit = (STATS_records > 0) ? STATS_median : NaN

    # 3. Calculate Speedup ratio
    speedup = mean_gambit / mean_logit

    # 4. Place a text label at the top of the plot (graph 0.95 for Y)
    # The X coordinate is i + 0.5 to center it perfectly between your grid lines.
    # We use label tags starting at 100 to avoid conflicting with your label 1.
    if (STATS_records > 0 && mean_logit > 0 && i != 17) {
        set label 100+i at graph 0.92, first (i + 0.5) \
            sprintf("%.1fx", speedup) \
            font ",8" tc rgb "#999999" front
    }
}
# ----------------------------

#set label 117 at graph 0.5, first (17.5) "{/:Bold 82x} speedup on random games" font ",12" tc rgb "#ffaa22" front


set yrange [0:22]
set y2range [0:22]
set key reverse maxrows 1 at screen 0.0, screen 0.01 bottom left samplen 12 width -8

# Transposed: Exact original plot command, with LogitNash.jl set to bold and red
plot NaN with points pt 7 ps 1 lc rgb "#ffaa22" title "{/:Bold LogitNash.jl}", \
     NaN with points pt 7 ps 0.7 lc rgb "#55666666" title "gambit-logit", \
     for [i=0:21] 'xeonlogitnash.dat' index i using ($1+$2):(0.5+i-0.2*(rand(0)-0.5)) with points pt 7 ps 0.9 lc rgb "#99ffaa22" notitle, \
     for [i=0:21] 'xeongambit.dat' index i using ($5 == 0 ? clamp($2+$3,0,100) : 0/0):(0.5+i-0.2*(rand(0)-0.5)) with points pt 7 ps 0.3 lc rgb "#55666666" notitle