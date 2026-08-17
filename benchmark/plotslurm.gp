set terminal pdfcairo size 8in,4in
set output 'plot.pdf'

clamp(v, mn, mx) = v < mn ? mn : (v > mx ? mx : v)

set ylabel "Time (s)"

unset grid
set grid xtics lw 1 lt 1 lc rgb "#cccccc"
set xtics rotate by -90 offset -2.2,0
set key outside
set style fill transparent solid 0.35 noborder

set logscale y
#set yrange [0.02:40]
set xrange [0:22]

set xtics ( "BertrandOligopoly" 1, "BidirectionalLEG-CG" 2, "BidirectionalLEG-RG" 3, "BidirectionalLEG-SG" 4, "CovariantGame" 5, "CovariantGame-Pos" 6, "CovariantGame-Zero" 7, "DispersionGame" 8, "GraphicalGame-RG" 9, "GraphicalGame-Road" 10, "GraphicalGame-SG" 11, "GraphicalGame-SW" 12, "MinimumEffortGame" 13, "PolymatrixGame-CG" 14, "PolymatrixGame-RG" 15, "PolymatrixGame-Road" 16, "PolymatrixGame-SW" 17, "RandomGame" 18, "TravelersDilemma" 19, "UniformLEG-CG" 20, "UniformLEG-RG" 21, "UniformLEG-SG" 22 )

plot_time = strftime("%Y-%m-%d", time(0))

set label 1 at screen 0.01,0.01 left font ",7" tc rgb "#666666" \
    sprintf("Plotted: %s, Intel Xeon Scalable Gold 6146", plot_time)

plot for [i=0:21] 'xeonlogitnash.dat' index i using  (0.5+i-0.2*(rand(0)-0.5)):($1+$2) with points pt 7 ps 0.7 lc rgb "#99cc1155" notitle,  for [i=0:21] 'xeongambit.dat' index i using  (0.5+i-0.2*(rand(0)-0.5)):($5 == 0 ? $2+$3 : 0/0) with points pt 7 ps 0.7 lc rgb "#992211cc" notitle