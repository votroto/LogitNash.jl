set terminal qt size 420,420

set view 60,30

set xtics 0.2
set ytics 0.2
set ztics 0.2

set xrange [0:1]
set yrange [0:1]
set zrange [0:1]

set border 4095
set xyplane 0

set xlabel 'p_1'
set ylabel 'p_2'
set zlabel 'p_3'

set cbrange [0:9]
set palette defined (0.0 '#3B4CC0', 0.2 '#2A9D8F', 0.4 '#5ab04a', 0.6 '#d4a72c', 0.8 '#d95f59', 1.0 '#b40426')

x0 = 0.5; y0 = 0.5; z0 = 0.5

splot 'path2.dat' u 1:3:(0) w l lc 'gray' dt 2 notitle, \
      '' u (0):3:5 w l lc 'gray' dt 2 notitle, \
      '' u 1:(1):5 w l lc 'gray' dt 2 notitle, \
      '' u (dx=$1-x0, dy=$3-y0, dz=$5-z0, x0=$1, y0=$3, z0=$5, $1-dx) : \
         ($3-dy) : ($5-dz) : (dx ) : (dy ) : (dz ) : (log($8)) \
         w vectors  filled lw 2 lc palette   t 'log(t)', \
      '+' every ::0::0 u (x0):(y0):(z0) w p pt 6 ps 2 lw 2 lc rgb '#b40426' t 'NE'
