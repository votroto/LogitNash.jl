set terminal pngcairo transparent enhanced size 480,1440
set output '/tmp/mini.png'

# set terminal qt size 800,800

set view 60, 110

set xrange [0:1]
set yrange [0:1]
set zrange [0:15]


set border 4095 lc rgb '#aa666666' dt 2
set xyplane 0

set cbrange [0:9]
unset colorbox
set palette defined (0.0 '#3B4CC0', 0.2 '#2A9D8F', 0.4 '#5ab04a', 0.6 '#d4a72c', 0.8 '#d95f59', 1.0 '#b40226')

set margins at screen 0.2, at screen 0.8, at screen 0.8, at screen 0.25

# splot  '< LC_ALL=C; awk ''NF>2{printf "%.1f %.3f %s\n", ($1+$2*0.37+$3*0.61), $17, $0}'' /tmp/path.dat | sort -g -k1,1 -k2,2 | cut -d'' '' -f3-' u 1:2:(($17 >= 0) ? $3 : NaN):(log($17+1)) w p lc palette pt 7 ps 1 notitle, '' u 1:2:(($17 < 0) ? $3 : NaN) w p lc 'black' pt 7 ps 0.01 notitle


splot  '/tmp/path.dat' u 1:2:(log($7+1)) w p notitle, '' u 4:5:(log($7+1)) w p notitle