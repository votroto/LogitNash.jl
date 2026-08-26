#set terminal qt size 240,240

set terminal pngcairo transparent enhanced size 240,240
set output '/tmp/mini.png'


# set xtics 0.5
# set ytics 0.5
# set ztics 1

unset xtics
unset ytics
unset ztics

set arrow 1 to 1,0,0 head filled lc rgb '#666666'
set arrow 2 to 0,1,0 head filled lc rgb '#666666'
set arrow 3 to 0,0,1 head filled lc rgb '#666666'

set label 1 'p_1' at 1,0,-0.2 center back font 'Fira Code,10' tc rgb '#666666'
set label 2 'p_2' at 0,1,-0.2 center back font 'Fira Code,10' tc rgb '#666666'
set label 3 'p_3' at 0,-0.2,1 center back font 'Fira Code,10' tc rgb '#666666'

set view 60, 110

set xrange [0:1]
set yrange [0:1]
set zrange [0:1]

set border 4095 lc rgb '#aa666666' dt 2
set xyplane 0

set cbrange [0:9]
unset colorbox
set palette defined (0.0 '#3B4CC0', 0.2 '#2A9D8F', 0.4 '#5ab04a', 0.6 '#d4a72c', 0.8 '#d95f59', 1.0 '#b40226')

set margins at screen 0.2, at screen 0.8, at screen 0.8, at screen 0.25

splot  '< LC_ALL=C; awk ''NF>2{printf "%.1f %.3f %s\n", ($1+$3+$5), $7, $0}'' /tmp/path.dat | sort -g -k1,1 -k2,2 | cut -d'' '' -f3-' u 1:3:(($7 >= 0) ? $5 : NaN):(log($7+1)) w p lc palette pt 7 ps 1 notitle, '' u 1:3:(($7 < 0) ? $5 : NaN) w p lc 'black' pt 7 ps 0.01 notitle