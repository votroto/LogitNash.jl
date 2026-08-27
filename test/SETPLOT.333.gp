
set terminal pngcairo transparent enhanced size 480,480
set output '/tmp/mini.png'

# set terminal qt size 240,240

unset xtics
unset ytics
unset ztics

set arrow 1 from 1,0,0 to 0,1,0 nohead lc rgb '#666666'
set arrow 2 from 0,1,0 to 0,0,1 nohead lc rgb '#666666'
set arrow 3 from 0,0,1 to 1,0,0 nohead lc rgb '#666666'

set label 1 'a_1' at 1,0,-0.2 center back font 'Fira Code,10' tc rgb '#666666'
set label 2 'a_2' at 0,1,-0.2 center back font 'Fira Code,10' tc rgb '#666666'
set label 3 'a_3' at 0,-0.2,1 center back font 'Fira Code,10' tc rgb '#666666'

set view 55, 135

set xrange [0:1]
set yrange [0:1]
set zrange [0:1]

set border 0 lc rgb '#aa666666' dt 2
set xyplane 0

set cbrange [0:9]
unset colorbox
set palette defined (0.0 '#3B4CC0', 0.2 '#2A9D8F', 0.4 '#5ab04a', 0.6 '#d4a72c', 0.8 '#d95f59', 1.0 '#b40226')

set lmargin at screen 0.2
set rmargin at screen 0.8
set tmargin at screen 0.7
set bmargin at screen 0.1


splot  '< LC_ALL=C; awk ''NF>2{printf "%.1f %.3f %s\n", ($1+$2*0.37+$3*0.61), $10, $0}'' /tmp/path.dat | sort -g -k1,1 -k2,2 | cut -d'' '' -f3-' u 1:2:(($10 >= 0) ? $3 : NaN):(log($10+1)) w p lc palette pt 7 ps 1 notitle, '' u 1:2:(($10 < 0) ? $3 : NaN) w p lc 'black' pt 7 ps 0.01 notitle