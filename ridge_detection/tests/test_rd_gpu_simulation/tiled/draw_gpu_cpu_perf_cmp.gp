#!/usr/bin/gnuplot
reset

set terminal pngcairo size 960,640 enhanced notransparent font 'Verdana,12'

set border 3 back lc rgb '#808080' lt 1.5
set grid back lc rgb '#808080' lt 0 lw 1

set style line  1 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#D53E4F' # red
set style line  2 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#F46D43' # orange
set style line  3 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#FDAE61' # pale orange
set style line  4 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#FEE08B' # pale yellow-orange
set style line  5 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#E6F598' # pale yellow-green
set style line  6 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#ABDDA4' # pale green
set style line  7 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#66C2A5' # green
set style line  8 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#3288BD' # blue
set style line  9 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#762A83' # dark purple
set style line 10 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#9970AB' # medium purple
set style line 11 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#C2A5CF' # 
set style line 12 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#E7D4E8' # pale purple
set style line 13 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#D9F0D3' # pale green
set style line 14 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#A6DBA0' # 
set style line 15 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#5AAE61' # medium green
set style line 16 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#1B7837' # dark green

set style fill solid 0.95 border rgb 'grey30'

set key outside right top vertical
set tics nomirror
set tmargin 3.5
# set rmargin 

set ylabel '[s]'
set logscale y
set ytics format "10^{%T}"

colStep = 0.3
bs = 2 * colStep    # box width
nCol = 14
groupStep = (nCol+1) * bs
nGroups = 2
offset = 16 * colStep
xEnd = offset + (nGroups-1) * groupStep + 16 * colStep 
set xrange [0:xEnd]
set xtics nomirror out ('2D' offset,'3D' offset + groupStep)

dataFile = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/gnuplot_data/gpu_cpu_2D_3D_all_perf_cmp.txt'

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/img/gpu_cpu_2D_3D_all_perf_cmp.png'

plot dataFile i  0 u (offset + $0 * groupStep - 13 * colStep):($8/1000.0):(bs) t 'GTX 750Ti COL-ROW BF'  w boxes ls 1, \
     ''       i  1 u (offset + $0 * groupStep - 11 * colStep):($8/1000.0):(bs) t 'GTX 750Ti COL-COL BF'  w boxes ls 2, \
     ''       i  2 u (offset + $0 * groupStep -  9 * colStep):($8/1000.0):(bs) t 'GTX 750Ti ROW-ROW BF'  w boxes ls 3, \
     ''       i  3 u (offset + $0 * groupStep -  7 * colStep):($8/1000.0):(bs) t 'GTX TITAN COL-ROW BF'  w boxes ls 4, \
     ''       i  4 u (offset + $0 * groupStep -  5 * colStep):($8/1000.0):(bs) t 'GTX TITAN COL-COL BF'  w boxes ls 5, \
     ''       i  5 u (offset + $0 * groupStep -  3 * colStep):($8/1000.0):(bs) t 'GTX TITAN ROW-ROW BF'  w boxes ls 6, \
     ''       i  6 u (offset + $0 * groupStep -  1 * colStep):($8/1000.0):(bs) t 'Intel Xeon ROW-ROW BF' w boxes ls 7, \
     ''       i  7 u (offset + $0 * groupStep +  1 * colStep):($14/1000.0):(bs) t 'GTX 750Ti COL-ROW T'   w boxes ls 8, \
     ''       i  8 u (offset + $0 * groupStep +  3 * colStep):($14/1000.0):(bs) t 'GTX 750Ti COL-COL T'   w boxes ls 9, \
     ''       i  9 u (offset + $0 * groupStep +  5 * colStep):($14/1000.0):(bs) t 'GTX 750Ti ROW-ROW T'   w boxes ls 10, \
     ''       i 10 u (offset + $0 * groupStep +  7 * colStep):($14/1000.0):(bs) t 'GTX TITAN COL-ROW T'   w boxes ls 11, \
     ''       i 11 u (offset + $0 * groupStep +  9 * colStep):($14/1000.0):(bs) t 'GTX TITAN COL-COL T'   w boxes ls 12, \
     ''       i 12 u (offset + $0 * groupStep + 11 * colStep):($14/1000.0):(bs) t 'GTX TITAN ROW-ROW T'   w boxes ls 13, \
     ''       i 13 u (offset + $0 * groupStep + 13 * colStep):($11/1000.0):(bs) t 'Intel Xeon ROW-ROW T'  w boxes ls 14, \
     ''       i  0 u (offset + $0 * groupStep - 13 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  1 u (offset + $0 * groupStep - 11 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  2 u (offset + $0 * groupStep -  9 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  3 u (offset + $0 * groupStep -  7 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  4 u (offset + $0 * groupStep -  5 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  5 u (offset + $0 * groupStep -  3 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  6 u (offset + $0 * groupStep -  1 * colStep):($8/1000.0 * 1.1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  7 u (offset + $0 * groupStep +  1 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  8 u (offset + $0 * groupStep +  3 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left, \
     ''       i  9 u (offset + $0 * groupStep +  5 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 10 u (offset + $0 * groupStep +  7 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 11 u (offset + $0 * groupStep +  9 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 12 u (offset + $0 * groupStep + 11 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 13 u (offset + $0 * groupStep + 13 * colStep):($11/1000.0 * 1.1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left
