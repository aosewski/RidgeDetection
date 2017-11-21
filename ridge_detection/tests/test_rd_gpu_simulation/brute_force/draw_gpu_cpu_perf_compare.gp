#!/usr/bin/gnuplot
reset

set terminal pngcairo size 960,640 enhanced notransparent font 'Verdana,12'

set border 3 back lc rgb '#808080' lt 1.5
set grid back lc rgb '#808080' lt 0 lw 1


set style line 1 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#D53E4F' # red
set style line 2 pt 2 ps 1 lt 1 lw 1.3 lc rgb '#F46D43' # orange
set style line 3 pt 3 ps 1 lt 1 lw 1.3 lc rgb '#FDAE61' # pale orange      
set style line 4 pt 4 ps 1 lt 1 lw 1.3 lc rgb '#FEE08B' # pale yellow-orange          
set style line 5 pt 5 ps 1 lt 1 lw 1.3 lc rgb '#E6F598' # pale yellow-green          
set style line 6 pt 6 ps 1 lt 1 lw 1.3 lc rgb '#ABDDA4' # pale green  
set style line 7 pt 7 ps 1 lt 1 lw 1.3 lc rgb '#66C2A5' # green  
set style line 8 pt 8 ps 1 lt 1 lw 1.3 lc rgb '#3288BD' # blue  
set style line 9 pt 9 ps 1 lt 1 lw 1.3 lc rgb '#000000 ' # black
                                                        
# set style line 9 pt 9 ps 1 lt 1 lw 1.3 lc rgb '#4575B4' # dark blue
# set style line 9 pt 9 ps 1 lt 1 lw 1.3 lc rgb '#542788' # dark purple      
# set style line 10 pt 10 ps 1 lt 1 lw 1.3 lc rgb '#8073AC' # medium purple          
# set style line 11 pt 11 ps 1 lt 1 lw 1.3 lc rgb '#B2ABD2' #  
# set style line 12 pt 12 ps 1 lt 1 lw 1.3 lc rgb '#D8DAEB' # pale purple      
# set style line 13 pt 13 ps 1 lt 1 lw 1.3 lc rgb '#FEE0B6' # pale orange      
# set style line 14 pt 14 ps 1 lt 1 lw 1.3 lc rgb '#FDB863' #  
# set style line 15 pt 15 ps 1 lt 1 lw 1.3 lc rgb '#E08214' # medium orange             
# set style line 16 pt 16 ps 1 lt 1 lw 1.3 lc rgb '#B35806' # dark orange   

# set style line 1 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#E31A1C' # dark red
# # set style line 2 pt 2 ps 1 lt 1 lw 1.3 lc rgb '#FFFF33' # yellow
# set style line 2 pt 2 ps 1 lt 1 lw 1.3 lc rgb '#ffee00' # yellow 
# set style line 3 pt 3 ps 1 lt 1 lw 1.3 lc rgb '#1F78B4' # dark blue
# set style line 4 pt 4 ps 1 lt 1 lw 1.3 lc rgb '#542788' # dark purple
# set style line 5 pt 5 ps 1 lt 1 lw 1.3 lc rgb '#FF7F00' # dark orange
# set style line 6 pt 6 ps 1 lt 1 lw 1.3 lc rgb '#33A02C' # dark green
# set style line 7 pt 7 ps 1 lt 1 lw 1.3 lc rgb '#F781BF' # pink
# set style line 8 pt 8 ps 1 lt 1 lw 1.3 lc rgb '#A65628' # brown

set style fill solid 0.95 border rgb 'grey30'

set key inside top center horizontal
set tics nomirror

set ylabel '[s]'
set logscale y

colStep = 0.3
bs = 2 * colStep    # box width
nCol = 7
groupStep = (nCol+1) * bs
nGroups = 2
offset = 9 * colStep
xEnd = offset + (nGroups-1) * groupStep + 9 * colStep 
set xrange [0:xEnd]
set xtics nomirror out ('2D' offset,'3D' offset + groupStep)

dataFile = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/brute_force/gnuplot_data/gpu_cpu_2D_3D_perf_compare_small.txt'

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/brute_force/img/gpu_cpu_2D_3D_perf_compare_s2.png'

plot dataFile i 0 u (offset + $0 * groupStep - 6 * colStep):($8/1000.0):(bs) t 'GTX 750Ti COL-ROW' w boxes ls 1, \
     ''       i 1 u (offset + $0 * groupStep - 4 * colStep):($8/1000.0):(bs) t 'GTX 750Ti COL-COL' w boxes ls 2, \
     ''       i 2 u (offset + $0 * groupStep - 2 * colStep):($8/1000.0):(bs) t 'GTX 750Ti ROW-ROW' w boxes ls 3, \
     ''       i 3 u (offset + $0 * groupStep - 0 * colStep):($8/1000.0):(bs) t 'GTX TITAN COL-ROW' w boxes ls 4, \
     ''       i 4 u (offset + $0 * groupStep + 2 * colStep):($8/1000.0):(bs) t 'GTX TITAN COL-COL' w boxes ls 5, \
     ''       i 5 u (offset + $0 * groupStep + 4 * colStep):($8/1000.0):(bs) t 'GTX TITAN ROW-ROW' w boxes ls 6, \
     ''       i 6 u (offset + $0 * groupStep + 6 * colStep):($8/1000.0):(bs) t 'Intel Xeon ROW-ROW' w boxes ls 7, \
     ''       i 0 u (offset + $0 * groupStep - 6 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 1 u (offset + $0 * groupStep - 4 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 2 u (offset + $0 * groupStep - 2 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 3 u (offset + $0 * groupStep - 0 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 4 u (offset + $0 * groupStep + 2 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 5 u (offset + $0 * groupStep + 4 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left, \
     ''       i 6 u (offset + $0 * groupStep + 6 * colStep):($8/1000.0 + 1):(gprintf("%.3f",$8/1000.0)) notitle w labels rotate by 70 left
     # ''       i 0 u (offset + $0 * groupStep - 6 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9, \
     # ''       i 1 u (offset + $0 * groupStep - 4 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9, \
     # ''       i 2 u (offset + $0 * groupStep - 2 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9, \
     # ''       i 3 u (offset + $0 * groupStep - 0 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9, \
     # ''       i 4 u (offset + $0 * groupStep + 2 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9, \
     # ''       i 5 u (offset + $0 * groupStep + 4 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9, \
     # ''       i 6 u (offset + $0 * groupStep + 6 * colStep):($8/1000.0):($9/1000.0):($10/1000.0) notitle w yerrorbars ls 9
