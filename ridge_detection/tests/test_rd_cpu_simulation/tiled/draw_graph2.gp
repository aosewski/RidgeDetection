#!/usr/bin/gnuplot
reset

set terminal pngcairo size 960,640 enhanced notransparent font 'Verdana,12'

set border 3 back lc rgb '#808080' lt 1.5
set grid back lc rgb '#808080' lt 0 lw 1


# set style line 1 pt 1 ps 1 lt 1 lw 1.3 lc rgb '#D53E4F' # red
# set style line 2 pt 2 ps 1 lt 1 lw 1.3 lc rgb '#F46D43' # orange
# set style line 3 pt 3 ps 1 lt 1 lw 1.3 lc rgb '#FDAE61' # pale orange      
# set style line 4 pt 4 ps 1 lt 1 lw 1.3 lc rgb '#FEE08B' # pale yellow-orange          
# set style line 5 pt 5 ps 1 lt 1 lw 1.3 lc rgb '#E6F598' # pale yellow-green          
# set style line 6 pt 6 ps 1 lt 1 lw 1.3 lc rgb '#ABDDA4' # pale green  
# set style line 7 pt 7 ps 1 lt 1 lw 1.3 lc rgb '#66C2A5' # green  
# set style line 8 pt 8 ps 1 lt 1 lw 1.3 lc rgb '#3288BD' # blue  
# set style line 9 pt 9 ps 1 lt 1 lw 1.3 lc rgb '#000000 '# black

# set style line 1 lt 1 lc rgb '#F7FBFF' # very light blue
# set style line 1 lt 1 lc rgb '#DEEBF7' # 
# set style line 3 lt 1 lc rgb '#C6DBEF' # 
set style line 1 lt 1 lc rgb '#9ECAE1' # light blue
# set style line 5 lt 1 lc rgb '#6BAED6' # 
set style line 2 lt 1 lc rgb '#4292C6' # medium blue
# set style line 7 lt 1 lc rgb '#2171B5' #
set style line 3 lt 1 lc rgb '#084594' # dark blue

# set style line 1 lt 1 lc rgb '#F7FCF5' # very light green
# set style line 4 lt 1 lc rgb '#E5F5E0' # 
# set style line 3 lt 1 lc rgb '#C7E9C0' # 
set style line 4 lt 1 lc rgb '#A1D99B' # light green
# set style line 5 lt 1 lc rgb '#74C476' # 
set style line 5 lt 1 lc rgb '#41AB5D' # medium green
# set style line 7 lt 1 lc rgb '#238B45' #
set style line 6 lt 1 lc rgb '#005A32' # dark green

# set style line 1 lt 1 lc rgb '#FCFBFD' # very light purple
# set style line 7 lt 1 lc rgb '#EFEDF5' # 
# set style line 3 lt 1 lc rgb '#DADAEB' # 
set style line 7 lt 1 lc rgb '#BCBDDC' # light purple
# set style line 5 lt 1 lc rgb '#9E9AC8' # 
set style line 8 lt 1 lc rgb '#807DBA' # medium purple
# set style line 7 lt 1 lc rgb '#6A51A3' #
set style line 9 lt 1 lc rgb '#4A1486' # dark purple

# set style line 10 lt 1 lc rgb '#FEE6CE' # 
# set style line 3 lt 1 lc rgb '#FDD0A2' # 
set style line 10 lt 1 lc rgb '#FDAE6B' # light orange
# set style line 5 lt 1 lc rgb '#FD8D3C' # 
set style line 11 lt 1 lc rgb '#F16913' # medium orange
# set style line 7 lt 1 lc rgb '#D94801' #
set style line 12 lt 1 lc rgb '#8C2D04' # dark orange

# set style line 1 lt 1 lc rgb '#ffffe0' # light yellow
# set style line 13 lt 1 lc rgb '#ffdfb8' #
# set style line 3 lt 1 lc rgb '#ffbc94' # light orange
# set style line 14 lt 1 lc rgb '#ff9777' #
set style line 13 lt 1 lc rgb '#ff6962' # light red
# set style line 15 lt 1 lc rgb '#ee4256' #
set style line 14 lt 1 lc rgb '#d21f47' # red
# set style line 8 lt 1 lc rgb '#b0062c' #
set style line 15 lt 1 lc rgb '#8b0000' # dark red

set style fill solid 0.95 border rgb 'grey30'

set key outside right top vertical
set tics nomirror

set ylabel '[s]'
set logscale y
set tmargin 2
set rmargin 7

colStep = 0.3
bs = 2 * colStep    # box width
nCol = 5
groupStep = (nCol+1) * bs
nGroups = 4
offset = 9 * colStep
xEnd = offset + (nGroups-1) * groupStep + 9 * colStep 
set xrange [0:xEnd]
set xtics nomirror out ('2D-wygł.' offset,'2D-bez wygł.' offset + groupStep, '3D-wygł.' offset + 2*groupStep,\
     '3D-bez wygł.' offset + 3*groupStep)

dataFile = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_cpu_simulation/tiled/gnuplot_data/cpu_tiled_2D_3D_perf1.txt'

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_cpu_simulation/tiled/img/cpu_tiled_2D_3D_perf2.png'

# plot dataFile i 0 u (offset + $0 * groupStep - 4 * colStep):($11/1000.0):(bs) t 'LT LRD' w boxes ls 1, \
#      ''       i 1 u (offset + $0 * groupStep - 2 * colStep):($11/1000.0):(bs) t 'LT MRD' w boxes ls 2, \
#      ''       i 2 u (offset + $0 * groupStep - 0 * colStep):($11/1000.0):(bs) t 'GT LRD' w boxes ls 5, \
#      ''       i 3 u (offset + $0 * groupStep + 2 * colStep):($11/1000.0):(bs) t 'GT MRD' w boxes ls 7, \
#      ''       i 4 u (offset + $0 * groupStep + 4 * colStep):($11/1000.0):(bs) t 'ET MRD' w boxes ls 8, \
#      ''       i 0 u (offset + $0 * groupStep - 4 * colStep):($11/1000.0 + 1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left, \
#      ''       i 1 u (offset + $0 * groupStep - 2 * colStep):($11/1000.0 + 1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left, \
#      ''       i 2 u (offset + $0 * groupStep - 0 * colStep):($11/1000.0 + 1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left, \
#      ''       i 3 u (offset + $0 * groupStep + 2 * colStep):($11/1000.0 + 1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left, \
#      ''       i 4 u (offset + $0 * groupStep + 4 * colStep):($11/1000.0 + 1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left
     

plot dataFile i 0 u (offset + $0 * groupStep - 4 * colStep):($11/1000.0):(bs) t 'LT LRD' w boxes ls 1, \
     ''       i 1 u (offset + $0 * groupStep - 2 * colStep):($11/1000.0):(bs) t 'LT MRD' w boxes ls 4, \
     ''       i 2 u (offset + $0 * groupStep - 0 * colStep):($11/1000.0):(bs) t 'GT LRD' w boxes ls 7, \
     ''       i 3 u (offset + $0 * groupStep + 2 * colStep):($11/1000.0):(bs) t 'GT MRD' w boxes ls 10, \
     ''       i 4 u (offset + $0 * groupStep + 4 * colStep):($11/1000.0):(bs) t 'ET MRD' w boxes ls 13, \
     ''       i 0 u (offset + $0 * groupStep - 4 * colStep):(($14+$15)/1000.0):(bs) notitle w boxes ls 2, \
     ''       i 1 u (offset + $0 * groupStep - 2 * colStep):(($14+$15)/1000.0):(bs) notitle w boxes ls 5, \
     ''       i 2 u (offset + $0 * groupStep - 0 * colStep):(($14+$15)/1000.0):(bs) notitle w boxes ls 8, \
     ''       i 3 u (offset + $0 * groupStep + 2 * colStep):(($14+$15)/1000.0):(bs) notitle w boxes ls 11, \
     ''       i 4 u (offset + $0 * groupStep + 4 * colStep):(($14+$15)/1000.0):(bs) notitle w boxes ls 14, \
     ''       i 0 u (offset + $0 * groupStep - 4 * colStep):($14/1000.0):(bs) notitle w boxes ls 3, \
     ''       i 1 u (offset + $0 * groupStep - 2 * colStep):($14/1000.0):(bs) notitle w boxes ls 6, \
     ''       i 2 u (offset + $0 * groupStep - 0 * colStep):($14/1000.0):(bs) notitle w boxes ls 9, \
     ''       i 3 u (offset + $0 * groupStep + 2 * colStep):($14/1000.0):(bs) notitle w boxes ls 12, \
     ''       i 4 u (offset + $0 * groupStep + 4 * colStep):($14/1000.0):(bs) notitle w boxes ls 15, \
     ''       i 0 u (offset + $0 * groupStep - 4 * colStep):($11/1000.0 * 1.1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 1 u (offset + $0 * groupStep - 2 * colStep):($11/1000.0 * 1.1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 2 u (offset + $0 * groupStep - 0 * colStep):($11/1000.0 * 1.1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 3 u (offset + $0 * groupStep + 2 * colStep):($11/1000.0 * 1.1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 4 u (offset + $0 * groupStep + 4 * colStep):($11/1000.0 * 1.1):(gprintf("%.3f",$11/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 0 u (offset + $0 * groupStep - 4 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 1 u (offset + $0 * groupStep - 2 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 2 u (offset + $0 * groupStep - 0 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 3 u (offset + $0 * groupStep + 2 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 4 u (offset + $0 * groupStep + 4 * colStep):($14/1000.0 * 1.1):(gprintf("%.3f",$14/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 0 u (offset + $0 * groupStep - 4 * colStep):(($14+$15)/1000.0 * 1.1):(gprintf("%.3f",($14+$15)/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 1 u (offset + $0 * groupStep - 2 * colStep):(($14+$15)/1000.0 * 1.1):(gprintf("%.3f",($14+$15)/1000.0)) notitle w labels rotate by 70 left offset -0.5,0, \
     ''       i 4 u (offset + $0 * groupStep + 4 * colStep):(($14+$15)/1000.0 * 1.1):(gprintf("%.3f",($14+$15)/1000.0)) notitle w labels rotate by 70 left offset -0.5,0
     
