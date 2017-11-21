#!/usr/bin/gnuplot
reset

set terminal pngcairo size 1280,720 enhanced notransparent font 'Verdana,12'

dataFile1 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_partition/gnuplot_data/GeForce_GTX_750Ti_best_per_dim.txt'
dataFile2 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_partition/gnuplot_data/GeForce_GTX_TITAN_best_per_dim.txt'


set border 3 back lc rgb '#808080' lt 1.5
set grid back lc rgb '#808080' lt 0 lw 1

set style line 1 pt 1 ps 1 lt 1 lw 1.5 lc rgb '#D53E4F' # red
set style line 2 pt 2 ps 1 lt 1 lw 1.5 lc rgb '#F46D43' # orange
set style line 3 pt 3 ps 1 lt 1 lw 1.5 lc rgb '#FDAE61' # pale orange      
set style line 4 pt 4 ps 1 lt 1 lw 1.5 lc rgb '#FEE08B' # pale yellow-orange          
set style line 5 pt 5 ps 1 lt 1 lw 1.5 lc rgb '#E6F598' # pale yellow-green          
set style line 6 pt 6 ps 1 lt 1 lw 1.5 lc rgb '#ABDDA4' # pale green  
set style line 7 pt 7 ps 1 lt 1 lw 1.5 lc rgb '#66C2A5' # green  
set style line 8 pt 8 ps 1 lt 1 lw 1.5 lc rgb '#3288BD' # blue  
set style line 9 pt 9 ps 1 lt 1 lw 1.5 lc rgb '#542788' # dark purple      
set style line 10 pt 10 ps 1 lt 1 lw 1.5 lc rgb '#8073AC' # medium purple          
set style line 11 pt 11 ps 1 lt 1 lw 1.5 lc rgb '#B2ABD2' #  
set style line 12 pt 12 ps 1 lt 1 lw 1.5 lc rgb '#D8DAEB' # pale purple      
set style line 13 pt 13 ps 1 lt 1 lw 1.5 lc rgb '#FEE0B6' # pale orange      
set style line 14 pt 14 ps 1 lt 1 lw 1.5 lc rgb '#FDB863' #  
set style line 15 pt 15 ps 1 lt 1 lw 1.5 lc rgb '#E08214' # medium orange             
set style line 16 pt 16 ps 1 lt 1 lw 1.5 lc rgb '#B35806' # dark orange    

set style fill solid 0.95 border rgb 'grey30'

set key outside center right vertical Left noopaque reverse
set tics nomirror

set xlabel 'Wymiar danych.'
set ylabel 'GB/s'

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_partition/img/GeForce_GTX_750Ti_best_per_dim.png'

plot dataFile1 i 0 u 1:10:xtic(1) t 'CUB RM LDG 0.6' w lp ls 1, \
    ''        i 1 u 1:10:xtic(1) t 'CUB RM CG 0.6'   w lp ls 2, \
    ''        i 2 u 1:10:xtic(1) t 'CUB CM LDG 0.6'  w lp ls 3, \
    ''        i 3 u 1:10:xtic(1) t 'CUB CM CG 0.6'   w lp ls 4, \
    ''        i 4 u 1:10:xtic(1) t 'CUB RM LDG 0.2'  w lp ls 5, \
    ''        i 5 u 1:10:xtic(1) t 'CUB RM CG 0.2'   w lp ls 6, \
    ''        i 6 u 1:10:xtic(1) t 'CUB CM LDG 0.2'  w lp ls 7, \
    ''        i 7 u 1:10:xtic(1) t 'CUB CM CG 0.2'   w lp ls 8

   
set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_partition/img/GeForce_GTX_TITAN_best_per_dim.png'

plot dataFile2 i 0 u 1:10:xtic(1) t 'CUB RM LDG 0.6' w lp ls 1, \
    ''        i 1 u 1:10:xtic(1) t 'CUB RM CS 0.6'   w lp ls 2, \
    ''        i 2 u 1:10:xtic(1) t 'CUB CM LDG 0.6'  w lp ls 3, \
    ''        i 3 u 1:10:xtic(1) t 'CUB CM CS 0.6'   w lp ls 4, \
    ''        i 4 u 1:10:xtic(1) t 'CUB RM CG 0.6'   w lp ls 5, \
    ''        i 5 u 1:10:xtic(1) t 'CUB CM CG 0.6'   w lp ls 6, \
    ''        i 6 u 1:10:xtic(1) t 'CUB RM LDG 0.2'  w lp ls 7, \
    ''        i 7 u 1:10:xtic(1) t 'CUB RM CG 0.2'   w lp ls 8, \
    ''        i 8 u 1:10:xtic(1) t 'CUB CM LDG 0.2'  w lp ls 9, \
    ''        i 9 u 1:10:xtic(1) t 'CUB CM CG 0.2'   w lp ls 10

