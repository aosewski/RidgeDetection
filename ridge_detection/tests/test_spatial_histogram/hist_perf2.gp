#!/usr/bin/gnuplot
reset

# set terminal pngcairo size 1280,720 enhanced notransparent font 'Verdana,12'
set terminal pngcairo size 600,400 enhanced notransparent font 'Verdana,12'

dataFile1 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/gnuplot_data/GeForce_GTX_750_Ti_hist_wgmem.txt'
dataFile2 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/gnuplot_data/GeForce_GTX_TITAN_hist_wgmem.txt'
dataFile3 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/gnuplot_data/GeForce_GTX_750_Ti_hist_wogmem.txt'
dataFile4 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/gnuplot_data/GeForce_GTX_TITAN_hist_wogmem.txt'

dataFile5 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/gnuplot_data/GeForce_GTX_750_Ti_hist_wgmem_16bin.txt'
dataFile6 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/gnuplot_data/GeForce_GTX_TITAN_hist_wgmem_16bin.txt'


set border 3 back lc rgb '#808080' lt 1.5
set grid back lc rgb '#808080' lt 0 lw 1

set style line 1 pt 1 ps 1 lt 1 lw 2 lc rgb '#A6CEE3' # light blue
set style line 2 pt 2 ps 1 lt 1 lw 2 lc rgb '#1F78B4' # dark blue
set style line 3 pt 3 ps 1 lt 1 lw 2 lc rgb '#B2DF8A' # light green
set style line 4 pt 4 ps 1 lt 1 lw 2 lc rgb '#33A02C' # dark green
set style line 5 pt 5 ps 1 lt 1 lw 2 lc rgb '#FB9A99' # light red
set style line 6 pt 6 ps 1 lt 1 lw 2 lc rgb '#E31A1C' # dark red
set style line 7 pt 7 ps 1 lt 1 lw 2 lc rgb '#FDBF6F' # light orange
set style line 8 pt 8 ps 1 lt 1 lw 2 lc rgb '#FF7F00' # dark orange

set style fill solid 0.95 border rgb 'grey30'

# set key right top
# set key inside right top horizontal
set key inside right top vertical
set tics nomirror

set xlabel 'Wymiar danych.'
set ylabel 'GB/s'

# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/img/GeForce_GTX_750_Ti_hist_wgmem.png'

# plot dataFile1 i 0 u 1:10:xtic(1) t 'COL (CUB)'   w lp ls 2, \
#     ''        i 1 u 1:10:xtic(1) t 'ROW (CUB)'   w lp ls 4, \
#     ''        i 2 u 1:10:xtic(1) t 'COL (trove)' w lp ls 6, \
#     ''        i 3 u 1:10:xtic(1) t 'ROW (trove)' w lp ls 8

   
# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/img/GeForce_GTX_TITAN_hist_wgmem.png'

# plot dataFile2 i 0 u 1:10:xtic(1) t 'COL (CUB)'   w lp ls 2, \
#     ''        i 1 u 1:10:xtic(1) t 'ROW (CUB)'   w lp ls 4, \
#     ''        i 2 u 1:10:xtic(1) t 'COL (trove)' w lp ls 6, \
#     ''        i 3 u 1:10:xtic(1) t 'ROW (trove)' w lp ls 8

# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/img/GeForce_GTX_750_Ti_hist_wogmem.png'

# plot dataFile3 i 0 u 1:10:xtic(1) t 'COL (CUB)'   w lp ls 2, \
#     ''        i 1 u 1:10:xtic(1) t 'ROW (CUB)'   w lp ls 4, \
#     ''        i 2 u 1:10:xtic(1) t 'COL (trove)' w lp ls 6, \
#     ''        i 3 u 1:10:xtic(1) t 'ROW (trove)' w lp ls 8

   
# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/img/GeForce_GTX_TITAN_hist_wogmem.png'

# plot dataFile4 i 0 u 1:10:xtic(1) t 'COL (CUB)'   w lp ls 2, \
#     ''        i 1 u 1:10:xtic(1) t 'ROW (CUB)'   w lp ls 4, \
#     ''        i 2 u 1:10:xtic(1) t 'COL (trove)' w lp ls 6, \
#     ''        i 3 u 1:10:xtic(1) t 'ROW (trove)' w lp ls 8

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/img/GeForce_GTX_750_Ti_hist_wgmem_16bin.png'

plot dataFile5 i 0 u 1:10:xtic(1) t 'COL (CUB)'   w lp ls 2, \
    ''        i 1 u 1:10:xtic(1) t 'ROW (CUB)'   w lp ls 4, \
    ''        i 2 u 1:10:xtic(1) t 'COL (trove)' w lp ls 6, \
    ''        i 3 u 1:10:xtic(1) t 'ROW (trove)' w lp ls 8

   
set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/test_spatial_histogram/img/GeForce_GTX_TITAN_hist_wgmem_16bin.png'

plot dataFile6 i 0 u 1:10:xtic(1) t 'COL (CUB)'   w lp ls 2, \
    ''        i 1 u 1:10:xtic(1) t 'ROW (CUB)'   w lp ls 4, \
    ''        i 2 u 1:10:xtic(1) t 'COL (trove)' w lp ls 6, \
    ''        i 3 u 1:10:xtic(1) t 'ROW (trove)' w lp ls 8
