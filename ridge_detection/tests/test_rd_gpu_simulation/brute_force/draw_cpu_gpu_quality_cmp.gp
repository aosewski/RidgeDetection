#!/usr/bin/gnuplot
reset

set terminal pngcairo size 520,400 enhanced notransparent font 'Verdana,13'

set border 3 back lc rgb '#808080' lt 1.5
set grid back lc rgb '#808080' lt 0 lw 1

# set style line 1 lt 1 lc rgb '#9ECAE1' # light blue
set style line 1 pt 1 lt 1 lw 1.5 lc rgb '#4292C6' # medium blue
set style line 2 pt 2 lt 1 lw 1.5 lc rgb '#2171B5' #
set style line 3 pt 3 lt 1 lw 1.5 lc rgb '#084594' # dark blue

# set style line 4 pt 1 lt 1 lw 1.5 lc rgb '#A1D99B' # light green
set style line 4 pt 4 lt 1 lw 1.5 lc rgb '#41AB5D' # medium green
set style line 5 pt 5 lt 1 lw 1.5 lc rgb '#238B45' #
set style line 6 pt 6 lt 1 lw 1.5 lc rgb '#005A32' # dark green

# set style line 7 pt 1 lt 1 lw 1.5 lc rgb '#BCBDDC' # light purple
set style line 7 pt 7 lt 1 lw 1.5 lc rgb '#807DBA' # medium purple
set style line 8 pt 8 lt 1 lw 1.5 lc rgb '#6A51A3' #
set style line 9 pt 9 lt 1 lw 1.5 lc rgb '#4A1486' # dark purple

# set style line 10 pt 1 lt 1 lw 1.5 lc rgb '#FDAE6B' # light orange
set style line 10 pt 10 lt 1 lw 1.5 lc rgb '#F16913' # medium orange
set style line 11 pt 11 lt 1 lw 1.5 lc rgb '#D94801' #
set style line 12 pt 12 lt 1 lw 1.5 lc rgb '#8C2D04' # dark orange

# set style line 13 pt 1 lt 1 lw 1.5 lc rgb '#ff6962' # light red
set style line 13 pt 13 lt 1 lw 1.5 lc rgb '#d21f47' # red
set style line 14 pt 14 lt 1 lw 1.5 lc rgb '#b0062c' #
set style line 15 pt 15 lt 1 lw 1.5 lc rgb '#8b0000' # dark red

# set style line 4 pt 1 lt 1 lw 1.5 lc rgb '#FA9FB5' # light red-purple
set style line 16 pt 16 lt 1 lw 1.5 lc rgb '#DD3497' # medium red-purple
set style line 17 pt 17 lt 1 lw 1.5 lc rgb '#AE017E' #
set style line 18 pt 18 lt 1 lw 1.5 lc rgb '#7A0177' # dark red-purple


set key inside right top vertical
set tics nomirror
set xtics format "10^{%T}"

set ylabel "odległość \\sigma" offset 1.5,0
set xlabel 'liczba punktów' offset 0,0.5
# set logscale y
set logscale x
# set tmargin 2
# set rmargin 7
set xrange[1e3:1e6]
sigma = 2.17

######################################################
# 
#   distance to point cloud size and dimensions   
#
######################################################

dataFile1 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/brute_force/gnuplot_data/gpu_cpu_quality_compare_segment_dist_dim_size.txt'

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/brute_force/img/gpu_quality_segment_dist_dim_size_median.png'

plot dataFile1 i  0 u 1:($12/(sigma)) t '2D' w lp ls 1, \
          ''  i  1 u 1:($12/(sigma)) t '3D' w lp ls 2, \
          ''  i  2 u 1:($12/(sigma)) t '4D' w lp ls 4, \
          ''  i  3 u 1:($12/(sigma)) t '5D' w lp ls 5, \
          ''  i  4 u 1:($12/(sigma)) t '6D' w lp ls 8, \
          ''  i  5 u 1:($12/(sigma)) t '7D' w lp ls 9, \
          ''  i  6 u 1:($12/(sigma)) t '8D' w lp ls 10, \
          ''  i  7 u 1:($12/(sigma)) t '9D' w lp ls 11, \
          ''  i  8 u 1:($12/(sigma)) t '10D' w lp ls 13, \
          ''  i  9 u 1:($12/(sigma)) t '11D' w lp ls 14, \
          ''  i 10 u 1:($12/(sigma)) t '12D' w lp ls 16

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/brute_force/img/gpu_quality_segment_dist_dim_size_hausdorff.png'

plot dataFile1 i  0 u 1:($11/(sigma)) t '2D' w lp ls 1, \
          ''  i  1 u 1:($11/(sigma)) t '3D' w lp ls 2, \
          ''  i  2 u 1:($11/(sigma)) t '4D' w lp ls 4, \
          ''  i  3 u 1:($11/(sigma)) t '5D' w lp ls 5, \
          ''  i  4 u 1:($11/(sigma)) t '6D' w lp ls 8, \
          ''  i  5 u 1:($11/(sigma)) t '7D' w lp ls 9, \
          ''  i  6 u 1:($11/(sigma)) t '8D' w lp ls 10, \
          ''  i  7 u 1:($11/(sigma)) t '9D' w lp ls 11, \
          ''  i  8 u 1:($11/(sigma)) t '10D' w lp ls 13, \
          ''  i  9 u 1:($11/(sigma)) t '11D' w lp ls 14, \
          ''  i 10 u 1:($11/(sigma)) t '12D' w lp ls 16


######################################################
# 
#   distance to point cloud size and radius  
#
######################################################

# set terminal pngcairo size 520,460 enhanced notransparent font 'Verdana,12'


# set key outside top center horizontal
# set ylabel "odległość \\sigma" offset 0,0
# set xlabel "promień \\sigma" offset 0,0.5

# set grid xtics ytics mxtics mytics back lc rgb '#808080' lt 0 lw 1

# # set xrange[1e3:1e6]

# set mxtics
# set mytics

# set logscale x
# set logscale y

# # set tmargin 2
# # set rmargin 7

# set ytics format "10^{%T}"
# unset xrange

# dataFile2 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
# test_rd_gpu_simulation/brute_force/gnuplot_data/gpu_cpu_quality_compare_segment_dist_size_radius.txt'

# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
# test_rd_gpu_simulation/brute_force/img/gpu_quality_segment_dist_radius_size_median.png'

# plot dataFile2 i  0 u ($3/(sigma)):($12/(sigma)) t '1 tys.'    w lp ls 1, \
#            ''  i  1 u ($3/(sigma)):($12/(sigma)) t '2 tys.'    w lp ls 2, \
#            ''  i  2 u ($3/(sigma)):($12/(sigma)) t '5 tys.'    w lp ls 4, \
#            ''  i  3 u ($3/(sigma)):($12/(sigma)) t '10 tys.'   w lp ls 5, \
#            ''  i  4 u ($3/(sigma)):($12/(sigma)) t '20 tys.'   w lp ls 7, \
#            ''  i  5 u ($3/(sigma)):($12/(sigma)) t '50 tys.'   w lp ls 8, \
#            ''  i  6 u ($3/(sigma)):($12/(sigma)) t '100 tys.'  w lp ls 10, \
#            ''  i  7 u ($3/(sigma)):($12/(sigma)) t '200 tys.'  w lp ls 11, \
#            ''  i  8 u ($3/(sigma)):($12/(sigma)) t '500 tys.'  w lp ls 13, \
#            ''  i  9 u ($3/(sigma)):($12/(sigma)) t '1 mln.'    w lp ls 14

# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
# test_rd_gpu_simulation/brute_force/img/gpu_quality_segment_dist_radius_size_hausdorff.png'

# plot dataFile2 i  0 u ($3/(sigma)):($11/(sigma)) t '1 tys.'    w lp ls 1, \
#            ''  i  1 u ($3/(sigma)):($11/(sigma)) t '2 tys.'    w lp ls 2, \
#            ''  i  2 u ($3/(sigma)):($11/(sigma)) t '5 tys.'    w lp ls 4, \
#            ''  i  3 u ($3/(sigma)):($11/(sigma)) t '10 tys.'   w lp ls 5, \
#            ''  i  4 u ($3/(sigma)):($11/(sigma)) t '20 tys.'   w lp ls 7, \
#            ''  i  5 u ($3/(sigma)):($11/(sigma)) t '50 tys.'   w lp ls 8, \
#            ''  i  6 u ($3/(sigma)):($11/(sigma)) t '100 tys.'  w lp ls 10, \
#            ''  i  7 u ($3/(sigma)):($11/(sigma)) t '200 tys.'  w lp ls 11, \
#            ''  i  8 u ($3/(sigma)):($11/(sigma)) t '500 tys.'  w lp ls 13, \
#            ''  i  9 u ($3/(sigma)):($11/(sigma)) t '1 mln.'    w lp ls 14


# set key outside top center horizontal
# set xlabel "liczba punktów" offset 0,0.5
# set grid xtics ytics mxtics mytics back lc rgb '#808080' lt 0 lw 1

# set xrange[1e3:1e6]

# set mxtics
# set mytics
# set logscale x
# set logscale y

# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
# test_rd_gpu_simulation/brute_force/img/gpu_quality_segment_dist_size_radius_median.png'

# plot dataFile2 i  10 u 1:($12/(sigma)) t 'R=0.217' w lp ls 1, \
#            ''  i  11 u 1:($12/(sigma)) t 'R=0.434' w lp ls 2, \
#            ''  i  12 u 1:($12/(sigma)) t 'R=1.085' w lp ls 3, \
#            ''  i  13 u 1:($12/(sigma)) t 'R=2.170' w lp ls 4, \
#            ''  i  14 u 1:($12/(sigma)) t 'R=2.604' w lp ls 5, \
#            ''  i  15 u 1:($12/(sigma)) t 'R=3.255' w lp ls 6, \
#            ''  i  16 u 1:($12/(sigma)) t 'R=3.906' w lp ls 7, \
#            ''  i  17 u 1:($12/(sigma)) t 'R=4.340' w lp ls 8, \
#            ''  i  18 u 1:($12/(sigma)) t 'R=6.510' w lp ls 9, \
#            ''  i  19 u 1:($12/(sigma)) t 'R=8.680' w lp ls 10, \
#            ''  i  20 u 1:($12/(sigma)) t 'R=10.850' w lp ls 11, \
#            ''  i  21 u 1:($12/(sigma)) t 'R=21.700' w lp ls 12

# set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
# test_rd_gpu_simulation/brute_force/img/gpu_quality_segment_dist_size_radius_hausdorff.png'

# plot dataFile2 i  10 u 1:($11/(sigma)) t 'R=0.217' w lp ls 1, \
#            ''  i  11 u 1:($11/(sigma)) t 'R=0.434' w lp ls 2, \
#            ''  i  12 u 1:($11/(sigma)) t 'R=1.085' w lp ls 3, \
#            ''  i  13 u 1:($11/(sigma)) t 'R=2.170' w lp ls 4, \
#            ''  i  14 u 1:($11/(sigma)) t 'R=2.604' w lp ls 5, \
#            ''  i  15 u 1:($11/(sigma)) t 'R=3.255' w lp ls 6, \
#            ''  i  16 u 1:($11/(sigma)) t 'R=3.906' w lp ls 7, \
#            ''  i  17 u 1:($11/(sigma)) t 'R=4.340' w lp ls 8, \
#            ''  i  18 u 1:($11/(sigma)) t 'R=6.510' w lp ls 9, \
#            ''  i  19 u 1:($11/(sigma)) t 'R=8.680' w lp ls 10, \
#            ''  i  20 u 1:($11/(sigma)) t 'R=10.850' w lp ls 11, \
#            ''  i  21 u 1:($11/(sigma)) t 'R=21.700' w lp ls 12
