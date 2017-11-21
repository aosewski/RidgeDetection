#!/usr/bin/gnuplot
reset

set terminal pngcairo size 720,480 enhanced notransparent font 'Verdana,12'


set style line  1 pt  1 lt 1 lw 1.5 lc rgb '#9ECAE1' # light blue
set style line  2 pt  2 lt 1 lw 1.5 lc rgb '#4292C6' # medium blue
set style line  3 pt  3 lt 1 lw 1.5 lc rgb '#2171B5' #
set style line  4 pt  4 lt 1 lw 1.5 lc rgb '#084594' # dark blue

set style line  5 pt  1 lt 1 lw 1.5 lc rgb '#A1D99B' # light green
set style line  6 pt  4 lt 1 lw 1.5 lc rgb '#41AB5D' # medium green
set style line  7 pt  1 lt 1 lw 1.5 lc rgb '#238B45' #
set style line  8 pt  6 lt 1 lw 1.5 lc rgb '#005A32' # dark green

set style line  9 pt  9 lt 1 lw 1.5 lc rgb '#BCBDDC' # light purple
set style line 10 pt 10 lt 1 lw 1.5 lc rgb '#807DBA' # medium purple
set style line 11 pt 11 lt 1 lw 1.5 lc rgb '#6A51A3' #
set style line 12 pt 12 lt 1 lw 1.5 lc rgb '#4A1486' # dark purple

set style line 13 pt 13 lt 1 lw 1.5 lc rgb '#FDAE6B' # light orange
set style line 14 pt 14 lt 1 lw 1.5 lc rgb '#F16913' # medium orange
set style line 15 pt 15 lt 1 lw 1.5 lc rgb '#D94801' #
set style line 16 pt 16 lt 1 lw 1.5 lc rgb '#8C2D04' # dark orange

set style line 17 pt  1 lt 1 lw 1.5 lc rgb '#ff6962' # light red
set style line 18 pt 18 lt 1 lw 1.5 lc rgb '#d21f47' # red
set style line 19 pt  1 lt 1 lw 1.5 lc rgb '#b0062c' #
set style line 20 pt 20 lt 1 lw 1.5 lc rgb '#8b0000' # dark red

set style line 21 pt 21 lt 1 lw 1.5 lc rgb '#FA9FB5' # light red-purple
set style line 22 pt 22 lt 1 lw 1.5 lc rgb '#DD3497' # medium red-purple
set style line 23 pt 23 lt 1 lw 1.5 lc rgb '#AE017E' #
set style line 24 pt 24 lt 1 lw 1.5 lc rgb '#7A0177' # dark red-purple

set style line 25 pt 0 lt 1 lw 1.5 lc rgb '#A1D99B' #
set style line 26 pt 0 lt 1 lw 1.5 lc rgb '#238B45' #
set style line 27 pt 0 lt 1 lw 1.5 lc rgb '#ff6962' #
set style line 28 pt 0 lt 1 lw 1.5 lc rgb '#b0062c' #

set border 3 back lc rgb '#808080' lt 1.5
set style fill solid 0.95 border rgb 'grey30'

set grid xtics ytics mxtics mytics back lc rgb '#808080' lt 0 lw 1
set key inside top left vertical 

set tics nomirror
set mxtics 10
set ytics format "10^{%T}"
set xtics format "10^{%T}"

set ylabel "ms"
set xlabel 'Liczba punktów'

set logscale y
set logscale x

set xrange[700:1e7]

######################################################
# 
#   time to point cloud dimension
#
######################################################

dataFile1 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/gnuplot_data/gpu_time_size_perf.txt'

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/img/gpu_time_size_perf1.png'
plot dataFile1 i  0 u 1:8 t 'Wersja siłowa GTX TITAN'  w lp ls 25, \
          ''   i  1 u 1:14 t 'Wersja kafelkowa GTX TITAN'  w lp ls 27, \
          ''   i  2 u 1:8 t 'Wersja siłowa GTX 750Ti'  w lp ls 26, \
          ''   i  3 u 1:14 t 'Wersja kafelkowa GTX 750Ti'  w lp ls 28, \
          ''   i  0 u 1:8:9:10 notitle w yerrorbars ls 5, \
          ''   i  1 u 1:14:15:16 notitle w yerrorbars ls 17,\
          ''   i  2 u 1:8:9:10 notitle w yerrorbars ls 7, \
          ''   i  3 u 1:14:15:16 notitle w yerrorbars ls 19
