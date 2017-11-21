#!/usr/bin/gnuplot
reset

# set terminal postscript color solid enhanced lw 5 "Helvetica"
set terminal pngcairo size 960,540 enhanced notransparent font 'Verdana,12'

# ------------------------------------------------------------------
#   I/O
# ------------------------------------------------------------------

dataFile1 = '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/gnuplot_data/gpu_time_tsize_ntpdim_mtx.txt'

# ------------------------------------------------------------------
#   Style definitions
# ------------------------------------------------------------------

# '#D73027' # red
# '#F46D43' # orange
# '#FDAE61' # 
# '#FEE090' # pale orange
# '#E0F3F8' # pale blue
# '#ABD9E9' # 
# '#74ADD1' # medium blue
# '#4575B4' # blue

# palette
set palette defined ( \
          0 '#D73027',\
          1 '#F46D43',\
          2 '#FDAE61',\
          3 '#FEE090',\
          4 '#E0F3F8',\
          5 '#ABD9E9',\
          6 '#74ADD1',\
          7 '#4575B4' )

# ------------------------------------------------------------------
#   Graph options
# ------------------------------------------------------------------

unset key
unset border

set xrange[2.5:5.5]
set yrange[2.5:27.5]

set cbtics nomirror
set palette negative

set cblabel "ms"

# ------------------------------------------------------------------
#   Macros
# ------------------------------------------------------------------

# Enable the use of macros
set macros

NOXTICS = "unset xtics; unset xlabel;"
XTICS = "set xtics nomirror out ('3' 3, '4' 4, '5' 5); set xlabel 'Liczba kafelków na wymiar' offset 0,0.5;"

NOYTICS = "unset ytics; unset ylabel;"
YTICS = "set ytics nomirror; set ylabel 'Max. poj. kafelka [tys. pkt.]' offset 1.5,0;"

# ------------------------------------------------------------------
#   plotting
# ------------------------------------------------------------------

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/img/gpu_time_tsize_ntpdim_mtx_gtx750ti.png'
set multiplot layout 1,2

@XTICS @YTICS
set label 1 '2D' at graph 0.92,0.97 center font 'Verdana,14' front
plot dataFile1 i 0 u ($1+3):(($2+1)*5):3 matrix with image

@XTICS @YTICS
set label 1 '3D' at graph 0.92,0.97 center font 'Verdana,14' front
plot dataFile1 i 1 u ($1+3):(($2+1)*5):3 matrix with image

unset multiplot

set output '/media/adamo/Dane_ubuntu/adamo/repozytoria/mgr/ridge_detection/tests/\
test_rd_gpu_simulation/tiled/img/gpu_time_tsize_ntpdim_mtx_gtxTitan.png'
set multiplot layout 1,2

@XTICS @YTICS
set label 1 '2D' at graph 0.92,0.97 center font 'Verdana,14' front
plot dataFile1 i 2 u ($1+3):(($2+1)*5):3 matrix with image

@XTICS @YTICS
set label 1 '3D' at graph 0.92,0.97 center font 'Verdana,14' front
plot dataFile1 i 3 u ($1+3):(($2+1)*5):3 matrix with image

unset multiplot
