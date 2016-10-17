# General project overview

This project is an integral part of the master thesis entitled:  

>Design of the parallel version of the ridge detection algorithm for the multidimensional 
>random variable density function and its implementation in CUDA technology.

The original and sequential version of algorithm was designed by Marek Rupniewski and published here:  

W. Rupniewski M. (2014). **Curve Reconstruction from Noisy and Unordered Samples.** In 
*Proceedings of the 3rd International Conference on Pattern Recognition Applications and Methods - 
Volume 1: ICPRAM*, pages 183-188. [DOI: 10.5220/0004814801830188](http://www.scitepress.org/DigitalLibrary/PublicationsDetail.aspx?ID=WVDPwlh33pE=&t=1)

As it is stated in the paper title, the algorithm was designed with curve reconstruction in mind. 
The input data is in general a multidimensional, unordered and noisy point cloud, which usually 
come from a measurement device. Additionally it is assumed that the points are independent. The 
output of the algorithm is a spatially ordered set of points. When we connect each consecutive pair of points from this set in result we will get a polygonal path that approximates 
reconstructed curve.

## Algorithm idea


![Ridge and point cloud](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/ridge.png "Ridge and point cloud")

The word ridge is of course connected with mountain range as it is a long narrow path that goes 
through the highest located points. This definition well resembles the one used in algorithm 
because the ridge is identified with reconstructed curve. We 
assume that our input point cloud is constructed in such a way, that the closer to the original 
curve the higher is the probability of point occurrence (see above picture). We may describe this 
phenomena with some probability density function, which gets highest values exactly where the 
ridge goes through. In the image below in the upper left corner there is a simplified 
two-dimensional case. The red dot marks point where ridge goes through. We can see, that the 
greater the distance from the red point is the lower the probability of cloud point occurrence is. 
This the basic fact that is utilized and that forms the foundations of ridge detection algorithm.

![Ridge detection algorithm idea](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/rd_alg.png "Ridge detection algorithm idea")

As it is seen on above picture, the first step in algorithm is initialization of a set of balls 
centered on a input cloud points. This set is called *chosen points*. The figure of probability 
density function is upside down only to better visualize the process of chosen points convergence 
to ridge. Next, there are two phases repeated in a loop until the *chosen points* set stabilize. 
In evolution stage we move points toward detected ridge and in decimation we reduce redundant 
points according to given criteria. In the end we get only points lying exactly on *red point*, that is detected ridge, which approximates reconstructed curve.

## Brute-force version

## Tiled version

<img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/build_tree_scheme.png" title="Building hierarchical tree structure" width="800">

<img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/2D_ETT_MR_mtc-10000_ntpd-[3_3]_np-100000_mtiles.png" title="Extended tiles" width="600">

<img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/2D_GT_LR_mtc-10000_ntpd-[3_3]_np-100000_tid_16.png" title="Gruped tiles" width="600">

<img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/2D_LT_LR_tiled_tree1.png" title="Example of build tree structure" width="600">


![Alt text](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/3D_GT_LR_tiled_tree.png "Example of build tree structure")

![Alt text](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/3D_LT_LR_tiled_tree.png "Example of build tree structure")

## Results

![Alt text](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/gpu_time_size_perf3.png "Performance comparison")

# API & usage
