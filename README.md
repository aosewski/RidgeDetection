# General project overview

This project is an integral part of the master thesis entitled:  

>Design of the parallel version of the ridge detection algorithm for the multidimensional 
>random variable density function and its implementation in CUDA technology.

You can find the accompanying text [here](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/thesis/Adam_Rogowiec_Detekcja_Grani_2016.pdf "Adam Rogowiec Detekcja Grani"), 
however it's written in polish. Below you can find some basic information about algorithm and its
implementation.

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

In original, sequential version algorithm follows brute-force (exhaustive) strategy during the
most computationally intensive phase - evolution. Let us give a little bit of context how it is done.
Each point from the *chosen-points* set is moved towards the mass centre of intersection of its 
ball with the appropriate Voronoi diagram's cell created for *chosen-points* set. In order to 
accomplish this task we need to calculate distance from each pair of points, where one is from the
*chosen-points* set and another is from input point cloud data set. We accelerated this part on 
the multicore CPUs using OpenMP directives. 
In version which uses GPU to accelerate computations we offload all three major parts of ridge 
detection algorithm to GPU. That is the *chosen points* set initialization, the evolution and the 
decimation phases. Although the first and the last of them doesn't show great opportunities to 
speed up computations, our motivation for this decision was to minimize data transfer overhead
between host and device.

## Tiled version

However brute-force strategy exhibits huge data-parallelism, thus perfectly fits into modern 
many-core GPUs architecture, the great majority of calculations are redundant. In order to 
alleviate this problem we may divide our point cloud space into n-dimensional tiles. Using such
tiles we have few possibilities of how to organize data structures and decompose computations. 
It is worth to emphasize that, at the moment, inside each tile there is still brute-force strategy 
used.

Tile data structures:  
- **Local tiles**  
   <img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/2D_LT_LR_tiled_tree1.png" title="Local tiles" width="500">  
   On above image, dark green colour marks respective tiles bounds, and red colour marks cloud points.  
   Local tiles store information only about points falling into its bounds.

- **Grouped tiles**  
   <img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/2D_GT_LR_mtc-10000_ntpd-[3_3]_np-100000_tid_16.png" title="Grouped tiles" width="500">  
   On above image the selected tile is marked out with red colour border. The points lying 
   inside this tile's bounds are painted with light blue. The considered tile neighbours' points 
   are painted light green and neighbouring tiles' borders with light blue.  
   Grouped tiles, in comparison to local tiles, store additional information about its neighbouring
   tiles.

- **Extended tiles**  
   <img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/2D_ETT_MR_mtc-10000_ntpd-[3_3]_np-100000_mtiles.png" title="Extended tiles" width="500">  
   On above image the selected tiles are marked out with red colour border and the points lying 
   inside this tile's bounds are painted with light blue. With green colour there are marked respective tile neighbouring points.  
   Extended tiles are modification of grouped tiles. In order to reduce amount of computations even more we restrict each tile neighbourhood to only some fixed distance.

The process of partitioning point cloud space into tiles introduce additional level of 
parallelism, since we can build each tile independently from other tiles. Therefore we use 
recursive algorithm utilizing CUDA Dynamic Parallelism on GPUs to build hierarchical bounding 
volumes tree data structure, which is outlined on the picture below.

<img src="https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/build_tree_scheme.png" title="Building hierarchical tree structure" width="700">

Here are additional two examples of built tree structure for 3D case:

![](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/3D_GT_LR_tiled_tree.png "Example of build tree structure with grouped tiles")

![](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/tiles/3D_LT_LR_tiled_tree.png "Example of build tree structure with local tiles")

Ridge detection computation decomposition with tiled tree data structure:  
- **Local ridge detection** - with this scheme we perform ridge detection within each tile 
independently and asynchronously to others. This means that while in some parts of the point cloud
space there may still be ongoing partitioning process, tiles which by this time finished it may be 
already performing given tasks (in this project ridge detection), marked with *Process tile* label 
in previously presented scheme. It is important to accent that, ridge detection is performed *only*
 on points falling into particular tile bounds. Due to this fact the final results often contains 
a lot of redundant points close to the tiles borders, thus distorting reconstructed curve. The 
solution for this problem is adding additional global results refinement phase in the end. Unfortunately this significantly prolongs execution time.

- **Hybrid (mixed) ridge detection** - with this scheme there are two phases of ridge detection 
algorithm: local and global. Building tiles, initialization of *chosen-points* set within each 
tile and evolution is performed locally (on data assigned to particular tile), independently to 
others. After each evolution there is a synchronization point in order to perform global, that is 
on all chosen-points sets from all tiles altogether, decimation phase. Shortly speaking such 
organization of calculations let us achieve better (globally) approximation of reconstructed curve. Therefore there is no need to perform additional (very costly) refinement phase in the end.

## Results

During the tests it turned out that the most efficient in terms of both execution time and quality 
of results is mixed ridge detection with use of extended tiles.

The picture below presents comparison of different ridge detection algorithm implementations execution 
time depending on input data set size. The input data was a generated three-dimensional spiral 
point cloud (with added noise). The tiled version is the one using extended tiles and a hybrid 
(mixed) ridge detection scheme.

![](https://github.com/arogowiec/RidgeDetection/blob/devel/ridge_detection/doc/gpu_time_size_perf3.png "Performance comparison")

We may notice that we achieve linear execution time scalability with increasing number of cloud points. Furthermore we get almost two order of magnitude speed-up over multithreaded version on CPU (8 threads). Finally, we may observe that the tiled version need sufficiently big input data set to amortise the cost of building a tree.

# Usage

The project has a structure of header-only library, since it makes it easy to use and secondly 
because of heavy usage of C++ templates. The algorithmic core is placed under *rd* directory, 
where user can find *cpu* and *gpu* directories with implementations specialized for appropriate
 device. One can find example usage inside *test_rd_cpu_simulation* and *test_rd_gpu_simulation* 
subdirectories of *tests* directory. However there are benchmarks of almost all possible combinations of algorithm parameters. Below is a short example of the simplest tiled ridge 
detection algorithm usage:

```c++
#include "rd/gpu/device/tiled/simulation.cuh" 

...
// input data acquisition, memory allocation etc.
...

using namespace rd::gpu::tiled;

RidgeDetection<
    DIM,                        // DIM - indicates input data points dimensionality
    rd::ROW_MAJOR,              // input data layout in GPU memory
    rd::ROW_MAJOR,              // output data layout in GPU memory
    RD_BRUTE_FORCE,             // ridge detection algorithm used within tile, currently only RD_BRUTE_FORCE
    RD_MIXED,                   // tiled ridge detection policy
    RD_EXTENDED_TILE,           // used tile type
    float> rdGpu(
        pointsNum,              // number of input cloud points
        inputPoints,            // pointer to host memory with input data, must be in ROW_MAJOR order
        enableTiming,           // bool flag indicating whether or not to enable execution time measurements
        debugSynchronous);      // bool flag indicating whether or not to synchronize after each kernel launch for debug purposes

...

rdGpu(
    r1,                         // ridge detection algorithm parameter for initializing chosen points set
    r2,                         // ridge detection algorithm parameter for reducing redundant points
    maxTileCapacity,            // maximum number of points (inclusive) each tile can contain
    dimTiles,                   // an array containing number of initial tiles to partition space onto
    endPhaseRefinement);        // whether or not to perform additional final results refinement
chosenPointsNum = rdGpu.getChosenPointsNum();
float * data = new float[chosenPointsNum];
rdGpu.getChosenPoints(data);
...
```

The `rd::gpu::tiled::RidgeDetection` class allocates all necessary memory on GPU, launches computations and synchronizes with GPU waiting for it to finish.