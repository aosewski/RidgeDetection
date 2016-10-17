
#include <limits>
#include <iostream>

#ifndef RD_USE_OPENMP
    #define RD_USE_OPENMP
#endif

#include "tests/test_util.hpp"
#include "rd/utils/bounding_box.hpp"

#include <omp.h>

static constexpr int POINT_NUM = 1 << 16;
static constexpr int DIM = 3;

void bounds_single_thread(
    rd::BoundingBox<float> &    bb,
    float const *               data)
{
    //init bbox
    for (int k = 0; k < DIM; ++k)
    {
        bb.min(k) = std::numeric_limits<float>::max();
        bb.max(k) = std::numeric_limits<float>::lowest();
    }

    for (size_t k = 0; k < POINT_NUM; ++k)
    {
        for (size_t i = 0; i < DIM; ++i)
        {
            float val = data[DIM*k + i];
            bb.min(i) = (val < bb.min(i)) ? val : bb.min(i);
            bb.max(i) = (val > bb.max(i)) ? val : bb.max(i);
        }
    }
}

int main()
{
    
    PointCloud<float> && fpc3d = 
        SpiralPointCloud<float>(22.f, 10.f, POINT_NUM, DIM, 4.f);
    fpc3d.initializeData();

    rd::BoundingBox<float> bb_omp(fpc3d.points_.data(), POINT_NUM, DIM);
    
    std::cout << "OMP finished!\n" << std::endl;

    rd::BoundingBox<float> bb_single_thr;
    bb_single_thr.bbox = new float[2 * DIM];
    bb_single_thr.dim = DIM;

    bounds_single_thread(bb_single_thr, fpc3d.points_.data());

    std::cout << "single thread finished!\n" << std::endl;

    for (int k = 0; k < DIM * 2; ++k)
    {
        std::cout << "omp[" << k << "]: " << bb_omp.bbox[k] 
            << ",\tsingle[" << k << "]: " << bb_single_thr.bbox[k] << std::endl;
    }

    return 0;
}