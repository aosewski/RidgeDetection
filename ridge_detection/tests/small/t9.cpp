
#include <iostream>
#include "tests/test_util.hpp"
#include "rd/utils/graph_drawer.hpp"

//------------------------------------------------------------
//  MAIN
//------------------------------------------------------------

int main()
{

    float a = 22.52f;
    float b = 11.31f;
    int pointCnt = int(1e7);
    int dim = 3;
    float stddev = 4.17f;
    
    std::cout << "spiral point cloud: "
        "--size=1e7 --a=22.52 --b=11.31 --stddev=4.17" << std::endl;

    PointCloud<float> && fpc = SpiralPointCloud<float>(a, b, pointCnt, dim, stddev);
    fpc.initializeData();
    std::cout << "data generated!" << std::endl;

    std::ostringstream outFileName;
    // outFileName << getCurrDateAndTime() << "_binary_out.bin";
    outFileName << "spiral_pc_np=1e7_a=22.52_b=11.31_stddev=4.17_d=3.bin";
    std::string outFilePath = rd::findPath("", outFileName.str());
    fpc.writeFile(outFilePath, fpc.points_, pointCnt, dim);
    std::cout << "data wrote!" << std::endl;

    std::vector<float> inBinData = fpc.readFile(outFilePath, pointCnt, dim);
    std::cout << "data read!" << std::endl;

    int result = CompareResult(inBinData.data(), fpc.points_.data(), pointCnt * dim);
    if (result)
    {
        std::cout << "FAIL!";
    }
    else
    {
        std::cout << "CORRECT!";
    }

    // if (dim <= 3) 
    // {
    //     std::cout << "\n drawing graph" << std::endl;
    //     rd::GraphDrawer<float> gDrawer;
    //     std::ostringstream graphName;

    //     graphName << getCurrDateAndTime() << "_"
    //         << "_np" << pointCnt
    //         << "_a" << a 
    //         << "_b" << b 
    //         << "_s" << stddev
    //         << "_point_cloud";

    //     std::string filePath = rd::findPath("", graphName.str());

    //     gDrawer.startGraph(filePath, dim);
    //     if (dim == 3)
    //     {
    //         gDrawer.setGraph3DConf();
    //     }

    //     gDrawer.addPlotCmd("'-' w p pt 1 lc rgb '#B8E186' ps 0.5 ",
    //          inBinData.data(), rd::GraphDrawer<float>::POINTS, pointCnt);
    //     gDrawer.endGraph();
    // }

    std::cout << "segment point cloud: "
        "--size=1e7 --segl=100 --stddev=2.17 --dim=12" << std::endl;

    float segLength = 100.0f;
    stddev = 2.17f;
    dim = 12;

    PointCloud<float> && fpc2 = SegmentPointCloud<float>(segLength, pointCnt, dim, stddev);
    fpc2.initializeData();
    std::cout << "data generated!" << std::endl;

    outFileName.clear();
    outFileName.str(std::string());

    outFileName << "segment_pc_np=1e7_segl=100_stddev=2.17_d=12.bin";
    outFilePath = rd::findPath("", outFileName.str());
    fpc2.writeFile(outFilePath, fpc2.points_, pointCnt, dim);
    std::cout << "data wrote!" << std::endl;

    inBinData = fpc2.readFile(outFilePath, pointCnt, dim);
    std::cout << "data read!" << std::endl;

    result = CompareResult(inBinData.data(), fpc2.points_.data(), pointCnt * dim);
    if (result)
    {
        std::cout << "FAIL!";
    }
    else
    {
        std::cout << "CORRECT!";
    }

    std::cout << "\nEND!" << std::endl;

    return EXIT_SUCCESS;
}

