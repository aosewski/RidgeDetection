#include <iostream>
#include <assert.h>

static constexpr int POINTS_NUM     = 1 << 20;
static constexpr int DIM            = 2;

__host__ __device__ __forceinline__ cudaError_t Debug(
    cudaError_t     error,
    const char*     filename,
    int             line)
{
    if (error)
    {
    #if (__CUDA_ARCH__ == 0)
        fprintf(stderr, "CUDA error %d [%s, %d]: %s\n",
            error, filename, line, cudaGetErrorString(error));
        fflush(stderr);
    #elif (__CUDA_ARCH__ >= 200)
        printf("CUDA error %d [block (%3d,%3d,%3d) thread (%3d,%3d,%3d), %s, %d]\n",
            error, blockIdx.z, blockIdx.y, blockIdx.x,
            threadIdx.z, threadIdx.y, threadIdx.x, filename, line);
    #endif
    }
    return error;
}

/**
 * @brief Macros for error checking.     
 */
#ifndef devCheckCall
    #define devCheckCall(e) if ( Debug((e), __FILE__, __LINE__) ) { assert(0); }
#endif

#ifndef checkCudaErrors
    #define checkCudaErrors(e) if ( Debug((e), __FILE__, __LINE__) ) { cudaDeviceReset(); exit(1); }
#endif

//--------------------------------------------------------------------------
// 
//--------------------------------------------------------------------------

static __device__ int treeNodeIdCounter = 0;

//--------------------------------------------------------------------------
//  Tree structures
//--------------------------------------------------------------------------

template <typename NodeT>
struct TreeRoot
{
    int childrenCount;
    NodeT *children;

    // __device__ __forceinline__ TreeRoot()
    // :
    //     childrenCount(0),
    //     children(nullptr)
    // {
    //     printf("TreeRoot(): childrenCount : %d\n", childrenCount);
    // }

    // __device__ __forceinline__ TreeRoot(
    //     int chCnt)
    // :
    //     childrenCount(chCnt)
    // {
    //     children = new NodeT[childrenCount];
    //     assert(children != nullptr);
    // }

    // __device__ __forceinline__ void init(
    //     int childrenNum)
    // {
    //     clear();
    //     childrenCount = childrenNum;
    //     children = new NodeT[childrenCount];
    //     assert(children != nullptr);
    // }

    // __device__ __forceinline__ ~TreeRoot()
    // {
    //     printf("~TreeRoot(), childrenCount: %d \n", childrenCount);
    //     clear();
    // }

    __device__ __forceinline__ bool empty() const
    {
        return childrenCount == 0 && children == nullptr;
    }

    __device__ __forceinline__ void clear()
    {
        printf("TreeRoot::clear() \n", childrenCount);

        if (!empty())
        {
            for (int i = 0; i < childrenCount; ++i)
            {
                children[i].clear();
            }

            delete[] children;
            children = nullptr;
        }
    }

};

template <int DIM, typename T>
struct TreeNode
{
    typedef TreeNode<DIM, T> NodeT;

    // some fields..
    int id;
    NodeT* parent;
    NodeT* left, *right;

    int treeLevel;
    int pointsCnt;
    T * samples;

    __device__ __forceinline__ TreeNode()
    :
        id(atomicAdd(&treeNodeIdCounter,1)),
        parent(nullptr),
        left(nullptr),
        right(nullptr),
        treeLevel(0),
        pointsCnt(0),
        samples(nullptr)
    {
        printf("TreeNode() id: %d\n", id);   
    }

    __device__ __forceinline__ ~TreeNode()
    {
        printf("~TreeNode() id: %d\n", id);
        clear();
    }

    __device__ __forceinline__ bool empty() const
    {
        return left == nullptr && right == nullptr;
    }

    __device__ __forceinline__ void clear()
    {
        printf("clear() id: %d, level: %d, withChildren: %d\n",
            id, treeLevel, !empty());

        if (samples != nullptr)         { delete[] samples; }
        if (left != nullptr)            { delete left; }
        if (right != nullptr)           { delete right; }
    }

    
    //----------------------------------------------------------------
    // Debug printing
    //----------------------------------------------------------------

    __device__ __forceinline__ void print() const
    {
        printf("Node id %d, pointsCnt: %d, treeLevel: %d\n", id, pointsCnt, treeLevel);
    }
};

//--------------------------------------------------------------------------
// Tree building algorithm
//--------------------------------------------------------------------------

template <
    int DIM, 
    typename T,
    typename LeafProcessingOpT>
static __global__ void addTreeRootNodesKernel(
    TreeNode<DIM, T> *  nodes,
    T const *           inputPoints,
    int                 inPointsNum,
    LeafProcessingOpT   leafProcessingOp)
{
    if (threadIdx.x == 0)
    {
        printf("Inside addTreeRootNodesKernel()!\n");
    }
    // data processing
}


template <int DIM, typename T>
class Tree
{

public:
    typedef TreeNode<DIM, T>    NodeT;
    typedef TreeRoot<NodeT>     RootT;

    int maxNodeCapacity;
    int* d_leafCount;

    __device__ __forceinline__ Tree(
        int maxCapacity)
    :
        maxNodeCapacity(maxCapacity),
        d_leafCount(nullptr)
    {
        printf("Tree()\n");
    }

    __device__ __forceinline__ ~Tree()
    {
        if (d_leafCount != nullptr)
        {
            delete d_leafCount;
        }
        // if (root.children != nullptr)
        // {
        //     delete[] root.children;
        // }
    }

    template <typename LeafProcessingOpT>
    __device__ __forceinline__ void buildTree(
        T const *           inputPoints,
        int                 inPointsNum,
        int                 initNodeCnt,
        cudaStream_t        stream,
        LeafProcessingOpT   leafProcessingOp)
    {
        // initialize leaf counter
        if (d_leafCount == nullptr)
        {
            d_leafCount = new int(0);
            assert(d_leafCount != nullptr);
        }
        // initialize root
        // root.init(initNodeCnt);

        // NodeT * children = new NodeT[initNodeCnt];
        root.childrenCount = initNodeCnt;
        root.children = new NodeT[initNodeCnt];
        assert(root.children != nullptr);

        // prepare parameters
        T * d_neededStorage = new T[initNodeCnt];
        assert(d_neededStorage != nullptr);

        devCheckCall(cudaMemsetAsync(d_neededStorage, 0, initNodeCnt * sizeof(T), stream));
        devCheckCall(cudaDeviceSynchronize());

        printf("Invoking addTreeRootNodesKernel()\n");

        dim3 gridSize(initNodeCnt);
        addTreeRootNodesKernel<<<gridSize, 128, 0, stream>>>(
            root.children, 
            // children, 
            inputPoints, 
            inPointsNum, 
            leafProcessingOp);

        devCheckCall(cudaPeekAtLastError());
        devCheckCall(cudaDeviceSynchronize());

        // delete[] children;
    }

private:
    RootT root;

};

//------------------------------------------------------------------------
// Test kernels
//------------------------------------------------------------------------

struct LeafProcessOp
{
    template <typename NodeT>
    __device__ __forceinline__ void operator()(NodeT const * node) const
    {
        if (threadIdx.x == 0)
        {
           node->print();
        }
    }
};

template <
    typename                TreeT,
    typename                LeafProcessOpT,
    typename                T>
__launch_bounds__ (1)
static __global__ void buildTreeKernel(
    T const *   inputPoints,
    int         pointsNum,
    int         maxTileCapacity,
    int         initNodeCnt)
{
    TreeT * tree = new TreeT(maxTileCapacity);
    assert(tree != nullptr);

    LeafProcessOpT leafProcessOp;
    cudaStream_t buildTreeStream;
    devCheckCall(cudaStreamCreateWithFlags(&buildTreeStream, cudaStreamNonBlocking));

    tree->buildTree(inputPoints, pointsNum, initNodeCnt, buildTreeStream, leafProcessOp);

    devCheckCall(cudaDeviceSynchronize());
    devCheckCall(cudaStreamDestroy(buildTreeStream));
    delete tree;
}

//--------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------

int main(void)
{

    float *h_data = new float[POINTS_NUM * DIM];
    float *d_in;

    checkCudaErrors(cudaSetDevice(0));

    checkCudaErrors(cudaMalloc(&d_in, POINTS_NUM * DIM * sizeof(float)));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            h_data[k * DIM + d] = k * 0.1f + d * 0.03f;
        }
    }

    checkCudaErrors(cudaMemcpy(d_in, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice));

    // checkCudaErrors(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 12));

    buildTreeKernel<Tree<DIM, float>, LeafProcessOp><<<1, 1>>>(d_in, POINTS_NUM, 5000, 10);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "\nEND!" << std::endl;

    checkCudaErrors(cudaDeviceReset());

    return 0;
}