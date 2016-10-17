
#include <iostream>

static constexpr int POINTS_NUM     = 1 << 20;
static constexpr int DIM            = 2;

static __device__ int treeNodeIdCounter = 0;

//--------------------------------------------------------------------------
//  Tree structures
//--------------------------------------------------------------------------

template <typename NodeT>
struct TreeRoot
{
    int childrenCount;
    NodeT *children;

    __device__ __forceinline__ TreeRoot()
    :
        childrenCount(0),
        children(nullptr)
    {
    }

    __device__ __forceinline__ void init(
        int childrenNum)
    {
        clear();
        childrenCount = childrenNum;
        children = new NodeT[childrenCount];
    }

    __device__ __forceinline__ ~TreeRoot()
    {
        clear();
    }

    __device__ __forceinline__ bool empty() const
    {
        return childrenCount == 0 && children == nullptr;
    }

    __device__ __forceinline__ void clear()
    {
        if (!empty())
        {
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
    NodeT* parent;
    NodeT* left, *right;

    int id;
    int treeLevel;
    int pointsCnt;
    T * samples;

    __device__ __forceinline__ TreeNode()
    :
        parent(nullptr),
        left(nullptr),
        right(nullptr),
        id(atomicAdd(&treeNodeIdCounter, 1)),
        treeLevel(0),
        pointsCnt(0),
        samples(nullptr)
    {
    }

    __device__ __forceinline__ ~TreeNode()
    {
        clear();
    }

    __device__ __forceinline__ bool empty() const
    {
        return left == nullptr && right == nullptr;
    }

    __device__ __forceinline__ void clear()
    {
        if (samples != nullptr)         { delete[] samples; }
        if (left != nullptr)            { delete left; }
        if (right != nullptr)           { delete right; }
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
    leafProcessingOp(nodes + blockIdx.x);
    // data processing
}


template <int DIM, typename T>
class Tree
{

public:
    typedef TreeNode<DIM, T>    NodeT;
    typedef TreeRoot<NodeT>     RootT;

    int maxNodeCapacity;

    __device__ __forceinline__ Tree(
        int maxCapacity)
    :
        maxNodeCapacity(maxCapacity)
    {
    }

    __device__ __forceinline__ ~Tree()
    {
    }

    template <typename LeafProcessingOpT>
    __device__ __forceinline__ void buildTree(
        T const *           inputPoints,
        int                 inPointsNum,
        int                 initNodeCnt,
        cudaStream_t        stream,
        LeafProcessingOpT   leafProcessingOp)
    {
        // initialize root
        root.init(initNodeCnt);

        // prepare parameters
        T * d_neededStorage = new T[initNodeCnt];
        cudaMemsetAsync(d_neededStorage, 0, initNodeCnt * sizeof(T), stream);
        cudaDeviceSynchronize();

        addTreeRootNodesKernel<<<initNodeCnt, 128, 0, stream>>>(
            root.children, 
            inputPoints, 
            inPointsNum, 
            leafProcessingOp);

        cudaPeekAtLastError();
        cudaDeviceSynchronize();
    }

private:
    RootT root;

};

//------------------------------------------------------------------------
// Test kernels
//------------------------------------------------------------------------

template <
    typename                TreeT,
    typename                T>
__launch_bounds__ (1)
static __global__ void buildTreeKernel(
    T const *   inputPoints,
    int         pointsNum,
    int         maxTileCapacity,
    int         initNodeCnt)
{
    TreeT tree(maxTileCapacity);
    tree.buildTree(inputPoints, pointsNum, initNodeCnt, nullptr, 
        [](typename TreeT::NodeT * node)
    {
        if (threadIdx.x == 0)
        {
            printf("Inside leafProcessingOp! node id: %d\n", node->id);
        }
    });

    cudaDeviceSynchronize();
}

//--------------------------------------------------------------------------
// main
//--------------------------------------------------------------------------

int main(void)
{

    float *h_data = new float[POINTS_NUM * DIM];
    float *d_in;

    cudaSetDevice(0);

    cudaMalloc(&d_in, POINTS_NUM * DIM * sizeof(float));

    for (int k = 0; k < POINTS_NUM; ++k)
    {
        for (int d = 0; d < DIM; ++d)
        {
            h_data[k * DIM + d] = k * 0.1f + d * 0.03f;
        }
    }

    cudaMemcpy(d_in, h_data, POINTS_NUM * DIM * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    buildTreeKernel<Tree<DIM, float>><<<1, 1>>>(d_in, POINTS_NUM, 5000, 10);
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaDeviceReset();

    std::cout << "END" << std::endl;

    return 0;
}
