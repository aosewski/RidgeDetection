/**
 * @file tree_node.hpp
 * @author     Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of
 * estimation of multidimensional random variable density function ridge
 * detection algorithm."
 * , which is supervised by prof. dr hab. inż. Marek Nałęcz.
 *
 * Institute of Control and Computation Engineering Faculty of Electronics and
 * Information Technology Warsaw University of Technology 2016
 */

#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

#include <vector>
#include <memory>
#include <iostream>
#include <string>

namespace rd
{
namespace tiled
{
    
    /**
 * @brief      Tree Node
 *
 * @tparam     ExternalData  Type of data which can be attached to node.
 */
template <typename ExternalData>
struct TreeNode
{
    std::shared_ptr<TreeNode> parent;
    std::vector<std::shared_ptr<TreeNode>> children;
    std::shared_ptr<ExternalData> data;

    #ifdef RD_DEBUG
        int id;
        int treeLevel;
        static int objCounter;
        static int idCounter;
    #endif

    TreeNode() 
    {
        #ifdef RD_DEBUG
            treeLevel = 0;
            id = idCounter++;
            objCounter++;
                std::cout << "TreeNode() id: " << id << " objCounter: " << objCounter << std::endl;
        #endif
    }

    ~TreeNode()
    {
        #ifdef RD_DEBUG
            std::cout << std::string(treeLevel, '\t')
                    << "~TreeNode() id: " << id << ", level: " << treeLevel;
            if (!children.empty()) std::cout << ", with children: " << children.size();
            if (data) std::cout << ", with data";
            std::cout << ", parent cnt: " << parent.use_count()
                    << ", NodeCounter: " << objCounter - 1;
            std::cout << std::endl;
        #endif
        if (!children.empty())
        {
            for (std::shared_ptr<TreeNode> &n : children)
            {
                n->clear();
            }
            #ifdef RD_DEBUG
                std::cout << "clearing children:" << std::endl;
            #endif
            children.clear();
        }
        #ifdef RD_DEBUG
            objCounter--;
        #endif
    }

    void clear()
    {
        #ifdef RD_DEBUG
            std::cout << std::string(treeLevel, '\t')
                    << "clear() id: " << id << ", level: " << treeLevel << "\n";
            if (!children.empty()) std::cout << ", with children: " << children.size() << "\n";
        #endif
        if (!children.empty())
        {
            for (std::shared_ptr<TreeNode> &n : children)
            {
                n->clear();
            }
            #ifdef RD_DEBUG
                std::cout << "clearing children:" << std::endl;
            #endif
            children.clear();
        }
    }

    void printRecursive()
    {
        print();
        if (!children.empty())
            for (std::shared_ptr<TreeNode> const &n : children)
                n->print();
    }

    void print()
    {
        
        #ifdef RD_DEBUG
            std::cout << std::string(treeLevel, '\t') 
                << "TreeNode id : " << id << ", level: "  << treeLevel
                << ", NodeCounter: " << objCounter << "\n";
        #endif
        if (!children.empty()) std::cout << ", with children: " << children.size();
        if (data) std::cout << ", with data";
        std::cout << ", parent cnt: " << parent.use_count() << "\n";
        if (data)
            data->print();
    }
};

#ifdef RD_DEBUG
    template <typename ExternalData>
    int TreeNode<ExternalData>::objCounter = 0;
    template <typename ExternalData>
    int TreeNode<ExternalData>::idCounter = 0;
#endif

template <typename NodeType>
struct TreeSentinel
{
    std::shared_ptr<NodeType> root;

    ~TreeSentinel()
    {
        #ifdef RD_DEBUG
            std::cout << "~TreeSentinel()" << std::endl;
        #endif
    }
};


}   // end namespace tiled
}   // end namespace rd

#endif // TREE_NODE_HPP