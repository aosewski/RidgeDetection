/**
 * @file rd_simulation_vis.inl
 * @author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of 
 *  estimation of multidimensional random variable density function ridge
 *  detection algorithm.",
 * which is supervised by prof. dr hab. inż. Marek Nałęcz.
 * 
 * Institute of Control and Computation Engineering
 * Faculty of Electronics and Information Technology
 * Warsaw University of Technology 2016
 */

#include "../../utils/utilities.hpp"

#include <iostream>
#include <limits>
#include <stdexcept>

namespace rd
{
    
namespace vis
{


template <typename T>
RDSimVis::RDSimVis()
{
    stateDesc_ = nullptr;
    ompNumThreads_ = 0;
    #if defined(RD_USE_OPENMP)
        ompNumThreads_ = omp_get_num_procs();
        omp_set_num_threads(ompNumThreads_);
        std::cout << "Default using " << ompNumThreads_ << " CPU threads" << std::endl;
    #endif

    phase = STOP;
}

template <typename T>
RDSimVis::~RDSimVis()
{
    if (stateDesc_ != nullptr)   delete stateDesc_;
}

/**
 * @brief      Start choose phase of ridge detection algorithm.
 */
template <typename T>
void RDSimVis::choose()
{
    ChoosePhaseDesc<T> *cd;

    // Prepare first algorithm phase: chose
    if (stateDesc_ == nullptr)
    {
        stateDesc_ = new ChosePhaseDesc<T>();
        cd = dynamic_cast<ChosePhaseDesc*>(stateDesc_);
        cd.idx++;
        cd.currPoint = P_ += dim_;
        cd.r2 = r1_ * r1_;

        // copy first point from P_ to S_
        copyTable(P_, S_, dim_);
        cd.chosenCount++;

        SList_.clear();
        SList_.push_back(S_);
        /*
         * ADD OPENGL DRAWING CALLBACK HERE
         * draw chosen points
         */
    }

    cd = dynamic_cast<ChosePhaseDesc*>(stateDesc_);

    while (doStep(cd))
    {
    }
}

/**
 * @brief      Start evolve phase of ridge detection algorithm.
 */
template <typename T>
void RDSimVis::evolve()
{
    EvolvePhaseDesc<T> *ed = dynamic_cast<EvolvePhaseDesc<T>*>(stateDesc_);   
    while (doStep(ed))
    {
    }
}

/**
 * @brief      Start decimate phase of ridge detection algorithm.
 */
template <typename T>
void RDSimVis::decimate()
{

    DecimatePhaseDesc<T> dd = dynamic_cast<DecimatePhaseDesc<T>*(stateDesc_);
    while (doStep(dd)) 
    {
    }
}

/**
 * @brief       Plays whole algorithm simulation.
 */
template <typename T>
void RDSimVis::run()
{
    finish();
}

/**
 * @brief      Performs one step of ridge detection algorithm.
 * 
 * The step depends on the algorithm current phase.
 */
template <typename T>
void RDSimVis::step()
{

    switch (phase)
    {
        case STOP:
            // Prepare first algorithm phase: chose
            if (stateDesc_ == nullptr)
            {
                stateDesc_ = new ChosePhaseDesc<T>();
                ChoosePhaseDesc<T> *cd = dynamic_cast<ChosePhaseDesc*>(stateDesc_);
                cd.idx++;
                cd.currPoint = P_ += dim_;
                cd.r2 = r1_ * r1_;

                // copy first point from P_ to S_
                copyTable(P_, S_, dim_);
                cd.chosenCount++;

                SList_.clear();
                SList_.push_back(S_);
                /*
                 * ADD OPENGL DRAWING CALLBACK HERE
                 * draw chosen points
                 */
                phase = CHOSE;
            }
            doStep(dynamic_cast<ChosePhaseDesc*>(stateDesc_));
            break;
        case CHOSE:
            doStep(dynamic_cast<ChosePhaseDesc*>(stateDesc_));
            break;
        case EVOLVE:
            doStep(dynamic_cast<EvolvePhaseDesc<T>*>(stateDesc_));
            break;
        case DECIMATE:
            doStep(dynamic_cast<DecimatePhaseDesc<T>*>(stateDesc_));
            break;
        default:
            throw std::logic_error("Unsupported phase!");
    }
}

template <typename T>
bool RDSimVis::doStep(ChosePhaseDesc *desc)
{
    if (desc.idx++ < np_)
    {
         // check whether there is no points in chosen from which the squared
         // distance to 'point' is no bigger than r2
        if (!countNeighbouringPoints(S_, desc.chosenCount, desc.currPoint,
             dim_, desc.r2, 1)) {
            copyTable(desc.currPoint, S_ + dim_ * desc.chosenCount, dim_);
            SList_.push_back(S_ + dim_ * desc.chosenCount++);
            /*
             * ADD OPENGL DRAWING CALLBACK HERE
             * draw chosen points
             */
        }
        desc.currPoint += dim_; 
    }
    else
    {
        // finish chose phase and create new descriptor
        ns_ = stateDesc_.chosenCount;
        delete stateDesc_;
        stateDesc_ = new EvolvePhaseDesc<T>();
        EvolvePhaseDesc<T> *ed = dynamic_cast<EvolvePhaseDesc<T>*>(stateDesc_);
        ed.r2 = r1_ * r1_;
        ed.chosenCount = ns_;
        ed.cordSums = new T[ns_ * dim_];
        ed.spherePointCount = new int[ns_];
        ed.sphereCellIdx = new int[np_];
        phase = EVOLVE;
        return false;
    }
    return true;
}

template <typename T>
bool RDSimVis::doStep(EvolvePhaseDesc *desc)
{

    if (!desc.contFlag)
    {
        delete stateDesc_;
        stateDesc_ = new DecimatePhaseDesc<T>();
        DecimatePhaseDesc<T> *dd = dynamic_cast<DecimatePhaseDesc<T>*>(stateDesc_);
        dd.chosenCount = ns_;
        dd.prevChosenCount = ns_;
        dd.r2 = r2_ * r2_;
        dd.decimateIdx = new int[ns_];
        fillTable(desc.decimateIdx, 0, ns_);
        phase = DECIMATE;
        return false;
    }

    switch (desc.part)
    {
        case EvolvePhaseDesc::SPHERE_CELLS_DIAGRAM:
            #if defined(RD_USE_OPENMP)
            #pragma omp parallel num_threads(ompNumThreads_)
                {
            #endif
            fillTable(desc.cordSums, T(0), ns_ * dim_);
            fillTable(desc.spherePointCount, int(0), ns_);
            fillTable(desc.sphereCellIdx, int(-1), np_);
            #if defined(RD_USE_OPENMP)
                #pragma omp for schedule (static)
            #endif
                // through all points in P
                for (size_t n = 0; n < np_; n++)
                {
                    T *sPtr = S_;
                    T minSquareDist = std::numeric_limits<T>::max();
                    T sqDist;
                    int minSIndex = -1;
                    T dist = 0;

                    // through all chosen points
                    for (size_t k = 0; k < ns_; k++) 
                    {
                        sqDist = 0;
                        // through point dimensions
                        for (size_t d = 0; d < dim_; d++, sPtr++) 
                        {
                            dist = *sPtr - P_[n * dim_ + d];
                            sqDist += dist * dist;
                        }
                        if (sqDist < minSquareDist) 
                        {
                            minSIndex = (int)k;
                            minSquareDist = sqDist;
                        }
                    }
                    if (minSquareDist <= r2) 
                    {
                #if defined(RD_USE_OPENMP)
                    #pragma omp critical
                    {
                        desc.spherePointCount[minSIndex]++;
                        for (size_t d = 0; d < dim_; d++) 
                        {
                            desc.cordSums[minSIndex * dim_ + d] += P_[n * dim_ + d];
                        }
                    }
                #else
                    desc.spherePointCount[minSIndex]++;
                    // sum point coordinates for later mass center calculation
                    for (size_t d = 0; d < dim_; d++) 
                    {
                        desc.cordSums[minSIndex * dim_ + d] += P_[n * dim_ + d];
                    }
                #endif
                    // assing to each particle index of closest chosen sample
                    // within given radius
                    desc.sphereCellIdx[n] = minSIndex;
                    }
                }

            #if defined(RD_USE_OPENMP)
                }   // end omp parallel
            #endif

            desc.contFlag = false;
            desc.part = EvolvePhaseDesc::MASS_CENTERS_SHIFT;
            break;
        case EvolvePhaseDesc::MASS_CENTERS_SHIFT:
            #if defined(RD_USE_OPENMP)
            #pragma omp parallel for num_threads(ompNumThreads_), schedule (static)
            #endif
            for (size_t k = 0; k < ns_; k++) 
            {
                for (size_t d = 0; d < dim_; d++) 
                {
                    if ((T)desc.spherePointCount[k]) 
                    {
                        T massCenter = desc.cordSums[k * dim_ + d] / (T)desc.spherePointCount[k];
                        // if distance from mass center is numerically distinguishable
                        if (std::fabs(massCenter - S_[k * dim_ + d])
                                > 2.f * std::fabs(massCenter) * desc.spherePointCount[k]
                                 * std::numeric_limits<T>::epsilon()) 
                        {
                        #if defined(RD_USE_OPENMP)
                        #pragma omp critical
                        {
                            contFlag = 1;
                        }
                        #else
                            contFlag = 1;
                        #endif
                        }
                        S_[k * dim_ + d] = massCenter;
                    }
                }
            }
            desc.part = EvolvePhaseDesc::SPHERE_CELLS_DIAGRAM;
            break;
        default:
            throw std::logic_error("Unsupported EvolvePhase part!");
    }

    return true;
}

template <typename T>
bool RDSimVis::doStep(DecimatePhaseDesc *desc)
{

    if (desc.left != ns_ && ns_ > 3)
    {
        desc.left = ns_;
        desc.deleteCount = 0;
        desc.currIdx = 0;
        for (typename std::list<T*>::iterator it = SList_.begin();
                     it != SList_.end(); desc.currIdx++) 
        {
            if (countNeighbouringPoints(SList_, *it, dim_, 4.f*r2, 4) ||
                    !countNeighbouringPoints(SList_, *it, dim_, 16.f*r2, 3)) 
            {
                ns_--;
                it = SList_.erase(it);
                desc.decimateIdx[deleteCount++] = desc.currIdx;
                if (ns_ < 3) break;
            } else it++;
        }
        /*
         * ADD OPENGL DRAWING CALLBACK HERE
         * draw points marked to delete
         */
    }
    else
    {
        T* dstAddr = S_;
        // copy real data
        for (typename std::list<T*>::iterator it = SList_.begin(); it != SList_.end();
                dstAddr += dim_, it++) 
        {
            if (*it != dstAddr) 
            {
                copyTable<T>(*it, dstAddr, dim_);
                *it = dstAddr;
            }
        }

        // check whether to prepare evolve descriptor
        if (desc.prevChosenCount != desc.chosenCount)
        {
            // finish chose phase and create new descriptor
            delete stateDesc_;
            stateDesc_ = new EvolvePhaseDesc<T>();
            EvolvePhaseDesc<T> *ed = dynamic_cast<EvolvePhaseDesc<T>*>(stateDesc_);
            ed.r2 = r1_ * r1_;
            ed.chosenCount = ns_;
            ed.cordSums = new T[ns_ * dim_];
            ed.spherePointCount = new int[ns_];
            ed.sphereCellIdx = new int[np_];
            phase = EVOLVE;
        } 
        else
        {
            delete stateDesc_;
            stateDesc_ = nullptr;
            phase = STOP;
        }
        return false;
    }
    return true;
}


/**
 * @brief      Runs the algorithm from current state till the end.
 */
template <typename T>
void RDSimVis::finish()
{
    do
    {
        step();
    } while (phase != STOP);
}

/**
 * @brief      Runs current phase till the end.
 */
template <typename T>
void RDSimVis::finishPhase()
{
    if (stateDesc_ != nullptr)
    {
        switch (phase)
        {
            case STOP:
                break;
            case CHOSE:
                chose();
                break;
            case EVOLVE:
                evolve();
                break;
            case DECIMATE:
                decimate();
                break;
            default:
                throw std::logic_error("Unsupported simulation phase!");
        }
    }
}



}   // end namespace vis

}   // end namespace rd
