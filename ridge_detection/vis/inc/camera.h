/**
 * @file camera.h
 * @author Adam Rogowiec
 *
 * This file is an integral part of the master thesis entitled:
 * "Elaboration and implementation in CUDA technology parallel version of 
 *  estimation of multidimensional random variable density function ridge
 *  detection algorithm.",
 * which is conducted under the supervision of prof. dr hab. inż. Marka Nałęcza.
 * 
 * Institute of Control and Computation Engineering
 * Faculty of Electronics and Information Technology
 * Warsaw University of Technology 2016
 */

#ifndef VIS_CAMERA_H
#define VIS_CAMERA_H

#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace rd
{

namespace vis
{

/**
 * @class FreeCamera
 * @brief Represents camera object in OpenGL and enables easy
 *      transformations of its view.
 *
 */
class FreeCamera
{

public:

    float fov, aspr;
    
    /// camera position
    glm::vec3 position;
    /// camera point of interest (direction we look toward)
    glm::vec3 poi;
    // up direction
    glm::vec3 up;
    /// model-view matrix
    glm::mat4 mviewMat; 
    /// projection matrix
    glm::mat4 projMat; 
    /// model-view-projection
    glm::mat4 mvpMat;

    FreeCamera();
    ~FreeCamera();
    
    
    //////////////////////////////////////////
    //
    //  Camera transformation
    //
    //////////////////////////////////////////

    /**
     * @brief Setup projection matrix
     * @param f [in] - field of view
     * @param w [in] - window width
     * @param h [in] - window height
     */
    void setupProjection(const float f,
                         const float w,
                         const float h);
    
    /**
     * @brief 
     *
     */
    void update();

    // Translations

    /**
     * @brief      Moves the camera forward or backward
     *
     * @param[in]  d     Distance to move.
     */ 
    void moveToward(const float d);
    /**
     * @brief      Moves the camera left sideways
     *
     * @param[in]  d     Distance to move.
     */
    void moveLeft(const float d);
    /**
     * @brief      Moves the camera right sideways
     *
     * @param[in]  d     Distance to move.
     */
    void moveRight(const float d);
    /**
     * @brief      Rise or lower camera position.
     *
     * @param[in]  d     Distance to move.
     */
    void moveVertically(const float d);

    void moveX(const float dx);
    void moveY(const float dy);
    void moveZ(const float dz);

    // Rotations
    // Argument is in degrees!

    /**
     * @brief      Rotates camera up or down
     *
     * @param[in]  d     Rotation anlge.
     */
    void rotateVertically(const float d);
    /**
     * @brief      Rotates camera left or right
     *
     * @param[in]  d     Rotation angle.
     */
    void rotateHorizontally(const float d);

    void rotateX(const float dx);
    void rotateY(const float dy);
    void rotateZ(const float dz);


protected:

    static const glm::vec3 X_AXIS;
    static const glm::vec3 Y_AXIS;
    static const glm::vec3 Z_AXIS;

    /**
     * @brief      Rotates camera around @p rotAxis.
     *             
     * @param[in]  rotAxis  Axis to rotate around.
     * @param[in]  angle    Angle to rotate.
     *
     * @return     Model view transformation matrix.
     */
    glm::mat4 rotate(const glm::vec3 &rotAxis, const float angle);


};

};  // end namespace rd::vis

};  // end namespace rd

#endif  // VIS_CAMERA_H

