/**
 *  \file camera.cpp
 *  \author Adam Rogowiec
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


#include "inc/camera.h"

#include "glm/gtx/transform.hpp"
#include "glm/geometric.hpp"

namespace rd
{

namespace vis
{

const glm::vec3 FreeCamera::X_AXIS = glm::vec3(1.f, 0.f, 0.f);
const glm::vec3 FreeCamera::Y_AXIS = glm::vec3(0.f, 1.f, 0.f);
const glm::vec3 FreeCamera::Z_AXIS = glm::vec3(0.f, 0.f, 1.f);

FreeCamera::FreeCamera() : fov(0.f), aspr(1.f), 
    up(glm::vec3(0.f, 1.f, 0.f))
{
    projMat = glm::mat4(1.f);
    mviewMat = glm::mat4(1.f);
    mvpMat = glm::mat4(1.f);
    poi = glm::vec3(0.f);
    position = glm::vec3(0.f);
}

FreeCamera::~FreeCamera()
{

}

/**
 * @brief Setup projection matrix
 * @param f[in] - field of view (in degrees)
 * @param w [in] - window width
 * @param h [in] - window height
 */
void FreeCamera::setupProjection(const float f,
                                 const float w,
                                 const float h)
{
    projMat = glm::perspective(glm::radians(f), w / h, 0.1f, 1000.0f); 
    fov = f;
    aspr = w / h;
    update();
}

void FreeCamera::update()
{
    mviewMat = glm::lookAt(position, poi, up);
    mvpMat = projMat * mviewMat;
}

void FreeCamera::moveToward(const float d)
{
    glm::vec3 dir = glm::normalize(poi - position);
    position += dir * d;
    poi += dir * d;
    update();
}

void FreeCamera::moveLeft(const float d)
{
    glm::vec3 forward = glm::normalize(poi - position);
    glm::vec3 dir = glm::cross(up, forward);
    position += dir * d;
    poi += dir * d;
    update();
}

void FreeCamera::moveRight(const float d)
{
    glm::vec3 forward = glm::normalize(poi - position);
    glm::vec3 dir = glm::cross(forward, up);
    position += dir * d;
    poi += dir * d;
    update();
}

void FreeCamera::moveVertically(const float d)
{
    moveY(d);
}

void FreeCamera::moveX(const float dx)
{
    position += glm::vec3(dx, 0.f, 0.f);
    poi += glm::vec3(dx, 0.f, 0.f);
    update();
}

void FreeCamera::moveY(const float dy)
{
    position += glm::vec3(0.f, dy, 0.f);
    poi += glm::vec3(0.f, dy, 0.f);
    update();
}

void FreeCamera::moveZ(const float dz)
{
    position += glm::vec3(0.f, 0.f, dz);
    poi += glm::vec3(0.f, 0.f, dz);
    update();
}

void FreeCamera::rotateVertically(const float d)
{
    glm::vec3 forward = glm::normalize(poi - position);
    glm::vec3 perpendicularV = glm::cross(up, forward);
    poi = glm::vec3(rotate(perpendicularV, d) * glm::vec4(poi, 1.f));
    update();
}

void FreeCamera::rotateHorizontally(const float d)
{
    rotateY(d);
}

void FreeCamera::rotateX(const float dx)
{
    poi = glm::vec3(rotate(X_AXIS, dx) * glm::vec4(poi, 1.f));
    update();
}

void FreeCamera::rotateY(const float dy)
{
    poi = glm::vec3(rotate(Y_AXIS, dy) * glm::vec4(poi, 1.f));
    update();
}

void FreeCamera::rotateZ(const float dz)
{
    poi = glm::vec3(rotate(Z_AXIS, dz) * glm::vec4(poi, 1.f));
    update();
}

glm::mat4 FreeCamera::rotate(const glm::vec3 &rotAxis, const float angle)
{
    return glm::translate(-position) *
                glm::rotate(glm::radians(angle), rotAxis) *
                glm::translate(position);
}

};  // end namespace rd::vis

};  // end namespace rd