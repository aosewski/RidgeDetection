/**
 *	\file camMove3D.cpp
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

#include "rd/utils/samples_set.hpp"
#include "rd/utils/bounding_box.hpp"

#include "inc/vis.h"
#include "inc/shader.h"
#include "inc/sb7ktx.h"
#include "inc/camera.h"

#define GLM_FORCE_RADIANS

#include "glm/gtx/transform.hpp"
#include "glm/gtx/string_cast.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/detail/type_mat.hpp"
#include "glm/detail/func_trigonometric.hpp"

#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>

class camMove3D : public rd::vis::VisApp
{

public:

    // samples count
    unsigned int np;
    // ptr to data to visualize
    float *dataPtr;

    camMove3D() 
        : np(0), dataPtr(nullptr), 
            samplesProg(0),
            rotStep(glm::radians(5.f))
    {

    }

    void init();
    virtual void startup();
    virtual void render(double currentTime);
    virtual void onKey(int key, int action);
    virtual void shutdown();

private:

    static constexpr int SDIM = 3;

    GLuint          samplesProg;
    GLuint          samplesVAO;
    GLuint          samplesBuffer;
    GLuint          pMatLoc;
    GLuint          mvMatLoc;

    float rotStep, translStep;
    rd::vis::FreeCamera cam;

    void initSamples();
    void renderSamples(glm::mat4 &mvMatrix, glm::mat4 &projMatrix);
};


void camMove3D::init()
{
    static const char title[] = "Ridge detection - moving camera vis 3D";

    rd::vis::VisApp::init();

    memcpy(info.title, title, sizeof(title));
}

void camMove3D::startup()
{

    cam.position = glm::vec3(0.f, 5.f, 0.f);
    cam.poi = glm::vec3(0.f, 5.f, -10.f);
    cam.setupProjection(60.f, (float)info.windowWidth,
                                 (float)info.windowHeight);
    initSamples();
}


void camMove3D::initSamples()
{
    static const char * vsSamples[] =
    {
        "#version 420 core                             \n"
        "                                              \n"
        "layout (location = 0) in vec3 pos;            \n"
        "                                              \n"
        "uniform mat4 pMat;                            \n"
        "uniform mat4 mvMat;                           \n"
        "                                              \n"
        "void main(void)                               \n"
        "{                                             \n"
        "    gl_Position = pMat * mvMat * vec4(pos.x, pos.y, pos.z, 1.0);   \n"
        "}                                             \n"
    };

    static const char * fsSamples[] =
    {
        "#version 420 core                             \n"
        "                                              \n"
        "out vec4 color;                               \n"
        "                                              \n"
        "void main(void)                               \n"
        "{                                             \n"
        "    color = vec4(1.0, 0.0, 0.0, 1.0);         \n"
        "}                                             \n"
    };

    glGenBuffers(1, &samplesBuffer);

    glBindBuffer(GL_ARRAY_BUFFER, samplesBuffer);
    glBufferData(GL_ARRAY_BUFFER, np * SDIM * sizeof(GLfloat),
                    NULL, GL_STATIC_DRAW);
    // copy data to buffer
    glBufferSubData(GL_ARRAY_BUFFER, 0, np * SDIM * sizeof(GLfloat), dataPtr);

    glGenVertexArrays(1, &samplesVAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    samplesProg = glCreateProgram();
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, fsSamples, NULL);
    glCompileShader(fs);

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, vsSamples, NULL);
    glCompileShader(vs);

    glAttachShader(samplesProg, vs);
    glAttachShader(samplesProg, fs);

    glLinkProgram(samplesProg);
    glDeleteShader(vs);
    glDeleteShader(fs);

    pMatLoc = glGetUniformLocation(samplesProg, "pMat");
    mvMatLoc = glGetUniformLocation(samplesProg, "mvMat");
}

void camMove3D::render(double currentTime)
{
    static const GLfloat black[] = { 0.f, 0.f, 0.f, 1.0f };
    static const GLfloat one = 1.0f;

    glViewport(0, 0, info.windowWidth, info.windowHeight);
    glClearBufferfv(GL_COLOR, 0, black);
    glClearBufferfv(GL_DEPTH, 0, &one);

    renderSamples(cam.mviewMat, cam.projMat);
    
}

void camMove3D::renderSamples(glm::mat4 &mvMatrix, glm::mat4 &projMatrix)
{
    glPointSize(2.0f);
    glUseProgram(samplesProg);
    glBindVertexArray(samplesVAO);

    glUniformMatrix4fv(pMatLoc, 1, GL_FALSE, glm::value_ptr(projMatrix));
    glUniformMatrix4fv(mvMatLoc, 1, GL_FALSE, glm::value_ptr(mvMatrix));

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    glDrawArrays(GL_POINTS, 0, np);
    // glBindVertexArray(0);
}

virtual void onResize(int w, int h)
{
    BaseT::onResize(w, h);
    cam.setupProjection(60.f, (float)info.windowWidth,
                             (float)info.windowHeight);
}

void camMove3D::onKey(int key, int action)
{
    if (action)
    {
        switch (key)
        {
            case GLFW_KEY_D: 
                cam.moveRight(translStep);
                break;
            case GLFW_KEY_A:
                    cam.moveLeft(translStep);
                break;
            case GLFW_KEY_W:
                    cam.moveToward(translStep);
                break;
            case GLFW_KEY_S:
                    cam.moveToward(-translStep);
                break;
            case GLFW_KEY_Q:
                    cam.moveVertically(translStep);
                break;
            case GLFW_KEY_Z:
                    cam.moveVertically(-translStep);
                break;
            default:
                break;
        }
    }
}

virtual void onMouseMove(int x, int y)
{
    glm::vec2 shift = glm::vec2(x, y) - glm::vec2(info.winCenterX,
                                                  info.winCenterY);
    // calculate rotation angles;
    float rotVertAngle = shift.y / info.winCenterY * 90.f * 0.01f;
    float rotHorAngle = shift.x / info.winCenterX * 180.f * 0.01f;

    // rotate camera
    cam.rotateHorizontally(-rotHorAngle);
    cam.rotateVertically(rotVertAngle);
}

void camMove3D::shutdown()
{
    glDeleteProgram(samplesProg);
    glDeleteBuffers(1, &samplesBuffer);
    glDeleteVertexArrays(1, &samplesVAO);
}


//////////////////////////////////////////////////////////////////////
//
//      MAIN FUNCTION
//
//////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {

    const int NSAMPLES = 10000;
    rd::Samples<float> sGen;
    float *samples = sGen.genSpiral3D(NSAMPLES, 3.f, 0.5f, 3.f);

    camMove3D *vis = new camMove3D();
    vis->dataPtr = samples;
    vis->np = NSAMPLES;
    vis->run(vis);

    delete vis;
    return 0;
}
