/**
 *	\file simpleVis3D.cpp
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

#include "vis.h"

#define GLM_FORCE_RADIANS

#include "glm/gtx/transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/detail/type_mat.hpp"
#include "glm/detail/func_trigonometric.hpp"

#include <cmath>

class SimpleVis3D : public rd::vis::VisApp
{

public:

    // samples count
    unsigned int np;
    // ptr to data to visualize
    float *dataPtr;

    SimpleVis3D() 
        : np(0), dataPtr(nullptr)
    {

    }

    void init();
    virtual void startup();
    virtual void render(double currentTime);
    virtual void shutdown();

private:

    static const int SDIM = 3;

    GLuint          program;
    GLuint          samplesVAO;
    GLuint          samplesBuffer;
    GLint           pMatLoc;
    GLint           mvMatLoc;

    BoundingBox<float> bbox;
};


void SimpleVis3D::init()
{
    static const char title[] = "Ridge detection - simple vis 3D";

    rd::vis::VisApp::init();

    memcpy(info.title, title, sizeof(title));
}

void SimpleVis3D::startup()
{
    static const char * vsSource[] =
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

    static const char * fsSource[] =
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
    glBindVertexArray(samplesVAO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);


    program = glCreateProgram();
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, fsSource, NULL);
    glCompileShader(fs);

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, vsSource, NULL);
    glCompileShader(vs);

    glAttachShader(program, vs);
    glAttachShader(program, fs);

    glLinkProgram(program);
    glDeleteShader(vs);
    glDeleteShader(fs);

    pMatLoc = glGetUniformLocation(program, "pMat");
    mvMatLoc = glGetUniformLocation(program, "mvMat");

    bbox.findBounds(dataPtr, np, SDIM);
}

void SimpleVis3D::render(double currentTime)
{
    static const GLfloat black[] = { 0.f, 0.f, 0.f, 1.0f };

    glm::mat4 projMat = glm::perspective(glm::radians(50.0f),
                    (float)info.windowWidth / (float)info.windowHeight,
                     0.1f, 
                     1000.0f);

    glm::mat4 mvMat = 
                glm::rotate(glm::radians(30.0f), glm::vec3(0.f, 1.f, 0.f)) * 
                glm::rotate(glm::radians(90.0f), glm::vec3(1.f, 0.f, 0.f)) * 
                glm::translate(glm::vec3(0.f, -40.f, 0.f));


    glViewport(0, 0, info.windowWidth, info.windowHeight);
    glClearBufferfv(GL_COLOR, 0, black);

    glPointSize(2.0f);
    glUseProgram(program);

    glUniformMatrix4fv(pMatLoc, 1, GL_FALSE, glm::value_ptr(projMat));
    glUniformMatrix4fv(mvMatLoc, 1, GL_FALSE, glm::value_ptr(mvMat));
    glDrawArrays(GL_POINTS, 0, np);
}

void SimpleVis3D::shutdown()
{
    glDeleteProgram(program);
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

    SimpleVis3D *vis = new SimpleVis3D();
    vis->dataPtr = samples;
    vis->np = NSAMPLES;
    vis->run(vis);

    delete vis;
    return 0;
}