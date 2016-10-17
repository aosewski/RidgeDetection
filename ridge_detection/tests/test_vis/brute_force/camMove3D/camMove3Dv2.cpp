/**
 *  \file camMove3Dv2.cpp
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

#include <inc/vis.h>
#include <inc/sb7ktx.h>
#include <inc/shader.h>
#include <inc/camera.h>

#define GLM_FORCE_RADIANS

#include "glm/gtx/transform.hpp"
#include "glm/gtx/string_cast.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "glm/detail/type_mat.hpp"
#include "glm/detail/func_trigonometric.hpp"

#include <cmath>

class dispmap_app : public rd::vis::VisApp
{

public:
    dispmap_app()
        : program(0),
          enable_displacement(true),
          wireframe(false),
          enable_fog(true),
          paused(false),
          translStep(1.f),
          rotStep(5.f)
    {

    }

    void load_shaders();

    void init()
    {
        static const char title[] = "OpenGL SuperBible - Displacement Mapping";

        rd::vis::VisApp::init();

        memcpy(info.title, title, sizeof(title));
    }

    virtual void startup()
    {
        load_shaders();

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glPatchParameteri(GL_PATCH_VERTICES, 4);

        glEnable(GL_CULL_FACE);

        tex_displacement = sb7::ktx::file::load("env/textures/terragen1.ktx");
        glActiveTexture(GL_TEXTURE1);
        tex_color = sb7::ktx::file::load("env/textures/terragen_color.ktx");

        cam.position = glm::vec3(0.f, 5.f, 0.f);
        cam.poi = glm::vec3(0.f, 5.f, -10.f);
        cam.setupProjection(60.f, (float)info.windowWidth,
                                     (float)info.windowHeight);
    }

    virtual void render(double currentTime)
    {
        static const GLfloat black[] = { 0.85f, 0.95f, 1.0f, 1.0f };
        static const GLfloat one = 1.0f;

        glViewport(0, 0, info.windowWidth, info.windowHeight);
        glClearBufferfv(GL_COLOR, 0, black);
        glClearBufferfv(GL_DEPTH, 0, &one);

        glUseProgram(program);

        glUniformMatrix4fv(uniforms.mv_matrix, 1, GL_FALSE, glm::value_ptr(cam.mviewMat));
        glUniformMatrix4fv(uniforms.proj_matrix, 1, GL_FALSE, glm::value_ptr(cam.projMat));
        glUniformMatrix4fv(uniforms.mvp_matrix, 1, GL_FALSE, glm::value_ptr(cam.mvpMat));
        glUniform1f(uniforms.dmap_depth, enable_displacement ? dmap_depth : 0.0f);
        glUniform1i(uniforms.enable_fog, enable_fog ? 1 : 0);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        glDrawArraysInstanced(GL_PATCHES, 0, 4, 64 * 64);
    }

    virtual void onResize(int w, int h)
    {
        BaseT::onResize(w, h);
        cam.setupProjection(60.f, (float)info.windowWidth,
                                 (float)info.windowHeight);
    }

    virtual void shutdown()
    {
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(program);
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

    void onKey(int key, int action)
    {
        if (action)
        {
            switch (key)
            {
                case GLFW_KEY_KP_ADD: dmap_depth += 0.1f;
                    break;
                case GLFW_KEY_KP_SUBTRACT: dmap_depth -= 0.1f;
                    break;
                case 'F': enable_fog = !enable_fog;
                    break;
                // case 'D': enable_displacement = !enable_displacement;
                //     break;
                // case 'W': wireframe = !wireframe;
                //     break;
                case 'P': paused = !paused;
                    break;
                case 'R':
                        load_shaders();
                    break;
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
            };
        }
    }

private:

    typedef rd::vis::VisApp BaseT;

    GLuint          program;
    GLuint          vao;
    GLuint          tex_displacement;
    GLuint          tex_color;
    float           dmap_depth;
    bool            enable_displacement;
    bool            wireframe;
    bool            enable_fog;
    bool            paused;

    rd::vis::FreeCamera cam;

    float translStep, rotStep;

    struct
    {
        GLint       mvp_matrix;
        GLint       mv_matrix;
        GLint       proj_matrix;
        GLint       dmap_depth;
        GLint       enable_fog;
    } uniforms;
};

void dispmap_app::load_shaders()
{
    if (program)
        glDeleteProgram(program);

    GLuint vs = rd::vis::shader::load("env/shaders/dispmap/dispmap.vs.glsl", GL_VERTEX_SHADER);
    GLuint tcs = rd::vis::shader::load("env/shaders/dispmap/dispmap.tcs.glsl", GL_TESS_CONTROL_SHADER);
    GLuint tes = rd::vis::shader::load("env/shaders/dispmap/dispmap.tes.glsl", GL_TESS_EVALUATION_SHADER);
    GLuint fs = rd::vis::shader::load("env/shaders/dispmap/dispmap.fs.glsl", GL_FRAGMENT_SHADER);

    program = glCreateProgram();

    glAttachShader(program, vs);
    glAttachShader(program, tcs);
    glAttachShader(program, tes);
    glAttachShader(program, fs);

    glLinkProgram(program);

    uniforms.mv_matrix = glGetUniformLocation(program, "mv_matrix");
    uniforms.mvp_matrix = glGetUniformLocation(program, "mvp_matrix");
    uniforms.proj_matrix = glGetUniformLocation(program, "proj_matrix");
    uniforms.dmap_depth = glGetUniformLocation(program, "dmap_depth");
    uniforms.enable_fog = glGetUniformLocation(program, "enable_fog");
    dmap_depth = 6.0f;
}


//////////////////////////////////////////////////////////////////////
//
//      MAIN FUNCTION
//
//////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]) {

    dispmap_app *vis = new dispmap_app();
    vis->run(vis);

    delete vis;
    return 0;
}

