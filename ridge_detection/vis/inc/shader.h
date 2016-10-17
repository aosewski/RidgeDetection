/**
 *  \file shader.h
 *  \author Adam Rogowiec
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

#ifndef __SHADER_H__
#define __SHADER_H__

namespace rd
{

namespace vis
{

namespace shader
{

GLuint load(const char * filename,
            GLenum shader_type = GL_FRAGMENT_SHADER,
#ifdef _DEBUG
            bool check_errors = true);
#else
            bool check_errors = false);
#endif

GLuint from_string(const char * source,
                   GLenum shader_type,
#ifdef _DEBUG
                   bool check_errors = true);
#else
                   bool check_errors = false);
#endif

}     // end namespace shader

namespace program
{

GLuint link_from_shaders(const GLuint * shaders,
                         int shader_count,
                         bool delete_shaders,
#ifdef _DEBUG
                         bool check_errors = true);
#else
                         bool check_errors = false);
#endif

}     // end namespace program 

}     // end namespace vis

}     // end namespace rd

#endif /* __SHADER_H__ */
