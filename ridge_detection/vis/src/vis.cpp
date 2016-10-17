/**
 *  \file vis.cpp
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

 /*
 * Copyright © 2012-2015 Graham Sellers
 *
 * This code is part of the OpenGL SuperBible, 6th Edition.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */


#include "inc/vis.h"

#include <string.h>

rd::vis::VisApp *rd::vis::VisApp::app = 0;

void APIENTRY rd::vis::VisApp::debug_callback(GLenum source,
                                               GLenum type,
                                               GLuint id,
                                               GLenum severity,
                                               GLsizei length,
                                               const GLchar* message,
                                               GLvoid* userParam)
{
    reinterpret_cast<VisApp *>(userParam)->
    	onDebugMessage(source, type, id, severity, length, message);
}

GL3WglProc sb6GetProcAddress(const char * funcname)
{
    return gl3wGetProcAddress(funcname);
}

int sb6IsExtensionSupported(const char * extname)
{
    GLint numExtensions;
    GLint i;

    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);

    for (i = 0; i < numExtensions; i++)
    {
        const GLubyte * e = glGetStringi(GL_EXTENSIONS, i);
        if (!strcmp((const char *)e, extname))
        {
            return 1;
        }
    }

    return 0;
}