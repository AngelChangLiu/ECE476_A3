#ifndef __PLATFORM_GL_H__
#define __PLATFORM_GL_H__

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#if defined(__has_include)
#if __has_include(<GL/glut.h>)
#include <GL/glut.h>
#elif __has_include(<GL/freeglut.h>)
#include <GL/freeglut.h>
#else
#error "OpenGL/GLUT headers not found. Install GLUT/freeglut or build with USE_OPENGL=0."
#endif
#else
#include <GL/glut.h>
#endif
#endif

#endif
