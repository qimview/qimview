#include "opengl_functions.hpp"

//void *GetAnyGLFuncAddress(const char *name)
//{
//  void *p = (void *)wglGetProcAddress(name);
//  if (p == 0 ||
//    (p == (void*)0x1) || (p == (void*)0x2) || (p == (void*)0x3) ||
//    (p == (void*)-1))
//  {
//    HMODULE module = LoadLibraryA("opengl32.dll");
//    p = (void *)GetProcAddress(module, name);
//  }
//
//  return p;
//}

bool GLcpp::InitGlew()
{
  GLenum err = glewInit();
  if (GLEW_OK != err)
  {
    /* Problem: glewInit failed, something is seriously wrong. */
    fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    return false;
  }
  return true;
}

