"""glu[Un]Project[4] convenience wrappers"""
from OpenGL.raw import GLU as _simple
from OpenGL import GL
from OpenGL.lazywrapper import lazy as _lazy
import ctypes 
POINTER = ctypes.POINTER


@_lazy( _simple.gluUnProject )
def gluUnProject( baseFunction, winX, winY, winZ, model=None, proj=None, view=None ):
    """Convenience wrapper for gluUnProject
    
    Automatically fills in the model, projection and viewing matrices
    if not provided.
    
    returns (objX,objY,objZ) doubles
    """
    if model is None:
        model = GL.glGetDoublev( GL.GL_MODELVIEW_MATRIX )
    if proj is None:
        proj = GL.glGetDoublev( GL.GL_PROJECTION_MATRIX )
    if view is None:
        view = GL.glGetIntegerv( GL.GL_VIEWPORT )
    objX = _simple.GLdouble( 0.0 )
    objY = _simple.GLdouble( 0.0 )
    objZ = _simple.GLdouble( 0.0 )
    result = baseFunction( 
        winX,winY,winZ,
        model,proj,view,
        ctypes.POINTER(ctypes.c_double)(objX),
        ctypes.POINTER(ctypes.c_double)(objY),
        ctypes.POINTER(ctypes.c_double)(objZ)
    )
    if not result:
        raise ValueError( """Projection failed!""" )
    return objX.value, objY.value, objZ.value 
