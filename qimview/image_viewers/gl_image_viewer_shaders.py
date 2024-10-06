#
#
# started from https://cyrille.rossant.net/2d-graphics-rendering-tutorial-with-pyopengl/
#
#

import OpenGL.GL as gl
from OpenGL.GL import shaders
import argparse
import sys
import numpy as np

from PySide6.QtOpenGL import (QOpenGLBuffer, QOpenGLShader,
                              QOpenGLShaderProgram, QOpenGLTexture)

from qimview.utils.qt_imports  import QtWidgets, QtGui
from .image_viewer             import trace_method, get_time, OverlapMode
from .gl_image_viewer_base     import GLImageViewerBase
from qimview.utils.viewer_image import ImageFormat

# Deal with compatibility with GLSL 1.2 
GLSL_VERSION = '120' if sys.platform=='darwin' else '330 core'
OUT = 'varying' if GLSL_VERSION=='120' else 'out'
IN  = 'varying' if GLSL_VERSION=='120' else 'in'

DECLARE_GLOBAL_COLOUR = ''                                 if GLSL_VERSION=='120' else 'out vec3 colour;'
DECLARE_LOCAL_COLOUR  = 'vec3 colour;'                     if GLSL_VERSION=='120' else ''
RETURN_COLOUR         = 'gl_FragColor=vec4(colour.xyz,1);' if GLSL_VERSION=='120' else ''
TEXTURE               = 'texture2D'                        if GLSL_VERSION=='120' else 'texture'

class GLImageViewerShaders(GLImageViewerBase):

    # glsl version
    glslVersion = f"""
        #version {GLSL_VERSION}
    """

    # vertex shader program
    vertexShader = f"""
        {glslVersion}

        attribute vec3 vert;
        attribute vec2 uV;
        uniform mat4 mvMatrix;
        uniform mat4 pMatrix;
        {OUT} vec2 UV;

        void main() {{
          gl_Position = pMatrix * mvMatrix * vec4(vert, 1.0);
          UV = uV;
        }}
        """

    fragmentShader_apply_filters = """
          vec3 apply_filters(const vec3 input_rgb,
              float max_value, float texture_scale, float max_type, float black_level,
              float g_r_coeff, float g_b_coeff,
              float white_level, float gamma)
          {
            vec3 res = input_rgb;

            /// Black level
            res = res/(max_value*texture_scale)*max_type;
            res = max((res-vec3(black_level).rgb),0);

            /// White balance
            res.r = res.r*g_r_coeff;
            res.b = res.b*g_b_coeff;

            /// Rescale to white level as saturation level
            res = res/(white_level-black_level);
            
            /// Apply gamma
            if (gamma!=1.f)
               res = pow(res, vec3(1.0/gamma).rgb);
            
            return res;
          }
        """

    fragmentShader_declare_filter_params = """
        uniform float white_level;
        uniform float black_level;
        uniform float g_r_coeff;
        uniform float g_b_coeff;
        uniform float max_value; // maximal value based on image precision
        uniform float max_type;  // maximal value based on image type (uint8, etc...)
        uniform float gamma;
        """

    fragmentShader_yuv2rgb = """
          vec3 yuv2rgb(const vec3 yuv)
          {
            float y = 1.1643*(yuv[0]-0.0625);
            float u = yuv[1]-0.5;
            float v = yuv[2]-0.5;

            float r = y+1.5958*v;
            float g = y-0.39173*u-0.81290*v;
            float b = y+2.017*u;
            return vec3(r,g,b);
          }
        """

        #   Another version
        #   r = y+1.13983*v;
        #   g = y-0.39465*u-0.5806*v;
        #   b = y+2.03211*u;

        #   r = y+1.28033*v;
        #   g = y-0.21482*u-0.38059*v;
        #   b = y+2.12798*u;


    # fragment shader program
    fragmentShader_RGB = f"""
        {glslVersion}
    
        {IN} vec2 UV;
        uniform sampler2D backgroundTexture;
        uniform int channels; // channel representation
        {fragmentShader_declare_filter_params}
        {DECLARE_GLOBAL_COLOUR}
    
        {fragmentShader_apply_filters}
        {fragmentShader_yuv2rgb}

        void main() {{
          {DECLARE_LOCAL_COLOUR}
          colour = {TEXTURE}(backgroundTexture, UV).rgb;
          colour = apply_filters(colour,max_value
              , 1 // texture_scale
              , max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
          {RETURN_COLOUR}
        }}
    """


    # fragment shader program
    fragmentShader_YUV420 = f"""
        {glslVersion}
    
        {IN} vec2 UV;
        uniform sampler2D YTex;
        uniform sampler2D UTex;
        uniform sampler2D VTex;
        uniform float texture_scale;
        uniform int channels; // channel representation
        {fragmentShader_declare_filter_params}
        float y,u,v, r, g, b;
        {DECLARE_GLOBAL_COLOUR}
        
        {fragmentShader_apply_filters}
        {fragmentShader_yuv2rgb}

        void main() {{
          {DECLARE_LOCAL_COLOUR}
          y = {TEXTURE}(YTex, UV).r*texture_scale;
          u = {TEXTURE}(UTex, UV).r*texture_scale;
          v = {TEXTURE}(VTex, UV).r*texture_scale;

          vec3 rgb = yuv2rgb(vec3(y,u,v));
          colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
          {RETURN_COLOUR}
        }}
    """

    fragmentShader_YUV420_twotex = f"""
        {glslVersion}
    
        {IN} vec2 UV;
        uniform sampler2D YTex;
        uniform sampler2D UTex;
        uniform sampler2D VTex;
        uniform sampler2D YTex2;
        uniform sampler2D UTex2;
        uniform sampler2D VTex2;
        uniform float difference_scaling;
        uniform int   overlap_mode;
        uniform float cursor_x;
        uniform float cursor_y;
        uniform float texture_scale;
        uniform int channels; // channel representation
        {fragmentShader_declare_filter_params}
        float y,u,v, r, g, b;
        {DECLARE_GLOBAL_COLOUR}
    
        {fragmentShader_apply_filters}
        {fragmentShader_yuv2rgb}

        void main() {{
          {DECLARE_LOCAL_COLOUR}
          if (overlap_mode==2) {{
            // duplicate rectangle
            float w = 0.05;
            float h = 0.2;
            float px = cursor_x;
            float py = cursor_y;
            float dx = w*(1.01);
            float dy = 0.0;
            if ((UV.x>px+dx)&&(UV.x<px+dx+w)&&(UV.y>py+dy)&&(UV.y<py+dy+h)) {{
                vec2 UV2 = UV-vec2(dx,dy);
                y = {TEXTURE}(YTex2, UV2).r*texture_scale;
                u = {TEXTURE}(UTex2, UV2).r*texture_scale;
                v = {TEXTURE}(VTex2, UV2).r*texture_scale;
            }} else {{
                y = {TEXTURE}(YTex, UV).r*texture_scale;
                u = {TEXTURE}(UTex, UV).r*texture_scale;
                v = {TEXTURE}(VTex, UV).r*texture_scale;
            }}
            vec3 rgb = yuv2rgb(vec3(y,u,v));
            colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
          }} else {{
            float overlap_pos   = (overlap_mode==0)?UV.x:UV.y;
            float overlap_ratio = (overlap_mode==0)?cursor_x:cursor_y;
            if (difference_scaling>0) {{
                float y2 = {TEXTURE}(YTex2, UV).r*texture_scale;
                float u2 = {TEXTURE}(UTex2, UV).r*texture_scale;
                float v2 = {TEXTURE}(VTex2, UV).r*texture_scale;
                if (overlap_pos<=overlap_ratio) {{
                    y = {TEXTURE}(YTex, UV).r*texture_scale;
                    u = {TEXTURE}(UTex, UV).r*texture_scale;
                    v = {TEXTURE}(VTex, UV).r*texture_scale;
                    vec3 rgb1 = yuv2rgb(vec3(y,u,v));
                    vec3 rgb2 = yuv2rgb(vec3(y2,u2,v2));
                    vec3 rgb_diff = min(max((rgb2-rgb1)*difference_scaling+0.5,0),1);
                    colour = apply_filters(rgb_diff,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
                }} else {{
                    vec3 rgb = yuv2rgb(vec3(y2,u2,v2));
                    colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
                }}
            }} else {{
                if (overlap_pos<=overlap_ratio) {{
                    y = {TEXTURE}(YTex, UV).r*texture_scale;
                    u = {TEXTURE}(UTex, UV).r*texture_scale;
                    v = {TEXTURE}(VTex, UV).r*texture_scale;
                }} else {{
                    y = {TEXTURE}(YTex2, UV).r*texture_scale;
                    u = {TEXTURE}(UTex2, UV).r*texture_scale;
                    v = {TEXTURE}(VTex2, UV).r*texture_scale;
                }}
                vec3 rgb = yuv2rgb(vec3(y,u,v));
                colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
            }}
          }}
          {RETURN_COLOUR}
        }}
    """

    fragmentShader_YUV420_interlaced = f"""
        {glslVersion}
    
        {IN} vec2 UV;
        uniform sampler2D YTex;
        uniform sampler2D UVTex;
        uniform float texture_scale;
        uniform int channels; // channel representation
        {fragmentShader_declare_filter_params}
        float y,u,v, r, g, b;
        {DECLARE_GLOBAL_COLOUR}
    
        {fragmentShader_apply_filters}
        {fragmentShader_yuv2rgb}

        void main() {{
          {DECLARE_LOCAL_COLOUR}
          y  = {TEXTURE}(YTex,  UV).r*texture_scale;
          u  = {TEXTURE}(UVTex, UV).r*texture_scale;
          v  = {TEXTURE}(UVTex, UV).g*texture_scale;

          vec3 rgb = yuv2rgb(vec3(y,u,v));
          colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
          {RETURN_COLOUR}
        }}
    """

    # Version with 2 textures
    fragmentShader_YUV420_interlaced_twotex = f"""
        {glslVersion}
    
        {IN} vec2 UV;
        uniform sampler2D YTex;
        uniform sampler2D UVTex;
        uniform sampler2D YTex2;
        uniform sampler2D UVTex2;
        uniform float difference_scaling;
        uniform int   overlap_mode;
        uniform float cursor_x;
        uniform float cursor_y;
        uniform float texture_scale;
        uniform int channels; // channel representation
        {fragmentShader_declare_filter_params}
        float y,u,v, r, g, b;
        {DECLARE_GLOBAL_COLOUR}
    
        {fragmentShader_apply_filters}
        {fragmentShader_yuv2rgb}

        void main() {{
          {DECLARE_LOCAL_COLOUR}
          vec3 rgb = yuv2rgb(vec3(y,u,v));
          colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
 
          float overlap_pos = (overlap_mode==0)?UV.x:UV.y;
          float overlap_ratio = (overlap_mode==0)?cursor_x:cursor_y;

          if (difference_scaling>0) {{
            float y2 = {TEXTURE}(YTex2,  UV).r*texture_scale;
            float u2 = {TEXTURE}(UVTex2, UV).r*texture_scale;
            float v2 = {TEXTURE}(UVTex2, UV).g*texture_scale;
            if (overlap_pos<=overlap_ratio) {{
                y = {TEXTURE}(YTex,  UV).r*texture_scale;
                u = {TEXTURE}(UVTex, UV).r*texture_scale;
                v = {TEXTURE}(UVTex, UV).g*texture_scale;
                vec3 rgb1 = yuv2rgb(vec3(y,u,v));
                vec3 rgb2 = yuv2rgb(vec3(y2,u2,v2));
                vec3 rgb_diff = min(max((rgb2-rgb1)*difference_scaling+0.5,0),1);
             colour = apply_filters(rgb_diff,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
            }} else {{
                vec3 rgb = yuv2rgb(vec3(y2,u2,v2));
                colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
            }}
          }} else {{
            if (overlap_pos<=overlap_ratio) {{
                y = {TEXTURE}(YTex,  UV).r*texture_scale;
                u = {TEXTURE}(UVTex, UV).r*texture_scale;
                v = {TEXTURE}(UVTex, UV).g*texture_scale;
            }} else {{
                y = {TEXTURE}(YTex2,  UV).r*texture_scale;
                u = {TEXTURE}(UVTex2, UV).r*texture_scale;
                v = {TEXTURE}(UVTex2, UV).g*texture_scale;
            }}
            vec3 rgb = yuv2rgb(vec3(y,u,v));
            colour = apply_filters(rgb,max_value, texture_scale, max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
          }}
          {RETURN_COLOUR}
        }}
    """

    fragmentShader_RAW = f"""
        {glslVersion}
        
        {IN} vec2 UV;
        uniform sampler2D backgroundTexture;
        uniform int channels; // channel representation
        {fragmentShader_declare_filter_params}
        {DECLARE_GLOBAL_COLOUR}

        {fragmentShader_apply_filters}

        void main() {{
          {DECLARE_LOCAL_COLOUR}
          vec4 bayer = {TEXTURE}(backgroundTexture, UV);
          // transform bayer data to RGB
          int r,gr,gb,b;
          r  = channels%4;
          gr = (8-channels+1)%4;
          gb = (channels+2)%4;
          b  = (8-channels+3)%4;
          // if (channels==4)      {{ r = 0; gr = 1; gb = 2; b = 3; }} // CH_RGGB = 4 phase 0, bayer 2
          // else if (channels==5) {{ r = 1; gr = 0; gb = 3; b = 2; }} // CH_GRBG = 5 phase 1, bayer 3 
          // else if (channels==6) {{ r = 2; gr = 3; gb = 0; b = 1; }} // CH_GBRG = 6 phase 2, bayer 0
          // else if (channels==7) {{ r = 3; gr = 2; gb = 1; b = 0; }} // CH_BGGR = 7 phase 3, bayer 1 
          // else                  {{r = 0;  gr = 1; gb = 2; b = 3;  }} // this should not happen

          // first retreive black point to get the coefficients right ...
          // 5% of dynamics?
          
          // bayer 2 rgb
          colour.r   = bayer[r];
          colour.g = (bayer[gr]+bayer[gb])/2.0;
          colour.b = bayer[b];

          colour = apply_filters(colour,max_value, 
              1, // texture_scale set to 1 
              max_type, black_level, g_r_coeff, g_b_coeff, white_level, gamma);
          {RETURN_COLOUR}
        }}
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAutoFillBackground(False)
        self.opengl_debug                                = False
        self.pMatrix                                     = np.identity(4, dtype=np.float32)
        self.mvMatrix                                    = np.identity(4, dtype=np.float32)
        self.program_RGB                                 = None
        self.program_YUV420                              = None
        self.program_YUV420_twotex                       = None
        self.program_YUV420_interlaced                   = None
        self.program_YUV420_interlaced_twotex            = None
        self.program_RAW                                 = None
        self.program                                     = None
        self._vertex_buffer       : QOpenGLBuffer | None = None
        self._vertex_buffer_param                        = None
        self._transform_param                            = None
        # output crop [ width min, height min, width max, height max]
        self._output_crop                                = np.array([0., 0., 1., 1.], dtype=np.float32)

    def set_shaders(self):
        if self.program_RGB is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_RGB, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_RGB = shaders.compileProgram(vs, fs, validate=False)
                self.print_log(f"\n***** self.program_RGB = {self.program_RGB} *****\n")
            except Exception as e:
                print(f'failed RGB shaders.compileProgram() {e}')
            try:
                shaders.glDeleteShader(vs)
                shaders.glDeleteShader(fs)
            except:
                pass

        if self.program_YUV420 is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_YUV420, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_YUV420 = shaders.compileProgram(vs, fs, validate=False)
                self.print_log(f"\n***** self.program_YUV420 = {self.program_YUV420} *****\n")
            except Exception as e:
                print(f'failed RGB shaders.compileProgram() {e}')
            try:
                shaders.glDeleteShader(vs)
                shaders.glDeleteShader(fs)
            except:
                pass

        if self.program_YUV420_twotex is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_YUV420_twotex, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_YUV420_twotex = shaders.compileProgram(vs, fs, validate=False)
                self.print_log(f"\n***** self.program_YUV420_twotex = {self.program_YUV420_twotex} *****\n")
            except Exception as e:
                print(f'failed RGB shaders.compileProgram() {e}')
            try:
                shaders.glDeleteShader(vs)
                shaders.glDeleteShader(fs)
            except:
                pass


        if self.program_YUV420_interlaced is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_YUV420_interlaced, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_YUV420_interlaced = shaders.compileProgram(vs, fs, validate=False)
                self.print_log(f"\n***** self.program_YUV420_interlaced = {self.program_YUV420_interlaced} *****\n")
            except Exception as e:
                print(f'failed RGB shaders.compileProgram() {e}')
            try:
                shaders.glDeleteShader(vs)
                shaders.glDeleteShader(fs)
            except:
                pass

        if self.program_YUV420_interlaced_twotex is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_YUV420_interlaced_twotex, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_YUV420_interlaced_twotex = shaders.compileProgram(vs, fs, validate=False)
                self.print_log(f"\n***** self.program_YUV420_interlaced_twotex = {self.program_YUV420_interlaced_twotex} *****\n")
            except Exception as e:
                print(f'failed RGB shaders.compileProgram() {e}')
            try:
                shaders.glDeleteShader(vs)
                shaders.glDeleteShader(fs)
            except:
                pass

        if self.program_RAW is None:
            vs = shaders.compileShader(self.vertexShader, gl.GL_VERTEX_SHADER)
            fs = shaders.compileShader(self.fragmentShader_RAW, gl.GL_FRAGMENT_SHADER)
            try:
                self.program_RAW = shaders.compileProgram(vs, fs, validate=False)
                self.print_log(f"\n***** self.program_RAW = {self.program_RAW} *****\n")
            except Exception as e:
                print(f'failed RAW shaders.compileProgram() {e}')
            try:
                shaders.glDeleteShader(vs)
                shaders.glDeleteShader(fs)
            except:
                pass

    def set_crop(self, crop):
        if not np.array_equal(crop,self._output_crop):
            self._output_crop = crop
            self.setBufferData()

    def setVerticesBufferData(self):
        try:
            x0, x1, y0, y1 = self.image_centered_position()
            # print(" x0, x1, y0, y1 {} {} {} {}".format(x0, x1, y0, y1))
        except Exception as e:
            print(" Failed image_centered_position() {}".format(e))
            x0, x1, y0, y1 = 0, 100, 0, 100
            # set background vertices
        new_vb_params = [x0,x1,y0,y1]
        if self._vertex_buffer_param != new_vb_params:
            backgroundVertices = [
                x0, y1, 0.0,
                x0, y0, 0.0,
                x1, y1, 0.0,
                x1, y1, 0.0,
                x0, y0, 0.0,
                x1, y0, 0.0]
            vertexData = np.array(backgroundVertices, np.float32)

            if self._vertex_buffer is not None:
                self._vertex_buffer.destroy()
            self._vertex_buffer = QOpenGLBuffer()
            self._vertex_buffer.create()
            self._vertex_buffer.bind()
            self._vertex_buffer.allocate(vertexData, 4 * len(vertexData))
            self._vertex_buffer_param = new_vb_params

    def setBufferData(self):
        # To crop the texture, we may want to change these values
        # For example, in video frames we may need to crop at the right
        # set background UV
        backgroundUV = [
            self._output_crop[0], self._output_crop[1],
            self._output_crop[0], self._output_crop[3],
            self._output_crop[2], self._output_crop[1],
            self._output_crop[2], self._output_crop[1],
            self._output_crop[0], self._output_crop[3],
            self._output_crop[2], self._output_crop[3],
            ]
        uvData = np.array(backgroundUV, np.float32)

        self.uvBuffer = QOpenGLBuffer()
        self.uvBuffer.create()
        self.uvBuffer.bind()
        self.uvBuffer.allocate(uvData, 4 * len(uvData))

    def setTexture(self):
        texture_ok = super(GLImageViewerShaders, self).setTexture()
        self.setVerticesBufferData()
        return texture_ok

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # print(f"resizeGL {width}x{height}")
        if self.trace_calls:
            t = trace_method(self.tab)
        self._width = width*self.devicePixelRatio()
        self._height = height*self.devicePixelRatio()
        self.setVerticesBufferData()
        self.update()

    def initializeGL(self):
        """
        Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        self.start_timing()

        time1 = get_time()
        self.set_shaders()
        self.add_time('set_shaders', time1)

        self.setVerticesBufferData()
        self.setBufferData()
        self.print_timing()

    def viewer_update(self):
        self.update()

    def paintGL(self):
        self.paintAll()

    def myPaintGL(self):
        """Paint the scene.
        """
        if self._image is not None:
            if self.texture is None or not self.isValid():
                print("paintGL() not ready")
                return
        else:
            print("Image is None")
            return

        # Asserts that avoid syntax highlighting errors
        assert self.texture is not None

        self.opengl_error()
        self.start_timing()

        # _gl = QtGui.QOpenGLContext.currentContext().functions()
        _gl = gl
        _gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        twotex = self.texture_ref is not None and (self.texture.interlaced_uv==self.texture_ref.interlaced_uv) and (self._show_overlap or self._show_image_differences)
        if self._image and self._image.channels in ImageFormat.CH_RAWFORMATS():
            self.program = self.program_RAW
        elif self._image and self._image.channels == ImageFormat.CH_YUV420:
            if self.texture.interlaced_uv:
                self.program = self.program_YUV420_interlaced_twotex if twotex else self.program_YUV420_interlaced
            else:
                self.program = self.program_YUV420_twotex if twotex else self.program_YUV420
        else:
            # TODO: check for other types: scalar ...
            self.program = self.program_RGB

        # Obtain uniforms and attributes
        self.aVert              = shaders.glGetAttribLocation(self.program, "vert")
        self.aUV                = shaders.glGetAttribLocation(self.program, "uV")
        self.uPMatrix           = shaders.glGetUniformLocation(self.program, 'pMatrix')
        self.uMVMatrix          = shaders.glGetUniformLocation(self.program, "mvMatrix")
        if self._image and self._image.channels == ImageFormat.CH_YUV420:
            self.uYTex = shaders.glGetUniformLocation(self.program, "YTex")
            if twotex:
                self.uYTex2 = shaders.glGetUniformLocation(self.program, "YTex2")
            if self.texture.interlaced_uv:
                self.uUVTex = shaders.glGetUniformLocation(self.program, "UVTex")
                if twotex:
                    self.uUVTex2 = shaders.glGetUniformLocation(self.program, "UVTex2")
            else:
                self.uUTex = shaders.glGetUniformLocation(self.program, "UTex")
                self.uVTex = shaders.glGetUniformLocation(self.program, "VTex")
                if twotex:
                    self.uUTex2 = shaders.glGetUniformLocation(self.program, "UTex2")
                    self.uVTex2 = shaders.glGetUniformLocation(self.program, "VTex2")
            if twotex:
                self.difference_scaling    = shaders.glGetUniformLocation(self.program, "difference_scaling")
                self.texture_overlap_mode  = shaders.glGetUniformLocation(self.program, "overlap_mode")
                self.texture_cursor_x      = shaders.glGetUniformLocation(self.program, "cursor_x")
                self.texture_cursor_y      = shaders.glGetUniformLocation(self.program, "cursor_y")
            self.texture_scale_location = shaders.glGetUniformLocation(self.program, "texture_scale")
        else:
            self.uBackgroundTexture = shaders.glGetUniformLocation(self.program, "backgroundTexture")
        self.channels_location    = shaders.glGetUniformLocation(self.program, "channels")
        self.black_level_location = shaders.glGetUniformLocation(self.program, "black_level")
        self.white_level_location = shaders.glGetUniformLocation(self.program, "white_level")
        self.g_r_coeff_location   = shaders.glGetUniformLocation(self.program, "g_r_coeff")
        self.g_b_coeff_location   = shaders.glGetUniformLocation(self.program, "g_b_coeff")
        self.max_value_location   = shaders.glGetUniformLocation(self.program, "max_value")
        self.max_type_location    = shaders.glGetUniformLocation(self.program, "max_type")
        self.gamma_location       = shaders.glGetUniformLocation(self.program, "gamma")

        # use shader program
        self.print_log("self.program = {}".format(self.program))
        shaders.glUseProgram(self.program)

        # set uniforms
        gl.glUniformMatrix4fv(self.uPMatrix, 1, gl.GL_FALSE, self.pMatrix)
        gl.glUniformMatrix4fv(self.uMVMatrix, 1, gl.GL_FALSE, self.mvMatrix)
        if self._image and self._image.channels == ImageFormat.CH_YUV420:
            _gl.glUniform1i(self.uYTex, 0)
            if twotex:
                _gl.glUniform1i(self.uYTex2, 1)
            if self.texture.interlaced_uv:
                _gl.glUniform1i(self.uUVTex, 2)
                if twotex:
                    _gl.glUniform1i(self.uUVTex2, 3)
            else:
                _gl.glUniform1i(self.uUTex, 2)
                _gl.glUniform1i(self.uVTex, 4)
                if twotex:
                    _gl.glUniform1i(self.uUTex2, 3)
                    _gl.glUniform1i(self.uVTex2, 5)
            match self._image.data.dtype:
                case np.uint8:  texture_scale = 1
                # Normalize image to full intensity range
                case np.uint16: texture_scale = 1<<(16-self._image.precision)
                case _: texture_scale = 1
            # print(f"-- texture_scale = {texture_scale}")
            if twotex:
                _gl.glUniform1f( self.difference_scaling,    self.filter_params.imdiff_factor.float if self._show_image_differences else -1)
                _gl.glUniform1i( self.texture_overlap_mode,  self._overlap_mode.value                                                      )
                if self._show_overlap:
                    _gl.glUniform1f( self.texture_cursor_x, self.cursor_imx_ratio )
                    _gl.glUniform1f( self.texture_cursor_y, self.cursor_imy_ratio )
                else:
                    _gl.glUniform1f( self.texture_cursor_x, 1 )
                    _gl.glUniform1f( self.texture_cursor_y, 1 )
            _gl.glUniform1f( self.texture_scale_location, texture_scale)
        else:
            _gl.glUniform1i(self.uBackgroundTexture, 0)

        _gl.glUniform1i( self.channels_location, self._image.channels.value)

        # set color transformation parameters
        self.print_log("levels {} {}".format(self.filter_params.black_level.value,
                                             self.filter_params.white_level.value))
        _gl.glUniform1f( self.black_level_location, self.filter_params.black_level.float)
        _gl.glUniform1f( self.white_level_location, self.filter_params.white_level.float)

        # white balance coefficients
        _gl.glUniform1f(self.g_r_coeff_location, self.filter_params.g_r.float)
        _gl.glUniform1f(self.g_b_coeff_location, self.filter_params.g_b.float)

        # Should work for unsigned types for the moment
        _gl.glUniform1f( self.max_value_location, (1 << self._image.precision)-1)
        _gl.glUniform1f( self.max_type_location,  np.iinfo(self._image.data.dtype).max)

        _gl.glUniform1f( self.gamma_location,       self.filter_params.gamma.float)

        # enable attribute arrays
        _gl.glEnableVertexAttribArray(self.aVert)
        _gl.glEnableVertexAttribArray(self.aUV)

        # set vertex and UV buffers
        # vert_buffers = VertexBuffers()
        # vert_buffers.vert_pos_buffer = vert_pos_buffer
        # vert_buffers.normal_buffer = normal_buffer
        # vert_buffers.tex_coord_buffer = tex_coord_buffer
        # vert_buffers.amount_of_vertices = int(len(index_array) / 3)

        _gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vertex_buffer.bufferId())
        gl.glVertexAttribPointer(self.aVert, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        _gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.uvBuffer.bufferId())
        gl.glVertexAttribPointer(self.aUV, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # bind background texture
        # gl.glActiveTexture(gl.GL_TEXTURE0)
        if self._image and self._image.channels == ImageFormat.CH_YUV420 and self.texture:
            _gl.glActiveTexture(gl.GL_TEXTURE0)
            _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.textureY)
            if twotex:
                _gl.glActiveTexture(gl.GL_TEXTURE1)
                _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_ref.textureY)
            if self.texture.interlaced_uv:
                _gl.glActiveTexture(gl.GL_TEXTURE2)
                _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.textureUV)
                if twotex:
                    _gl.glActiveTexture(gl.GL_TEXTURE3)
                    _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_ref.textureUV)
            else:
                _gl.glActiveTexture(gl.GL_TEXTURE2)
                _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.textureU)
                _gl.glActiveTexture(gl.GL_TEXTURE4)
                _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.textureV)
                if twotex:
                    _gl.glActiveTexture(gl.GL_TEXTURE3)
                    _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_ref.textureU)
                    _gl.glActiveTexture(gl.GL_TEXTURE5)
                    _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_ref.textureV)
        else:
            if self.texture:
                _gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture.textureID)
        _gl.glEnable(gl.GL_TEXTURE_2D)

        # draw
        _gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)

        # disable attribute arrays
        _gl.glDisableVertexAttribArray(self.aVert)
        _gl.glDisableVertexAttribArray(self.aUV)
        _gl.glDisable(gl.GL_TEXTURE_2D)

        shaders.glUseProgram(0)

        self.print_timing(force=True)

    def updateTransforms(self, make_current=False, force=True) -> float:
        if self.trace_calls:
            t = trace_method(self.tab)
        if self.display_timing:
            start_time = get_time()
        w = self._width
        h = self._height
        dx, dy = self.new_translation()
        # Deduce new scale from mouse vertical displacement
        scale = self.new_scale(-self.mouse_zoom_displ.y(), self.texture.height)
        new_transform_params = [w,h,dx,dy,scale]
        if self._transform_param != new_transform_params or force:
            # update the window size
            if make_current:
                self.makeCurrent()
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            translation_unit = min(w, h)/2
            gl.glScale(scale, scale, scale)
            gl.glTranslate(dx/translation_unit, dy/translation_unit, 0)
            # the window corner OpenGL coordinates are (-+1, -+1)
            gl.glOrtho(0, w, 0, h, -1, 1)
            self.pMatrix = np.array(gl.glGetFloatv(gl.GL_PROJECTION_MATRIX), dtype=np.float32).flatten()

            gl.glMatrixMode(gl.GL_MODELVIEW)
            gl.glLoadIdentity()
            self.mvMatrix = np.array(gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX), dtype=np.float32).flatten()
            self._transform_param = new_transform_params
        if self.display_timing:
            self.print_log('updateTransforms time {:0.1f} ms'.format((get_time()-start_time)*1000))
        return scale


if __name__ == '__main__':
    from qimview.image_readers import gb_image_reader
    # import numpy for generating random data points
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_image', help='input image')
    args = parser.parse_args()
    _params = vars(args)

    # define a Qt window with an OpenGL widget inside it
    # class TestWindow(QtGui.QMainWindow):
    class TestWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()
            self.widget = GLImageViewerShaders(self)
            self.show()
        def load(self):
            im = gb_image_reader.read(_params['input_image'])
            self.widget.set_image(im)
            self.widget.image_name = _params['input_image']
            # put the window at the screen position (100, 100)
            self.setGeometry(0, 0, self.widget._width, self.widget._height)
            self.setCentralWidget(self.widget)

    # create the Qt App and window
    app = QtWidgets.QApplication(sys.argv)
    window = TestWindow()
    window.load()
    window.show()
    app.exec()