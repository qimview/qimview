
import os

ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib
import numpy as np
import time
from qimview.utils.qt_imports                          import QtWidgets, QtCore, QtGui
from qimview.image_viewers.gl_image_viewer_shaders     import GLImageViewerShaders
from qimview.utils.viewer_image                        import ViewerImage, ImageFormat
from qimview.parameters.numeric_parameter              import NumericParameter
from qimview.parameters.numeric_parameter_gui          import NumericParameterGui
from qimview.image_viewers.image_filter_parameters     import ImageFilterParameters
from qimview.image_viewers.image_filter_parameters_gui import ImageFilterParametersGui


class TestVideoPlayer(QtWidgets.QMainWindow):
    def __init__(self, filename1, filename2, device_type) -> None:
        super().__init__()
        self.filename1 = filename1
        self.filename2 = filename2
        self.device_type = device_type

        self.main_widget = QtWidgets.QWidget()
        vertical_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(vertical_layout)

        self.widget = GLImageViewerShaders()
        self.widget.show_histogram = False
        self.widget._show_text = False
        self.setGeometry(0, 0, self.widget.width(), self.widget.height())
        self.setCentralWidget(self.main_widget)

        filters_layout = self._add_filters()

        hor_layout = QtWidgets.QHBoxLayout()
        self._add_play_pause_button(       hor_layout)
        self._add_playback_speed_slider(   hor_layout)
        self._add_playback_position_slider(hor_layout)

        vertical_layout.addLayout(filters_layout)
        vertical_layout.addWidget(self.widget)
        vertical_layout.addLayout(hor_layout)

        self._pause = True
        self._button_play_pause.clicked.connect(self.play_pause)

        self.video_decoder1 = None
        self.video_decoder2 = None
        
        self.slow_down = 1
        
        self.show()
        
    def setSlowDown(self,s):
        self.slow_down = s

    def _add_filters(self):
        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params, name="TestViewer")

        filters_layout = QtWidgets.QHBoxLayout()
        # Add color difference slider
        self.filter_params_gui.add_imdiff_factor(filters_layout, self.update_image_intensity_event)

        self.filter_params_gui.add_blackpoint(filters_layout, self.update_image_intensity_event)
        # white point adjustment
        self.filter_params_gui.add_whitepoint(filters_layout, self.update_image_intensity_event)
        # Gamma adjustment
        self.filter_params_gui.add_gamma(filters_layout, self.update_image_intensity_event)
        # G_R adjustment
        self.filter_params_gui.add_g_r(filters_layout, self.update_image_intensity_event)
        # G_B adjustment
        self.filter_params_gui.add_g_b(filters_layout, self.update_image_intensity_event)

        return filters_layout

    def update_image_intensity_event(self):
        self.widget.filter_params.copy_from(self.filter_params)
        # print(f"parameters {self.filter_params}")
        self.widget.viewer_update()

    def _add_play_pause_button(self, hor_layout):
        self._button_play_pause = QtWidgets.QPushButton()
        self._icon_play = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        self._icon_pause = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        self._button_play_pause.setIcon(self._icon_play)
        hor_layout.addWidget(self._button_play_pause)

    def _add_playback_speed_slider(self, hor_layout):
        # Playback speed slider
        self.playback_speed = NumericParameter()
        self.playback_speed.float_scale = 100
        self.playback_speed_gui = NumericParameterGui(name="x", param=self.playback_speed)
        self.playback_speed_gui.decimals = 1
        self.playback_speed_gui.set_pressed_callback(self.pause)
        self.playback_speed_gui.set_released_callback(self.reset_play)
        self.playback_speed_gui.set_valuechanged_callback(self.speed_value_changed)
        self.playback_speed_gui.create()
        self.playback_speed_gui.setTextFormat(lambda p: f"{pow(2,p.float):0.2f}")
        self.playback_speed_gui.setRange(-300, 300)
        self.playback_speed_gui.update()
        self.playback_speed_gui.updateText()
        self.playback_speed_gui.add_to_layout(hor_layout,1)
        self.playback_speed_gui.setSingleStep(1)
        self.playback_speed_gui.setPageStep(10)
        self.playback_speed_gui.setTickInterval(10)
        self.playback_speed_gui.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)

    def _add_playback_position_slider(self, hor_layout):
        # Position slider
        self.play_position = NumericParameter()
        self.play_position.float_scale = 1000
        self.play_position_gui = NumericParameterGui(name="sec:", param=self.play_position)
        self.play_position_gui.decimals = 3
        self.play_position_gui.set_pressed_callback(self.pause)
        self.play_position_gui.set_released_callback(self.reset_play)
        self.play_position_gui.set_valuechanged_callback(self.slider_value_changed)
        self.play_position_gui.create()
        self.play_position_gui.add_to_layout(hor_layout,5)

    def pause(self):
        """ Method called from video player """
        # self._was_active = self._scheduler._timer.isActive()
        # self._scheduler.pause()
        pass

    def reset_play(self):
        # if self._was_active:
        #     self._scheduler.play()
        pass

    def play_pause(self):
        if self.video_decoder1 is None:
            # device_type = "cuda" for HW decoding
            self.open_video1( self.filename1, self.device_type)
            if self.filename2:
                self.open_video2( self.filename2, self.device_type)
        self.decode()

    def slider_value_changed(self):
        self.set_play_position()

    def set_play_position(self):
        print(f"self.play_position {self.play_position.float}")
        # if self._frame_provider.frame_buffer:
        #     self._frame_provider.frame_buffer.reset()
        # self._frame_provider.set_time(self.play_position.float)
        # self._start_video_time = self.play_position.float
        # self.display_frame(self._frame_provider.frame)

    def speed_value_changed(self):
        print(f"New speed value {self.playback_speed.float}")
        # self._scheduler.set_playback_speed(pow(2,self.playback_speed.float))

    def open_video1(self, filename, device_type : str|None):
        self.video_decoder1 = decode_lib.VideoDecoder()
        self.video_decoder1.open(filename, device_type)

        self._video_stream1 = self.video_decoder1.getStream()
        codecpar = self._video_stream1.codecpar
        shape = (codecpar.height, codecpar.width)
        print(f"Video dimensions (h,w)={shape}")

        st = self._video_stream1
        # Assume all frame have the same time base as the video stream
        # fps = self.stream.base_rate, base_rate vs average_rate
        print(f"FPS avg:{st.avg_frame_rate} base:{st.r_frame_rate}")
        # --- Initialize several constants/parameters for this video
        self._framerate       = float(st.avg_frame_rate) # get the frame rate
        self._time_base       = float(st.time_base)
        self._frame_duration  = float(1/self._framerate)
        self._ticks_per_frame = int(self._frame_duration / self._time_base)
        self._duration        = float(st.duration * self._time_base)
        self._end_time        = float(self._duration-self._frame_duration)

    def open_video2(self, filename, device_type : str|None):
        self.video_decoder2 = decode_lib.VideoDecoder()
        self.video_decoder2.open(filename, device_type)

    def decode(self):

        assert self.video_decoder1 is not None
        self.st = time.perf_counter()

        self.display_frames(0,0)
        self.display_frames(1,1000)

    
    def display_frames(self, current, max):
        # if current%10==0:
        #     print(f"{current}")
        assert self.video_decoder1 is not None
        if current%1 == 0:
            self.video_decoder1.nextFrame(convert=True)
            self.frame1 = self.video_decoder1.getFrame()
            # print(self.frame1.get().pict_type)
            if self.video_decoder2:
                self.video_decoder2.nextFrame(convert=True)
                self.frame2 = self.video_decoder2.getFrame()
            else:
                self.frame2 = None
            self.display_frame(current)
        else:
            self.video_decoder1.nextFrame(convert=False)
            if self.video_decoder2:
                self.video_decoder2.nextFrame(convert=False)
        if current<max:
            QtCore.QTimer.singleShot(self.slow_down, lambda : self.display_frames(current+1, max))
        else:
            duration = time.perf_counter()-self.st
            print(f"took {duration:0.3f} sec {max/duration:0.2f} fps")

    def display_frames2(self, current, max):
        # if current%10==0:
        #     print(f"{current}")
        assert self.video_decoder1 is not None
        for i in range(current, max+1):
            self.video_decoder1.nextFrame()
            self.frame1 = self.video_decoder1.getFrame()
            self.display_frame(i)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
        duration = time.perf_counter()-self.st
        print(f"took {duration:0.3f} sec {max/duration:0.2f} fps")

    def display_frame(self, frame_nb):

        def getArray(frame, index, height, width, dtype):
            dtype_size = np.dtype(dtype).itemsize
            mem = frame.getData(index, height, width)
            linesize = int(frame.getLinesize()[index]/dtype_size)
            array = np.frombuffer(mem, dtype=dtype).reshape(-1, linesize)
            return mem, array

        height1, width1 = self.frame1.getShape()
        if self.frame2:
            height2, width2 = self.frame2.getShape()
            assert self.frame1.getFormat() == self.frame2.getFormat(), "Videos have different frame formats"
        match self.frame1.getFormat():
            case decode_lib.AVPixelFormat.AV_PIX_FMT_P010LE:
                # Create numpy array from Y and UV
                self.memY1,  self.Y1  = getArray(self.frame1, 0, height1,    width1, np.uint16)
                self.memUV1, self.UV1 = getArray(self.frame1, 1, height1//2, width1, np.uint16)

                prec=16
                self._im1 = ViewerImage(self.Y1, channels = ImageFormat.CH_YUV420, precision=prec)
                self._im1.uv = self.UV1
                self._im2 = None
            case decode_lib.AVPixelFormat.AV_PIX_FMT_YUV420P10LE:
                # Create numpy array from Y, U and V
                self.memY1, self.Y1 = getArray(self.frame1, 0, height1,    width1,    np.uint16)
                self.memU1, self.U1 = getArray(self.frame1, 1, height1//2, width1//2, np.uint16)
                self.memV1, self.V1 = getArray(self.frame1, 2, height1//2, width1//2, np.uint16)

                prec=10
                self._im1 = ViewerImage(self.Y1, channels = ImageFormat.CH_YUV420, precision=prec)
                self._im1.u = self.U1
                self._im1.v = self.V1
                self._im2 = None
            case decode_lib.AVPixelFormat.AV_PIX_FMT_YUVJ420P | decode_lib.AVPixelFormat.AV_PIX_FMT_YUV420P:
                # Create numpy array from Y, U and V
                prec=8
                self.memY1, self.Y1 = getArray(self.frame1, 0, height1,    width1,    np.uint8)
                self.memU1, self.U1 = getArray(self.frame1, 1, height1//2, width1//2, np.uint8)
                self.memV1, self.V1 = getArray(self.frame1, 2, height1//2, width1//2, np.uint8)
                self._im1 = ViewerImage(self.Y1, channels = ImageFormat.CH_YUV420, precision=prec)
                self._im1.u = self.U1
                self._im1.v = self.V1
                if self.frame2:
                    self.memY2, self.Y2 = getArray(self.frame2, 0, height2,    width2,    np.uint8)
                    self.memU2, self.U2 = getArray(self.frame2, 1, height2//2, width2//2, np.uint8)
                    self.memV2, self.V2 = getArray(self.frame2, 2, height2//2, width2//2, np.uint8)
                    self._im2 = ViewerImage(self.Y2, channels = ImageFormat.CH_YUV420, precision=prec)
                    self._im2.u = self.U2
                    self._im2.v = self.V2
                else:
                    self._im2 = None
            case decode_lib.AVPixelFormat.AV_PIX_FMT_NV12:
                prec=8
                self.memY1,  self.Y1  = getArray(self.frame1, 0, height1,    width1, np.uint8)
                self.memUV1, self.UV1 = getArray(self.frame1, 1, height1//2, width1, np.uint8)
                self._im1 = ViewerImage(self.Y1, channels = ImageFormat.CH_YUV420, precision=prec)
                self._im1.uv = self.UV1

                if self.frame2:
                    self.memY2,  self.Y2  = getArray(self.frame2, 0, height2,    width2, np.uint8)
                    self.memUV2, self.UV2 = getArray(self.frame2, 1, height2//2, width2, np.uint8)
                    self._im2 = ViewerImage(self.Y2, channels = ImageFormat.CH_YUV420, precision=prec)
                    self._im2.uv = self.UV2
                else:
                    self._im2 = None

            case _:
                assert False, f"frame format {self.frame1.getFormat()} not available"

        self.widget.set_image_fast(self._im1, image_ref=self._im2)
        self.widget.image_name = "Test frame"
        self.widget.viewer_update()

def main():

    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_videos', help='input videos',  nargs='+')
    parser.add_argument('-c','--cuda', action='store_true', help='use cuda hardware')
    parser.add_argument('-s','--slow', type=int, default=1, help='slow down time in ms between frames')
    args = parser.parse_args()

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    format = QtGui.QSurfaceFormat.defaultFormat()
    format.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(format)

    app = QtWidgets.QApplication()

    device_type = "cuda" if args.cuda else None
    if len(args.input_videos)>2:
        print("Warning: Using only first 2 videos ...")
    if len(args.input_videos)==1:
        window = TestVideoPlayer(args.input_videos[0], None, device_type)
    else:
        window = TestVideoPlayer(args.input_videos[0], args.input_videos[1], device_type)
    window.setSlowDown(args.slow)
    window.show()

    app.exec()


if __name__ == '__main__':
    main()
