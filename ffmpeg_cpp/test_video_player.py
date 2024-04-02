
from build.Release import decode_video_py as decode_lib
import numpy as np
import time
from qimview.utils.qt_imports import QtWidgets, QtCore, QtGui
from qimview.image_viewers.gl_image_viewer_shaders import GLImageViewerShaders
from qimview.utils.viewer_image import ViewerImage, ImageFormat
from qimview.image_viewers.image_filter_parameters import ImageFilterParameters
from qimview.image_viewers.image_filter_parameters_gui import ImageFilterParametersGui


class TestVideoPlayer(QtWidgets.QMainWindow):
    def __init__(self, filename, device_type) -> None:
        super().__init__()
        self.filename = filename
        self.device_type = device_type

        self.main_widget = QtWidgets.QWidget()
        vertical_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(vertical_layout)

        self.widget = GLImageViewerShaders()
        self.widget.show_histogram = False
        self.widget._show_text = False
        self.setGeometry(0, 0, self.widget.width(), self.widget.height())
        self.setCentralWidget(self.main_widget)

        self.filter_params = ImageFilterParameters()
        self.filter_params_gui = ImageFilterParametersGui(self.filter_params, name="TestViewer")

        hor_layout = QtWidgets.QHBoxLayout()
        self.filter_params_gui.add_blackpoint(hor_layout, self.update_image_intensity_event)
        # white point adjustment
        self.filter_params_gui.add_whitepoint(hor_layout, self.update_image_intensity_event)
        # Gamma adjustment
        self.filter_params_gui.add_gamma(hor_layout, self.update_image_intensity_event)
        # G_R adjustment
        self.filter_params_gui.add_g_r(hor_layout, self.update_image_intensity_event)
        # G_B adjustment
        self.filter_params_gui.add_g_b(hor_layout, self.update_image_intensity_event)

        vertical_layout.addLayout(hor_layout)
        vertical_layout.addWidget(self.widget)

        # Add play button
        hor_layout = QtWidgets.QHBoxLayout()
        self._button_play_pause = QtWidgets.QPushButton()
        self._icon_play = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        self._icon_pause = self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPause)
        self._button_play_pause.setIcon(self._icon_play)
        hor_layout.addWidget(self._button_play_pause)
        vertical_layout.addLayout(hor_layout)

        self._pause = True
        self._button_play_pause.clicked.connect(self.play_pause)

        self.video_decoder = None
        self.show()

    def update_image_intensity_event(self):
        self.widget.filter_params.copy_from(self.filter_params)
        # print(f"parameters {self.filter_params}")
        self.widget.viewer_update()

    def play_pause(self):
        if self.video_decoder is None:
            # device_type = "cuda" for HW decoding
            self.open_video( self.filename, self.device_type)
        self.decode()

    def open_video(self, filename, device_type : str|None):
        self.video_decoder = decode_lib.VideoDecoder()
        self.video_decoder.open(filename, device_type)

        self._video_stream = self.video_decoder.getStream()
        codecpar = self._video_stream.codecpar
        shape = (codecpar.height, codecpar.width)
        print(f"Video dimensions (h,w)={shape}")

        st = self._video_stream
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

    def decode(self):

        assert self.video_decoder is not None
        self.st = time.perf_counter()

        self.display_frames(0,0)
        self.display_frames(1,1000)

    
    def display_frames(self, current, max):
        # if current%10==0:
        #     print(f"{current}")
        assert self.video_decoder is not None
        if current%1 == 0:
            self.video_decoder.nextFrame(convert=True)
            self.frame = self.video_decoder.getFrame()
            self.display_frame()
        else:
            self.video_decoder.nextFrame(convert=False)
        if current<max:
            QtCore.QTimer.singleShot(1, lambda : self.display_frames(current+1, max))
        else:
            duration = time.perf_counter()-self.st
            print(f"took {duration:0.3f} sec {max/duration:0.2f} fps")

    def display_frames2(self, current, max):
        # if current%10==0:
        #     print(f"{current}")
        assert self.video_decoder is not None
        for i in range(current, max+1):
            self.video_decoder.nextFrame()
            self.frame = self.video_decoder.getFrame()
            self.display_frame()
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
        duration = time.perf_counter()-self.st
        print(f"took {duration:0.3f} sec {max/duration:0.2f} fps")



    def display_frame(self):

        def getArray(index, height, width, dtype):
            dtype_size = np.dtype(dtype).itemsize
            mem = self.frame.getData(index, height, width)
            linesize = int(self.frame.getLinesize()[index]/dtype_size)
            array = np.frombuffer(mem, dtype=dtype).reshape(-1, linesize)
            return mem, array

        height, width = self.frame.getShape()
        match self.frame.getFormat():
            case decode_lib.AVPixelFormat.AV_PIX_FMT_P010LE:
                # Create numpy array from Y and UV
                self.memY,  self.Y  = getArray(0, height,    width, np.uint16)
                self.memUV, self.UV = getArray(1, height//2, width, np.uint16)

                prec=16
                self._im = ViewerImage(self.Y, channels = ImageFormat.CH_YUV420, precision=prec)
                self._im.uv = self.UV
            case  decode_lib.AVPixelFormat.AV_PIX_FMT_YUV420P10LE:
                # Create numpy array from Y, U and V
                self.memY, self.Y = getArray(0, height,    width,    np.uint16)
                self.memU, self.U = getArray(1, height//2, width//2, np.uint16)
                self.memV, self.V = getArray(2, height//2, width//2, np.uint16)

                prec=10
                self._im = ViewerImage(self.Y, channels = ImageFormat.CH_YUV420, precision=prec)
                self._im.u = self.U
                self._im.v = self.V
            case _:
                assert False, f"frame format {self.frame.getFormat()} not available"

        self.widget.set_image_fast(self._im)
        self.widget.image_name = "Test frame"
        self.widget.viewer_update()

def main():

    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video', help='input video')
    parser.add_argument('-c','--cuda', action='store_true', help='use cuda hardware')
    args = parser.parse_args()

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    format = QtGui.QSurfaceFormat.defaultFormat()
    format.setSwapInterval(0)
    QtGui.QSurfaceFormat.setDefaultFormat(format)

    app = QtWidgets.QApplication()

    device_type = "cuda" if args.cuda else None
    window = TestVideoPlayer(args.input_video, device_type)
    window.show()

    app.exec()


if __name__ == '__main__':
    main()
