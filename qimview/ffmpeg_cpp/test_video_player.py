
import os

ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib
import numpy as np
import time
from qimview.utils.qt_imports                          import QtWidgets, QtCore, QtGui
from qimview.utils.viewer_image                        import ViewerImage, ImageFormat
from qimview.video_player.video_player_base            import VideoPlayerBase
from qimview.video_player.video_frame_buffer_cpp       import VideoFrameBufferCpp


class TestVideoPlayer(VideoPlayerBase):
    def __init__(self, parent, filename1, filename2, device_type) -> None:
        super().__init__(parent)
        self.filename1 = filename1
        self.filename2 = filename2
        self.device_type = device_type

        self.main_widget = QtWidgets.QWidget()
        vertical_layout = QtWidgets.QVBoxLayout()
        self.main_widget.setLayout(vertical_layout)

        self._button_play_pause.clicked.connect(self.play_pause)

        self.video_decoder1 = None
        self.video_decoder2 = None
        
        self.slow_down = 1

        self._video_framebuffer1 = None
        self._video_framebuffer2 = None
                
        self.show()
        
    def setSlowDown(self,s):
        self.slow_down = s

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
        if self.video_decoder1:
            self.video_decoder1.seek(self.play_position.float)
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
        self._video_framebuffer1 = VideoFrameBufferCpp(self.video_decoder1)

        # TODO: align code, in VideoPlayerAV, some values are in frame_provider
        print(f"duration = {self._duration} seconds")
        slider_single_step = int(self._ticks_per_frame*
                                 self._time_base*
                                 self.play_position.float_scale+0.5)
        slider_page_step   = int(self.play_position.float_scale+0.5)
        self.play_position_gui.setSingleStep(slider_single_step)
        self.play_position_gui.setPageStep(slider_page_step)
        self.play_position.range = [0, int(self._end_time*
                                           self.play_position.float_scale)]
        print(f"range = {self.play_position.range}")
        self.play_position_gui.setRange(0, self.play_position.range[1])
        self.play_position_gui.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        self.play_position_gui.update()
        self.play_position_gui.changed()

        print(f"ticks_per_frame {self._ticks_per_frame}")


    def open_video2(self, filename, device_type : str|None):
        self.video_decoder2 = decode_lib.VideoDecoder()
        self.video_decoder2.open(filename, device_type)
        self._video_framebuffer2 = VideoFrameBufferCpp(self.video_decoder2)

    def decode(self):

        assert self.video_decoder1 is not None
        self.start_time = time.perf_counter()
        self.frame_count = 0

        self.get_frames(display=True)
        self.display_frames()


    def get_frames(self, display=True):
        if display:
            self.frame1 = self._video_framebuffer1.get_frame()
            # print(self.frame1.get().pict_type)
            if self.video_decoder2:
                self.frame2 = self._video_framebuffer2.get_frame()
            else:
                self.frame2 = None
            self.display_frame()
            self.frame_count += 1
        else:
            self.video_decoder1.nextFrame(convert=False)
            if self.video_decoder2:
                self.video_decoder2.nextFrame(convert=False)

    def display_frames(self):
        # if current%10==0:
        #     print(f"{current}")
        assert self.video_decoder1 is not None
        try:
            self.get_frames(display=True)
            QtCore.QTimer.singleShot(self.slow_down, lambda : self.display_frames())
        except Exception as e:
            print(f"Exception: {e}")
            duration = time.perf_counter()-self.start_time
            # TODO: get frame number from Frame
            fps = self.frame_count/duration
            print(f"took {duration:0.3f} sec  {fps:0.2f} fps")

    def display_frames2(self, current, max):
        # if current%10==0:
        #     print(f"{current}")
        assert self._video_framebuffer1 is not None
        for i in range(current, max+1):
            self.frame1 = self._video_framebuffer1.get_frame()
            self.display_frame(i)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 1)
        duration = time.perf_counter()-self.st
        print(f"took {duration:0.3f} sec {max/duration:0.2f} fps")

    def display_frame(self):

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
    app.setApplicationDisplayName(f'Video comparison: {args.input_videos}')

    device_type = "cuda" if args.cuda else None
    if len(args.input_videos)>2:
        print("Warning: Using only first 2 videos ...")

    main_window = QtWidgets.QMainWindow()
    main_widget = QtWidgets.QWidget(main_window)
    main_window.setCentralWidget(main_widget)
    main_layout  = QtWidgets.QVBoxLayout()
    video_layout = QtWidgets.QHBoxLayout()
    main_widget.setLayout(main_layout)
    main_layout.addLayout(video_layout)

    if len(args.input_videos)==1:
        player1 = TestVideoPlayer(main_widget, args.input_videos[0], None, device_type)
    else:
        player1 = TestVideoPlayer(main_widget, args.input_videos[0], args.input_videos[1], device_type)
    video_layout.addWidget(player1, 1)
    player1.setSlowDown(args.slow)
    player1.show()

    main_window.show()
    app.exec()


if __name__ == '__main__':
    main()
