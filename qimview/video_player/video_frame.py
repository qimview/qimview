"""
    Class VideoFrame will deal with frames from either pyav or decode_lib (bound ffmpeg)
    and convert the frames to ViewerImage data
"""

import os
from typing import Optional
import numpy as np

if os.name == 'nt' and os.path.isdir("c:\\ffmpeg\\bin"):
    os.add_dll_directory("c:\\ffmpeg\\bin")
import decode_video_py as decode_lib
from cv2 import cvtColor, COLOR_YUV2RGB_I420 # type: ignore
import av
from av.video.frame import VideoFrame as AVVideoFrame
from qimview.utils.viewer_image  import ViewerImage, ImageFormat

class VideoFrame:

    @staticmethod
    def useful_array(plane, crop=True, dtype=np.uint8):
        """
        Return the useful part of the VideoPlane as a single dimensional array.

        We are simply discarding any padding which was added for alignment.
        """
        total_line_size = int(abs(plane.line_size)/dtype().itemsize)
        arr = np.frombuffer(plane, dtype).reshape(-1, total_line_size)
        if crop:
            arr = arr[:,:plane.width]
            return np.ascontiguousarray(arr)
        else:
            return arr

    @staticmethod
    def to_ndarray_v1(frame, yuv_array: np.ndarray, crop=False) -> Optional[np.ndarray]:
        match frame.format.name:
            case 'yuv420p' | 'yuvj420p':
                dtype = np.uint8
            case 'yuv420p10le':
                dtype = np.uint16
            case _:
                dtype = None
        if dtype is not None:
            # assert frame.width % 2 == 0
            # assert frame.height % 2 == 0
            # assert frame.planes[0].line_size == 2*frame.planes[1].line_size
            # assert frame.planes[0].width     == 2*frame.planes[1].width
            # assert frame.planes[1].line_size == frame.planes[2].line_size
            # assert frame.planes[1].width     == frame.planes[2].width
            # width = frame.planes[0].line_size
            v0 = VideoFrame.useful_array(frame.planes[0], crop=crop, dtype=dtype).ravel()
            v1 = VideoFrame.useful_array(frame.planes[1], crop=crop, dtype=dtype).ravel()
            v2 = VideoFrame.useful_array(frame.planes[2], crop=crop, dtype=dtype).ravel()
            total_size = v0.size+ v1.size + v2.size
            if yuv_array.size != total_size:
                output_array = np.empty((total_size,), dtype=dtype)
            else:
                output_array = yuv_array
            output_array[0:v0.size]                                   = v0
            output_array[v0.size:(v0.size+v1.size)]                   = v1
            output_array[(v0.size+v1.size):(v0.size+v1.size+v2.size)] = v2
            return output_array
            # if output_array.size == total_size:
            # else:
            #     # print(f"{v0.shape} {v1.shape} {v2.shape}")
            #     return np.hstack((v0, v1, v2)).reshape(-1, width)
        else:
            return None

    @staticmethod
    def to_yuv(frame) -> Optional[list[np.ndarray]]:
        match frame.format.name:
            case 'yuv420p' | 'yuvj420p':
                dtype = np.uint8
            case 'yuv420p10le':
                dtype = np.uint16
            case _:
                print(f"Unknow format {frame.format.name}")
                dtype = None
        if dtype is not None:
            y = VideoFrame.useful_array(frame.planes[0], crop=False, dtype=dtype)
            u = VideoFrame.useful_array(frame.planes[1], crop=False, dtype=dtype)
            v = VideoFrame.useful_array(frame.planes[2], crop=False, dtype=dtype)
            return [y,u,v]
        else:
            return None

    def __init__(self, frame:decode_lib.Frame | AVVideoFrame):
        self._frame : decode_lib.Frame | AVVideoFrame = frame
        # Pre-allocated array to avoid allocation for each new frame
        self._yuv_array : np.ndarray = np.empty((1), dtype=np.uint8)


    def _libFrameToViewer(self) -> ViewerImage | None:
        linesizeall = self._frame.getLinesizeAll()
        def getArray(frame, index, height, width, dtype):
            dtype_size = np.dtype(dtype).itemsize
            mem = frame.getData(index, height, width)
            linesize = int(linesizeall[index]/dtype_size)
            array = np.frombuffer(mem, dtype=dtype).reshape(-1, linesize)
            return mem, array

        height, width = self._frame.getShape()
        # if self.frame2:
        #     height2, width2 = self.frame2.getShape()
        #     assert self._frame.getFormat() == self.frame2.getFormat(), "Videos have different frame formats"
        match self._frame.getFormat():
            case decode_lib.AVPixelFormat.AV_PIX_FMT_P010LE:
                dtype, prec, hasUV = np.uint16, 16, True
            case decode_lib.AVPixelFormat.AV_PIX_FMT_YUV420P10LE:
                dtype, prec, hasUV = np.uint16, 10, False
            case decode_lib.AVPixelFormat.AV_PIX_FMT_YUVJ420P | decode_lib.AVPixelFormat.AV_PIX_FMT_YUV420P:
                dtype, prec, hasUV = np.uint8, 8, False
            case decode_lib.AVPixelFormat.AV_PIX_FMT_NV12:
                dtype, prec, hasUV = np.uint8, 8, True
            case _:
                assert False, f"frame format {self._frame.getFormat()} not available"
        # Create numpy array from Y and UV
        self.memY,Y  = getArray(self._frame, 0, height, width, dtype)
        im = ViewerImage(Y, channels = ImageFormat.CH_YUV420, precision=prec)
        if hasUV:
            self.memUV, UV = getArray(self._frame, 1, height//2, width, dtype)
            im.uv = UV
        else:
            self.memU, U = getArray(self._frame, 1, height//2, width//2, dtype)
            self.memV, V = getArray(self._frame, 2, height//2, width//2, dtype)
            im.u = U
            im.v = V
        dtype_size = np.dtype(dtype).itemsize
        linesize = int(linesizeall[0]/dtype_size)
        if width < linesize:
            # Apply crop on the right
            im.crop = np.array([0,0,width/linesize,1], dtype=np.float32)
        return im

    def _avFrameToViewer(self, rgb=False) -> ViewerImage | None:
        frame = self._frame
        if rgb:
            # Retun RGB Image
            crop_yuv=True
            self._yuv_array = VideoFrame.to_ndarray_v1(frame, self._yuv_array, crop=crop_yuv)
            if crop_yuv:
                a = cvtColor(self._yuv_array.reshape(-1,frame.planes[0].width), COLOR_YUV2RGB_I420)
            else:
                a = cvtColor(self._yuv_array.reshape(-1,frame.planes[0].line_size), COLOR_YUV2RGB_I420)
                a = a[:,:frame.width]
            if not a.flags.contiguous:
                a = np.ascontiguousarray(a)

            format = ImageFormat.CH_Y if len(a.shape) == 2 else ImageFormat.CH_RGB
            im = ViewerImage(a, channels = format, precision=8)
            return im
        else:
            # Retun YUV Image
            res = VideoFrame.to_yuv(frame)
            if res is not None:
                y,u,v = res
                prec = 8
                match y.dtype:
                    case np.uint8:  prec=8
                    case np.uint16: prec=10

                im = ViewerImage(y, channels = ImageFormat.CH_YUV420, precision=prec)
                im.u = u
                pl = frame.planes[0]
                im_width = y.data.shape[1]
                im.v = v
                if im_width != pl.width:
                    # Apply crop on the right
                    im.crop = np.array([0,0,pl.width/im_width,1], dtype=np.float32)
                else:
                    im.crop = np.array([0,0,1,1], dtype=np.float32)
                return im
        

    def toViewerImage(self, rgb=False) -> ViewerImage | None:
        if type(self._frame) is decode_lib.Frame:
            return self._libFrameToViewer()
        if type(self._frame) is AVVideoFrame:
            return self._avFrameToViewer(rgb)

