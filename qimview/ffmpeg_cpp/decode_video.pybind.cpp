#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "decode_video.hpp"
#include <string> 

namespace py = pybind11;


/**
 * Very convenient, with this code we can directly include C++ code for the viewer,
 * it will help improving the performance when we are not using OpenGL shader directly
 *
*/
/*
};
*/
PYBIND11_MODULE(decode_video_py, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

#define ADD_PIX_FMT(v) .value(#v, AVPixelFormat::v)

  py::enum_<AVPixelFormat>(m, "AVPixelFormat")
  ADD_PIX_FMT(AV_PIX_FMT_NONE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P)
  ADD_PIX_FMT(AV_PIX_FMT_YUYV422)
  ADD_PIX_FMT(AV_PIX_FMT_RGB24)
  ADD_PIX_FMT(AV_PIX_FMT_BGR24)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P)
  ADD_PIX_FMT(AV_PIX_FMT_YUV410P)
  ADD_PIX_FMT(AV_PIX_FMT_YUV411P)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY8)
  ADD_PIX_FMT(AV_PIX_FMT_MONOWHITE)
  ADD_PIX_FMT(AV_PIX_FMT_MONOBLACK)
  ADD_PIX_FMT(AV_PIX_FMT_PAL8)
  ADD_PIX_FMT(AV_PIX_FMT_YUVJ420P)
  ADD_PIX_FMT(AV_PIX_FMT_YUVJ422P)
  ADD_PIX_FMT(AV_PIX_FMT_YUVJ444P)
  ADD_PIX_FMT(AV_PIX_FMT_UYVY422)
  ADD_PIX_FMT(AV_PIX_FMT_UYYVYY411)
  ADD_PIX_FMT(AV_PIX_FMT_BGR8)
  ADD_PIX_FMT(AV_PIX_FMT_BGR4)
  ADD_PIX_FMT(AV_PIX_FMT_BGR4_BYTE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB8)
  ADD_PIX_FMT(AV_PIX_FMT_RGB4)
  ADD_PIX_FMT(AV_PIX_FMT_RGB4_BYTE)
  ADD_PIX_FMT(AV_PIX_FMT_NV12)
  ADD_PIX_FMT(AV_PIX_FMT_NV21)
  ADD_PIX_FMT(AV_PIX_FMT_ARGB)
  ADD_PIX_FMT(AV_PIX_FMT_RGBA)
  ADD_PIX_FMT(AV_PIX_FMT_ABGR)
  ADD_PIX_FMT(AV_PIX_FMT_BGRA)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY16BE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV440P)
  ADD_PIX_FMT(AV_PIX_FMT_YUVJ440P)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P)
  ADD_PIX_FMT(AV_PIX_FMT_RGB48BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB48LE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB565BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB565LE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB555BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB555LE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR565BE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR565LE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR555BE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR555LE)
  ADD_PIX_FMT(AV_PIX_FMT_VAAPI)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P16BE)
  ADD_PIX_FMT(AV_PIX_FMT_DXVA2_VLD)
  ADD_PIX_FMT(AV_PIX_FMT_RGB444LE)
  ADD_PIX_FMT(AV_PIX_FMT_RGB444BE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR444LE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR444BE)
  ADD_PIX_FMT(AV_PIX_FMT_YA8)
  ADD_PIX_FMT(AV_PIX_FMT_BGR48BE)
  ADD_PIX_FMT(AV_PIX_FMT_BGR48LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P9BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P9LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P9BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P9LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P9BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P9LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP9BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP9LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP10BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP10LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP16BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P9BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P9LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P9BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P9LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P9BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P9LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA420P16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P16LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P16LE)
  ADD_PIX_FMT(AV_PIX_FMT_VDPAU)
  ADD_PIX_FMT(AV_PIX_FMT_XYZ12LE)
  ADD_PIX_FMT(AV_PIX_FMT_XYZ12BE)
  ADD_PIX_FMT(AV_PIX_FMT_NV16)
  ADD_PIX_FMT(AV_PIX_FMT_NV20LE)
  ADD_PIX_FMT(AV_PIX_FMT_NV20BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBA64BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBA64LE)
  ADD_PIX_FMT(AV_PIX_FMT_BGRA64BE)
  ADD_PIX_FMT(AV_PIX_FMT_BGRA64LE)
  ADD_PIX_FMT(AV_PIX_FMT_YVYU422)
  ADD_PIX_FMT(AV_PIX_FMT_YA16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YA16LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP16BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP16LE)
  ADD_PIX_FMT(AV_PIX_FMT_QSV)
  ADD_PIX_FMT(AV_PIX_FMT_MMAL)
  ADD_PIX_FMT(AV_PIX_FMT_D3D11VA_VLD)
  ADD_PIX_FMT(AV_PIX_FMT_CUDA)
  ADD_PIX_FMT(AV_PIX_FMT_0RGB)
  ADD_PIX_FMT(AV_PIX_FMT_RGB0)
  ADD_PIX_FMT(AV_PIX_FMT_0BGR)
  ADD_PIX_FMT(AV_PIX_FMT_BGR0)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P12BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P12LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P14BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV420P14LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P12BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P12LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P14BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV422P14LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P12BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P12LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P14BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV444P14LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP12BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP12LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP14BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRP14LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVJ411P)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_BGGR8)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_RGGB8)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_GBRG8)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_GRBG8)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_BGGR16LE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_BGGR16BE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_RGGB16LE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_RGGB16BE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_GBRG16LE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_GBRG16BE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_GRBG16LE)
  ADD_PIX_FMT(AV_PIX_FMT_BAYER_GRBG16BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV440P10LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV440P10BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV440P12LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUV440P12BE)
  ADD_PIX_FMT(AV_PIX_FMT_AYUV64LE)
  ADD_PIX_FMT(AV_PIX_FMT_AYUV64BE)
  ADD_PIX_FMT(AV_PIX_FMT_VIDEOTOOLBOX)
  ADD_PIX_FMT(AV_PIX_FMT_P010LE)
  ADD_PIX_FMT(AV_PIX_FMT_P010BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP12BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP12LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP10BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP10LE)
  ADD_PIX_FMT(AV_PIX_FMT_MEDIACODEC)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY12BE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY12LE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY10BE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY10LE)
  ADD_PIX_FMT(AV_PIX_FMT_P016LE)
  ADD_PIX_FMT(AV_PIX_FMT_P016BE)
  ADD_PIX_FMT(AV_PIX_FMT_D3D11)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY9BE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY9LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRPF32BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRPF32LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAPF32BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAPF32LE)
  ADD_PIX_FMT(AV_PIX_FMT_DRM_PRIME)
  ADD_PIX_FMT(AV_PIX_FMT_OPENCL)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY14BE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAY14LE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAYF32BE)
  ADD_PIX_FMT(AV_PIX_FMT_GRAYF32LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P12BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA422P12LE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P12BE)
  ADD_PIX_FMT(AV_PIX_FMT_YUVA444P12LE)
  ADD_PIX_FMT(AV_PIX_FMT_NV24)
  ADD_PIX_FMT(AV_PIX_FMT_NV42)
  ADD_PIX_FMT(AV_PIX_FMT_VULKAN)
  ADD_PIX_FMT(AV_PIX_FMT_Y210BE)
  ADD_PIX_FMT(AV_PIX_FMT_Y210LE)
  ADD_PIX_FMT(AV_PIX_FMT_X2RGB10LE)
  ADD_PIX_FMT(AV_PIX_FMT_X2RGB10BE)
  ADD_PIX_FMT(AV_PIX_FMT_X2BGR10LE)
  ADD_PIX_FMT(AV_PIX_FMT_X2BGR10BE)
  ADD_PIX_FMT(AV_PIX_FMT_P210BE)
  ADD_PIX_FMT(AV_PIX_FMT_P210LE)
  ADD_PIX_FMT(AV_PIX_FMT_P410BE)
  ADD_PIX_FMT(AV_PIX_FMT_P410LE)
  ADD_PIX_FMT(AV_PIX_FMT_P216BE)
  ADD_PIX_FMT(AV_PIX_FMT_P216LE)
  ADD_PIX_FMT(AV_PIX_FMT_P416BE)
  ADD_PIX_FMT(AV_PIX_FMT_P416LE)
  ADD_PIX_FMT(AV_PIX_FMT_VUYA)
  ADD_PIX_FMT(AV_PIX_FMT_RGBAF16BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBAF16LE)
  ADD_PIX_FMT(AV_PIX_FMT_VUYX)
  ADD_PIX_FMT(AV_PIX_FMT_P012LE)
  ADD_PIX_FMT(AV_PIX_FMT_P012BE)
  ADD_PIX_FMT(AV_PIX_FMT_Y212BE)
  ADD_PIX_FMT(AV_PIX_FMT_Y212LE)
  ADD_PIX_FMT(AV_PIX_FMT_XV30BE)
  ADD_PIX_FMT(AV_PIX_FMT_XV30LE)
  ADD_PIX_FMT(AV_PIX_FMT_XV36BE)
  ADD_PIX_FMT(AV_PIX_FMT_XV36LE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBF32BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBF32LE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBAF32BE)
  ADD_PIX_FMT(AV_PIX_FMT_RGBAF32LE)
  ADD_PIX_FMT(AV_PIX_FMT_P212BE)
  ADD_PIX_FMT(AV_PIX_FMT_P212LE)
  ADD_PIX_FMT(AV_PIX_FMT_P412BE)
  ADD_PIX_FMT(AV_PIX_FMT_P412LE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP14BE)
  ADD_PIX_FMT(AV_PIX_FMT_GBRAP14LE)
  ;
    //.export_values();

#define ADD_ENUM(_class,v) .value(#v, _class::v)

  py::enum_<AVPictureType>(m, "AVPictureType")
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_NONE)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_I)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_P)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_B)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_S)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_SI)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_SP)
    ADD_ENUM(AVPictureType, AV_PICTURE_TYPE_BI)
    ;


  py::enum_<AVHWDeviceType>(m, "AVHWDeviceType")
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_NONE)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_VDPAU)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_CUDA)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_VAAPI)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_DXVA2)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_QSV)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_VIDEOTOOLBOX)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_D3D11VA)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_DRM)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_OPENCL)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_MEDIACODEC)
    ADD_ENUM( AVHWDeviceType, AV_HWDEVICE_TYPE_VULKAN)
    ;


#define RO(_class, member) .def_readonly(#member, &_class::member)
#define RW(_class, member) .def_readwrite(#member, &_class::member)

  py::class_<AVFrame>(m, "AVFrame")
    //  .def(py::init<>()) // constructor
    RO(AVFrame, width)
    RO(AVFrame, height)
    RO(AVFrame, format)
    RO(AVFrame, pict_type)
    RO(AVFrame, pts)
    RO(AVFrame, pkt_dts)
    RO(AVFrame, best_effort_timestamp)
    RO(AVFrame, flags)
    ;

  py::class_<AVRational>(m, "AVRational")
    RW(AVRational, num)
    RW(AVRational, den)
    .def("__float__", [](const AVRational& self) { return float(self.num) / float(self.den);  })
    .def("__str__", [](const AVRational& self) { 
      return std::to_string(self.num) + "/" + std::to_string(self.den);  
    })
    ;

  py::enum_<AVMediaType>(m, "AVMediaType")
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_UNKNOWN)
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_VIDEO)
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_AUDIO)
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_DATA)
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_SUBTITLE)
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_ATTACHMENT)
    ADD_ENUM(AVMediaType, AVMEDIA_TYPE_NB)
    ;

  py::class_<AVCodecParameters>(m, "AVCodecParameters")
    RO(AVCodecParameters, codec_type) // AVMediaType
    RO(AVCodecParameters, codec_id) // AVCodecID, many values in the enum
    RO(AVCodecParameters, codec_tag)
    RO(AVCodecParameters, format)
    RO(AVCodecParameters, bit_rate)
    RO(AVCodecParameters, bits_per_coded_sample)
    RO(AVCodecParameters, bits_per_raw_sample)
    RO(AVCodecParameters, profile)
    RO(AVCodecParameters, level)
    RO(AVCodecParameters, width)
    RO(AVCodecParameters, height)
    RO(AVCodecParameters, sample_aspect_ratio)
    RO(AVCodecParameters, field_order)
    RO(AVCodecParameters, video_delay)
    RO(AVCodecParameters, sample_rate)
    RO(AVCodecParameters, block_align)
    RO(AVCodecParameters, frame_size)
    RO(AVCodecParameters, framerate)
    ;

  py::class_<AVStream>(m, "AVStream")
    //  .def(py::init<>()) // constructor
    RO(AVStream, index)
    RO(AVStream, id)
    RO(AVStream, codecpar)
    RO(AVStream, time_base)
    RO(AVStream, start_time)
    RO(AVStream, duration)
    RO(AVStream, nb_frames)
    RO(AVStream, disposition)
    // AVDiscard discard
    RO(AVStream, sample_aspect_ratio)
    // AVDictionary * metadata
    RO(AVStream, avg_frame_rate)
    //RO(AVStream, event_flags)
    RO(AVStream, r_frame_rate)
    //RO(AVStream, pts_wrap_bits)
    ;

    py::class_<AVCodecContext>(m, "AVCodecContext")
    RO(AVCodecContext, bit_rate)
    RO(AVCodecContext, gop_size)
    ;

  py::class_<AVFormatContext>(m, "AVFormatContext")
    RO(AVFormatContext, nb_streams)
    ;

#define ARG(v) py::arg(#v)

  py::class_<AV::Frame>(m, "Frame")
    .def                  (py::init<>()) // constructor
    .def                  ("get",         &AV::Frame::get, py::return_value_policy::reference)
    .def                  ("getData",     &AV::Frame::getData, "get frame data", ARG(channel), ARG(height), ARG(width)
                                         , ARG(verbose) = false)
    .def                  ("getFormat",             &AV::Frame::getFormat)
    .def                  ("getShape",              &AV::Frame::getShape)
    .def                  ("getLinesizeAll",        &AV::Frame::getLinesizeAll)
    .def                  ("getLinesize",           &AV::Frame::getLinesize)
    .def_property         ("flags",                 &AV::Frame::getFlags, &AV::Frame::setFlags)
    .def_property_readonly("pts",                   &AV::Frame::pts)
    .def_property_readonly("best_effort_timestamp", &AV::Frame::best_effort_timestamp)
    .def_property_readonly("key_frame",             &AV::Frame::key_frame)
    .def_property_readonly("interlaced_frame",      &AV::Frame::interlaced_frame)
    ;

  py::class_<AV::CodecContext>(m, "CodecContext")
    .def(py::init<>()) // constructor
    .def("get", &AV::CodecContext::get, py::return_value_policy::reference)
    ;

  py::class_<AV::VideoDecoder>(m, "VideoDecoder")
    .def(py::init<const int&>(), ARG(nb_frames)=int(8)) // constructor
    .def("open",             &AV::VideoDecoder::open, "Open decoder", 
            ARG(filename), 
            ARG(devide_type_name) = nullptr, 
            ARG(video_stream_index) = -1,
            ARG(num_threads) = 4,
            ARG(thread_type) = "FRAME")
    .def("nextFrame",        &AV::VideoDecoder::nextFrame, "Decode next video frame", ARG(convert) = true)
    .def("getFrame",         &AV::VideoDecoder::getFrame, py::return_value_policy::reference)
    .def("seek",             &AV::VideoDecoder::seek)
    .def("seek_file",        &AV::VideoDecoder::seek_file)
    .def("getStream",        &AV::VideoDecoder::getStream, "Get stream", ARG(idx) = -1, py::return_value_policy::reference)
    .def("getFormatContext", &AV::VideoDecoder::getFormatContext,                       py::return_value_policy::reference)
    .def("get_nb_frames",    &AV::VideoDecoder::get_nb_frames)
    .def("useHw",            &AV::VideoDecoder::useHw)
    .def("get_codec_ctx",     &AV::VideoDecoder::get_codec_ctx, py::return_value_policy::reference)
    ;

  m.def("load_frames", &AV::load_frames, "Loads n first video frames");

  m.def("averror2str", &AV::averror2str, "Convert error code to string");

  auto hw = m.def_submodule("HW");
  hw.def("get_device_types", &AV::HW::get_device_types, "Get hardware device types");

  hw.def("get_device_type_names", &AV::HW::get_device_type_names, "Get hardware device type names");

  // define all classes
  py::class_<AV::Packet>(m, "Packet")
        .def(py::init<>()) // constructor
        .def("unRef", &AV::Packet::unRef)
        ;
        
  
  // C++ defines
  m.attr("AV_NOPTS_VALUE")                = py::int_(AV_NOPTS_VALUE);
  m.attr("AV_FRAME_FLAG_CORRUPT")         = py::int_(AV_FRAME_FLAG_CORRUPT);
  m.attr("AV_FRAME_FLAG_KEY ")            = py::int_(AV_FRAME_FLAG_KEY);
  m.attr("AV_FRAME_FLAG_DISCARD")         = py::int_(AV_FRAME_FLAG_DISCARD);
  m.attr("AV_FRAME_FLAG_INTERLACED ")     = py::int_(AV_FRAME_FLAG_INTERLACED );
  m.attr("AV_FRAME_FLAG_TOP_FIELD_FIRST") = py::int_(AV_FRAME_FLAG_TOP_FIELD_FIRST);

  //   py::class_<AV::FormatContext>(m, "FormatContext")
//         .def(py::init<>) // constructor
//         .def("openFile", &AV::FormatContext::openFile)

  // define all standalone functions
//  m.def("StandAloneFunction", &StandAloneFunction);


}