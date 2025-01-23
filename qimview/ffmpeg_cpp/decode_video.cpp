/*
 * Copyright (c) 2001 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
 
/**
 * @file libavcodec video decoding API usage example
 * @example decode_video.c *
 *
 * Read from an MPEG1 video file, decode frames, and generate PGM images as
 * output.
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <inttypes.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}
 
#include <chrono>
#include <iostream>
#include <exception>

#include "decode_video.hpp"

using namespace std::chrono;


namespace AV
{

    //-----------------------------------------------------------------------------------------------
    // AVException
    //-----------------------------------------------------------------------------------------------
    AVException::AVException(const char* msg)
    {
        _message = msg;
    }

    AVException::AVException(const std::string& msg)
    {
      _message = msg;
    }
    const char * AVException::what() const noexcept {
      return _message.c_str();
    }

    std::string averror2str(int averror)
    {
      std::string error_msg = "";
      if (averror < 0) {
        char err[256];
        av_strerror(averror, err, 256);
        error_msg = std::string(err);
      }
      return error_msg;
    }

    //-----------------------------------------------------------------------------------------------
    // Packet
    //-----------------------------------------------------------------------------------------------
    Packet::Packet()
    {
      _packet = av_packet_alloc();
      if (!_packet) {
        _packet = nullptr;
        throw AVException("Failed to allocated packet");
      }
    }

    Packet::~Packet()
    {
      if (_packet != nullptr) {
        av_packet_free(&_packet);
        _packet = nullptr;
      }
    }

    void Packet::unRef()
    {
      if (_packet != nullptr)
        av_packet_unref(_packet);
    }

    //-----------------------------------------------------------------------------------------------
    // FormatContext
    //-----------------------------------------------------------------------------------------------
    FormatContext::FormatContext(const char* filename) {
      _format_ctx = avformat_alloc_context();
      if (!_format_ctx) {
        _format_ctx = nullptr;
        throw AVException("Failed to create AVFormatContext");
      }
      _file_opened = false;
      if (filename != nullptr)
        openFile(filename);
    }

    FormatContext::~FormatContext() {
      if (_file_opened) {
        avformat_close_input(&_format_ctx);
        _file_opened = false;
      }
      if (_format_ctx != nullptr) {
        avformat_free_context(_format_ctx);
        _format_ctx = nullptr;
      }
    }

    void FormatContext::openFile(const char* filename)
    {
      if (avformat_open_input(&_format_ctx, filename, NULL, NULL) != 0) {
        _file_opened = false;
        throw AVException("Failed to open video file");
      }
      else
        _file_opened = true;
    }

    int FormatContext::findFirstValidVideoStream()
    {
      // Find the first valid video stream
      int stream_index = -1;
      AVCodecParameters* av_codec_params;
      const AVCodec*     av_codec;

      for (int i = 0; i < _format_ctx->nb_streams; ++i) {
        av_codec_params = getCodecParams(i);
        av_codec = findDecoder(i);
        if (!av_codec) continue;
        if (av_codec_params->codec_type == AVMEDIA_TYPE_VIDEO) {
          stream_index = i;
          break;
        }
      }
      if (stream_index == -1) throw AVException("Couldn't find a video stream ");
      return stream_index;
    }

    int  FormatContext::getStreamIndex(const int& video_stream_index)
    {

        int n=0;        
        for (int i = 0; i < _format_ctx->nb_streams; ++i) {
            auto av_codec_params = getCodecParams(i);
            auto av_codec = findDecoder(i);
            if (!av_codec) continue;
            if (av_codec_params->codec_type == AVMEDIA_TYPE_VIDEO) {
                if (n==video_stream_index)
                    return i;
                else
                    n++;
            }
        }
        return -1;
    }

    AVCodecParameters* FormatContext::getCodecParams(const int& stream_index)
    {
      return _format_ctx->streams[stream_index]->codecpar;
    }

    const AVCodec* FormatContext::findDecoder(const int& stream_index)
    {
      return avcodec_find_decoder(getCodecParams(stream_index)->codec_id);
    }

    void FormatContext::findStreamInfo()
    {
      if (avformat_find_stream_info(_format_ctx, NULL) < 0)
        throw AVException("Failed to find input stream information");
    }

    int FormatContext::findBestVideoStream(const AVCodec** codec) {
      int ret = av_find_best_stream(_format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, codec, 0);
      if (ret < 0)
        throw AVException("Cannot find a video stream in the input file");
      return ret;
    }

    int FormatContext::readVideoFrame(AVPacket* pkt, int video_stream_index)
    {
      int res;
      while ((res = av_read_frame(_format_ctx, pkt)) >= 0) {
        if (pkt->stream_index != video_stream_index) {
          // need to unRef pkt?
          continue;
        }
        else
          return res; // success
      }
      return res; // failure
    }

    //-----------------------------------------------------------------------------------------------
    // CodecContext
    //-----------------------------------------------------------------------------------------------
    CodecContext::CodecContext()
    {
      _codec_ctx = nullptr;
    }

    CodecContext::CodecContext(const AVCodec* codec)
    {
      _codec_ctx = avcodec_alloc_context3(codec);
      if (_codec_ctx == nullptr)
        throw AVException("Failed to create AVCodecContext");
    }

    void CodecContext::alloc(const AVCodec* codec)
    {
      if (_codec_ctx != nullptr)
        throw AVException("AVCodecContext already allocated");
      _codec_ctx = avcodec_alloc_context3(codec);
      if (_codec_ctx == nullptr)
        throw AVException("Failed to create AVCodecContext");
    }

    CodecContext::~CodecContext()
    {
      if (_codec_ctx != nullptr) {
        avcodec_free_context(&_codec_ctx);
        _codec_ctx = nullptr;
      }
    }

    int CodecContext::initFromParam(const AVCodecParameters* params) {
      int averror = avcodec_parameters_to_context(_codec_ctx, params);
      if (averror < 0) {
        char err[256];
        av_strerror(averror, err, 256);
        std::string error_msg = "AV::CodeContext: "+ std::string(err) +"\n" ;
        throw AVException(error_msg);
      }
      return averror;
    }

    void CodecContext::setThreading( int count, int type)
    {
      _codec_ctx->thread_count = count;
      _codec_ctx->thread_type  = type;
    }

    int CodecContext::initHw(const enum AVHWDeviceType type)
    {
      int err = 0;
      AVBufferRef* ctx;
      if ((err = av_hwdevice_ctx_create(&ctx, type, NULL, NULL, 0)) < 0) 
        throw AVException("Failed to create specified HW device.");
      _codec_ctx->hw_device_ctx = av_buffer_ref(ctx);

      if (_codec_ctx->hw_device_ctx == nullptr) {
        av_buffer_unref(&ctx);
        throw AVException("Failed to create av buffer ref.");
      }
      av_buffer_unref(&ctx);

      enum AVPixelFormat * 	formats;
      av_hwframe_transfer_get_formats(_codec_ctx->hw_device_ctx, AV_HWFRAME_TRANSFER_DIRECTION_TO,
        &formats, 0
      );
      std::cout << "transfer formats" << std::endl;
      int i = 0;
      while (formats[i] != AV_PIX_FMT_NONE)
      {
        std::cout << "format " << (int)formats[i] << std::endl;
        i++;
      }
      av_free(formats);

      return err;
    }

    void CodecContext::open(const AVCodec* codec, AVDictionary** options)
    {
      if (avcodec_open2(_codec_ctx, codec, options) < 0) {
        throw AVException("Failed to open codec");
      }
    }

    int CodecContext::receiveFrame(const AVPacket* pkt, AVFrame* frame)
    {
      int response = avcodec_send_packet(_codec_ctx, pkt);
      if (response < 0) {
        char error_message[255];
        av_strerror(response, error_message, 255);
        std::cerr << "Failed to send packet: " << error_message << std::endl;
        return response;
      }
      response = avcodec_receive_frame(_codec_ctx, frame);
      if ((response < 0)&&(response!=AVERROR(EAGAIN))&&(response != AVERROR_EOF)) {
        char error_message[255];
        av_strerror(response, error_message, 255);
        std::cerr << "Failed to receive frame: " << error_message << std::endl;
      }
      return response;
    }

    //-----------------------------------------------------------------------------------------------
    // CodecContext
    //-----------------------------------------------------------------------------------------------
    Frame::Frame()
    {
      _frame = av_frame_alloc();
      if (!_frame) {
        throw AVException("Failed to allocate AVFrame.");
      }
    }
    Frame::~Frame()
    {
      av_frame_free(&_frame);
    }

    /**
      if the current frame is of hw_pix_format, convert it to CPU frame in the cpu_frame argument variable
      return the converted or initial frame pointer
    */
    Frame* Frame::gpu2Cpu(const AVPixelFormat& hw_pix_fmt, Frame* cpu_frame)
    {
      bool timing = false;
      if (_frame->format == hw_pix_fmt) {
        //std::cout << "gpu2Cpu data transfer" << std::endl;
        // retrieve data from GPU to CPU 
        //auto prev = high_resolution_clock::now();
        auto ret = av_hwframe_transfer_data(cpu_frame->get(), _frame, 0);
        if (ret < 0)
          throw AV::AVException("Error transferring the data to system memory");
        //auto curr = high_resolution_clock::now();
        //auto duration = duration_cast<microseconds>(curr - prev);
        //std::cout << " GPU --> CPU took " << duration.count() / 1000.0f << " ms" << std::endl;
        return cpu_frame;
      }
      //std::cout << "gpu2Cpu returning initial frame" << std::endl;
      return this;
    }

    // Below: commented previous tested code, but not retained
    // TODO: check pixel format compatibility

    //return  py::array_t<uint16_t>(py::buffer_info(
    //  _frame->data[0],                      /* Pointer to Y data */
    //  scalar_size,
    //  py::format_descriptor<uint16_t>::format(),
    //  2,
    //  { height, width },                       /* Buffer dimensions */
    //  { linesize , scalar_size }
    //));

    //// Create a Python object that will free the allocated
    //// memory when destroyed:
    //py::capsule free_when_done(foo, [](void *f) {
    //  double *foo = reinterpret_cast<double *>(f);
    //  std::cerr << "Element [0] = " << foo[0] << "\n";
    //  std::cerr << "freeing memory @ " << f << "\n";
    //  delete[] foo;
    //});

    // Slow
    //return py::array_t<uint16_t>(
    //  { height, width }, // shape
    //  { linesize , scalar_size }, // strides 
    //  (uint16_t*) _frame->data[0] // the data pointer
    //  //free_when_done
    //  ); // numpy array references this parent

    // This one is not working
    //auto buf_info = py::buffer_info(
    //  _frame->data[0],                      /* Pointer to Y data */
    //  scalar_size,
    //   py::format_descriptor<uint16_t>::format(),
    //  2,
    //  { height, width },                       /* Buffer dimensions */
    //  { linesize , scalar_size }
    //);

    //// Speed: good ?
    //return  py::memoryview(buf_info);

    // Speed: good
    //return  py::memoryview::from_buffer(
    //  _frame->data[0],                      /* Pointer to Y data */
    //  { height, width },                       /* Buffer dimensions */
    //  { linesize , scalar_size }
    //);

    /// Temporary method to test the data transfer from a frame to python
    py::object Frame::getData(const int& channel, const int& height, const int& width, bool verbose)
    {
      AVPixelFormat format = (AVPixelFormat)_frame->format;

      auto linesize = _frame->linesize[channel];
      if (verbose) {
        std::cout << "linesize=" << linesize << std::endl;
        std::cout << "width=" << width << std::endl;
      }
      int scalar_size = int(linesize / width);
      if (verbose) {
        std::cout << "scalar_size=" << scalar_size << std::endl;
      }

      if ((scalar_size == 2)||(scalar_size==1))
      {
        return py::memoryview::from_memory(
          _frame->data[channel],
          linesize*height
        );
      }
      else {
        std::cout << "Only scalar size of 1 or 2 byte(s) are available, scalar_size="  << scalar_size << std::endl;
        return py::none();
      }
    }


} // end namespace AV


//-------------------------------------------------------------------------------------------------
// decode HW
//-------------------------------------------------------------------------------------------------

//
// Add AV::HW namespace
//
namespace AV {
  namespace HW {

    // TODO: Check if this callback is really needed
    AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts, const enum AVPixelFormat& pixel_format)
    {
      const enum AVPixelFormat *p;

      for (p = pix_fmts; *p != -1; p++) {
        if (*p == pixel_format)
          return *p;
      }
      std::cerr << "Failed to get HW surface format." << std::endl;
      return AV_PIX_FMT_NONE;
    }

    AVHWDeviceType get_device_type(const char* device_type_name)
    {
      auto hw_device_type = av_hwdevice_find_type_by_name(device_type_name);
      if (hw_device_type == AV_HWDEVICE_TYPE_NONE) {
        fprintf(stderr, "Device type %s is not supported.\n", device_type_name);
        fprintf(stderr, "Available device types:");
        while ((hw_device_type = av_hwdevice_iterate_types(hw_device_type)) != AV_HWDEVICE_TYPE_NONE)
          fprintf(stderr, " %s", av_hwdevice_get_type_name(hw_device_type));
        fprintf(stderr, "\n");
        return AV_HWDEVICE_TYPE_NONE;
      }
      return hw_device_type;
    }

    const AVCodecHWConfig* get_codec_hwconfig(const AVCodec *codec, const AVHWDeviceType & device_type)
    {
      for (int i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(codec, i);
        if (!config) {
          fprintf(stderr, "Decoder %s does not support device type %s.\n",
            codec->name, av_hwdevice_get_type_name(device_type));
          return nullptr;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == device_type) {
          return config;
        }
      }
    }
  }
}

// static int decode_write(AVCodecContext *avctx, AVPacket *packet, const enum AVPixelFormat& hw_pix_fmt)
// {
// ...
    //size = av_image_get_buffer_size((AVPixelFormat)tmp_frame->format, tmp_frame->width, tmp_frame->height, 1);
    //buffer = (uint8_t*) av_malloc(size);
    //if (!buffer) {
    //  fprintf(stderr, "Can not alloc buffer\n");
    //  ret = AVERROR(ENOMEM);
    //  goto fail;
    //}
    //ret = av_image_copy_to_buffer(buffer, size,
    //  (const uint8_t * const *)tmp_frame->data,
    //  (const int *)tmp_frame->linesize, (AVPixelFormat)tmp_frame->format,
    //  tmp_frame->width, tmp_frame->height, 1);
    //if (ret < 0) {
    //  fprintf(stderr, "Can not copy image to buffer\n");
    //  goto fail;
    //}

    // Don't write to file here
    //if ((ret = fwrite(buffer, 1, size, output_file)) < 0) {
    //  fprintf(stderr, "Failed to dump raw data.\n");
    //  goto fail;
    //}
// }


bool AV::VideoDecoder::open(
        const char* filename, 
        const char* device_type_name, 
        const int&  video_stream_index,
        const int&  num_threads
        )
{
  try {

    const AVCodec *codec = NULL;
    _filename = filename;
    _format_ctx.openFile(_filename.c_str());
    _format_ctx.findStreamInfo();

    /* find the video stream information */
    if (video_stream_index != -1) {
        _stream_index = _format_ctx.getStreamIndex(video_stream_index);
    }
    if (_stream_index != -1) {
        codec = _format_ctx.findDecoder(_stream_index);
    } else {
        _stream_index = _format_ctx.findBestVideoStream(&codec);
    }

    AVHWDeviceType hw_device_type = AV_HWDEVICE_TYPE_NONE;

    if (device_type_name != nullptr) {
      std::cout << "check HW" << std::endl;
      hw_device_type = AV::HW::get_device_type(device_type_name);
      if (hw_device_type == AV_HWDEVICE_TYPE_NONE)  return false;

      auto* hwconfig = AV::HW::get_codec_hwconfig(codec, hw_device_type);
      if (hwconfig == nullptr) return false;
      hw_pix_fmt = hwconfig->pix_fmt;
    }
    _use_hw = (hw_pix_fmt != AV_PIX_FMT_NONE) && (hw_device_type != AV_HWDEVICE_TYPE_NONE);
    std::cout << "_use_hw = " << _use_hw << " " << (hw_pix_fmt != AV_PIX_FMT_NONE) << " " << (hw_device_type != AV_HWDEVICE_TYPE_NONE) << std::endl;

    _codec_ctx.alloc(codec);

    AVStream *video = _format_ctx.get()->streams[_stream_index];
    _codec_ctx.initFromParam(video->codecpar);

    if (_use_hw) {
      // Use lambda to create a callback that captures hw_pix_fmt
      _codec_ctx.get()->get_format = [](AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {return AV::HW::get_hw_format(ctx, pix_fmts, hw_pix_fmt); };
      _codec_ctx.initHw(hw_device_type);
    }
    _codec_ctx.setThreading(num_threads, FF_THREAD_FRAME);
    _codec_ctx.open(codec);
    return true;
  }
  catch (AV::AVException except) {
    std::cerr << "Exceptions:" << except.what() << std::endl;
    return false;
  }

}

bool AV::VideoDecoder::seek(int64_t timestamp)
{
    int ret = av_seek_frame(_format_ctx.get(),_stream_index,timestamp,
                AVSEEK_FLAG_FRAME | AVSEEK_FLAG_BACKWARD);
    if (ret<0) return false;
    avcodec_flush_buffers(_codec_ctx.get());
    // int ret = avformat_seek_file(_format_ctx.get(),_stream_index,INT64_MIN,seekTarget,INT64_MAX,0);
    return true;
}

bool AV::VideoDecoder::seek_file(int64_t timestamp)
{
    int ret = avformat_seek_file(_format_ctx.get(),_stream_index,
                INT64_MIN,timestamp,INT64_MAX,
                AVSEEK_FLAG_FRAME | AVSEEK_FLAG_BACKWARD);
    if (ret<0) return false;
    avcodec_flush_buffers(_codec_ctx.get());
    return true;
}


int AV::VideoDecoder::nextFrame(bool convert)
{
  bool frame_timer = false;
  //auto prev = high_resolution_clock::now();
  //auto curr = high_resolution_clock::now();
  //auto start = curr;
  int res;


  auto initial_idx = _current_frame_idx;
  AV::Frame* frame;
  if (!_use_hw) 
    frame = _frames[initial_idx];
  else
    // in case of GPU, use a single frame for decoding and save the converted frame in the array
    frame = &_gpu_frame;
  _current_frame_idx = (_current_frame_idx + 1)%_nb_frames;

  while (true) 
  {
    res =_format_ctx.readVideoFrame(_packet.get(), _stream_index);
    if (res == 0) {
      // std::cout << "nextFrame: packet pts=" << _packet.get()->pts 
      //           << std::endl;
      int response = _codec_ctx.receiveFrame(_packet.get(), frame->get());
      if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) continue;
      else if (response < 0) return response;

      if (convert && _use_hw) {
        // use previous circular frame as frame for CPU 2 GPU convertion?
        // _nb_frames must be > 1
        AV::Frame* cpu_frame = _frames[initial_idx];
        _current_frame = frame->gpu2Cpu(hw_pix_fmt, cpu_frame);
      }
      else
        _current_frame = frame;
      if (_current_frame->pts() != frame->get()->pts)
      {
        // std::cout << " current frame pts != frame pts " << _current_frame->pts() << ", " << _frame.pts() <<  std::endl;
        // Copy timestamp from packet
        _current_frame->get()->pts = frame->get()->pts;
      }

      if (frame_timer) {
        //curr = high_resolution_clock::now();
        //auto duration = duration_cast<microseconds>(curr - prev);
        //std::cout << " got frame " << _framenum << std::endl;
        //std::cout << "took " << duration.count() / 1000.0f << " ms" << std::endl;
        //prev = curr;
      }

      _framenum++;
      _packet.unRef();
      return response;
    }
    else
      return res;
  }
  return res;

}

AV::Frame* AV::VideoDecoder::getFrame() const
{
//   std::cout << "AV::VideoDecoder::getFrame() " << _current_frame << "  " << _current_frame->get() << std::endl;
  return _current_frame;
}

AVPixelFormat AV::VideoDecoder::hw_pix_fmt = AV_PIX_FMT_NONE;

bool AV::load_frames(const char* filename, const char* device_type_name, int nb_frames)
{
  try {
    AV::VideoDecoder decoder;
    decoder.open(filename, device_type_name);

    auto start = high_resolution_clock::now();
    int res = 0;
    do  {
      res = decoder.nextFrame();
      //std::cout << "ok " << ok << std::endl;
    } while ((res==0) && (decoder.frameNumber() < nb_frames));

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << " decode first " << nb_frames << " frames took " << duration.count() / 1000.0f << " ms" << std::endl;
    return true;
  }
  catch (AV::AVException except) {
    std::cerr << "Exceptions:" << except.what() << std::endl;
    return false;
  }
}


#define INBUF_SIZE 4096
 
static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename)
{
    FILE *f;
    int i;
 
    f = fopen(filename,"wb");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}
 
static void decode(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *pkt,
                   const char *filename)
{
    char buf[1024];
    int ret;
 
    ret = avcodec_send_packet(dec_ctx, pkt);
    if (ret < 0) {
        fprintf(stderr, "Error sending a packet for decoding\n");
        exit(1);
    }
 
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec_ctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during decoding\n");
            exit(1);
        }
 
        //printf("saving frame %3"PRId64"\n", dec_ctx->frame_num);
        fflush(stdout);
 
        /* the picture is allocated by the decoder. no need to
           free it */
        //snprintf(buf, sizeof(buf), "%s-%"PRId64, filename, dec_ctx->frame_num);
        pgm_save(frame->data[0], frame->linesize[0],
                 frame->width, frame->height, buf);
    }
}

