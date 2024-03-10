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


#define NB_FRAMES 200
using namespace std::chrono;


namespace AV
{

  //-----------------------------------------------------------------------------------------------
  class AVException : public std::exception {
  public:
    AVException(const char* msg)
    {
      _message = msg;
    }
    AVException(const std::string& msg)
    {
      _message = msg;
    }
    const char * what() const{
      return _message.c_str();
    }
  private:
    std::string _message;
  };

  //-----------------------------------------------------------------------------------------------
  class Packet
  {
  public:
    Packet()
    {
      _packet = av_packet_alloc();
      if (!_packet) {
        _packet = nullptr;
        throw AVException("Failed to allocated packet");
      }
    }

    ~Packet()
    {
      if (_packet != nullptr) {
        av_packet_free(&_packet);
        _packet = nullptr;
      }
    }

    void unRef()
    {
      if (_packet != nullptr)
        av_packet_unref(_packet);
    }

    AVPacket* get() { return _packet; }

  private:
    AVPacket* _packet;
  };

  //-----------------------------------------------------------------------------------------------
  class FormatContext
  {
  public:
    FormatContext(const char* filename=nullptr) {
      _format_ctx = avformat_alloc_context();
      if (!_format_ctx) {
        _format_ctx = nullptr;
        throw AVException("Failed to create AVFormatContext");
      }
      _file_opened = false;
      if (filename != nullptr)
        openFile(filename);
    }

    ~FormatContext() {
      if (_file_opened) {
        avformat_close_input(&_format_ctx);
        _file_opened = false;
      }
      if (_format_ctx != nullptr) {
        avformat_free_context(_format_ctx);
        _format_ctx = nullptr;
      }
    }
    AVFormatContext* get() {
      return _format_ctx;
    }
    AVFormatContext** getPtr() {
      return &_format_ctx;
    }

    void openFile(const char* filename)
    {
      if (avformat_open_input(&_format_ctx, filename, NULL, NULL) != 0) {
        _file_opened = false;
        throw AVException("Failed to open video file");
      }
      else
        _file_opened = true;
    }

    bool fileOpened() const { return _file_opened; }

    int findFirstValidVideoStream()
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

    AVCodecParameters* getCodecParams(const int& stream_index)
    {
      return _format_ctx->streams[stream_index]->codecpar;
    }

    const AVCodec* findDecoder(const int& stream_index)
    {
      return avcodec_find_decoder(getCodecParams(stream_index)->codec_id);
    }

    void findStreamInfo()
    {
      if (avformat_find_stream_info(_format_ctx, NULL) < 0)
        throw AVException("Failed to find input stream information");
    }

    int findBestVideoStream(const AVCodec** codec) {
      int ret = av_find_best_stream(_format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, codec, 0);
      if (ret < 0)
        throw AVException("Cannot find a video stream in the input file");
      return ret;
    }

  private:
    AVFormatContext* _format_ctx;
    bool             _file_opened;

  };

  //-----------------------------------------------------------------------------------------------
  class CodecContext
  {
  public:
    CodecContext(const AVCodec* codec)
    {
      _codec_ctx = avcodec_alloc_context3(codec);
      if (_codec_ctx == nullptr)
        throw AVException("Failed to create AVCodecContext");
    }
    ~CodecContext()
    {
      if (_codec_ctx != nullptr) {
        avcodec_free_context(&_codec_ctx);
        _codec_ctx = nullptr;
      }
    }

    int initFromParam(const AVCodecParameters* params) {
      int averror = avcodec_parameters_to_context(_codec_ctx, params);
      if (averror < 0) {
        char err[256];
        av_strerror(averror, err, 256);
        std::string error_msg = "AV::CodeContext: "+ std::string(err) +"\n" ;
        throw AVException(error_msg);
      }
      return averror;
    }

    void setThreading( int count=8, int type = FF_THREAD_FRAME)
    {
      _codec_ctx->thread_count = count;
      _codec_ctx->thread_type  = type;
    }

    int initHw(const enum AVHWDeviceType type)
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
      return err;
    }

    void open(const AVCodec* codec, AVDictionary** options=nullptr)
    {
      if (avcodec_open2(_codec_ctx, codec, options) < 0) {
        throw AVException("Failed to open codec");
      }
    }

    AVCodecContext* get() {
      return _codec_ctx;
    }

  private:
    AVCodecContext* _codec_ctx;
  };

} // end namespace AV

#define CHECK_RETURN(condition, message, returnvalue) if(!(condition)) { fprintf(stderr, message); return returnvalue; }


//-------------------------------------------------------------------------------------------------
// decode SW
//-------------------------------------------------------------------------------------------------

bool load_frame(const char* filename, int* width_out, int* height_out, unsigned char** data_out) {

  try
  {
    // Open the file using libavformat
    AV::FormatContext format_ctx(filename);

    // Find the first valid video stream
    int video_stream_index = format_ctx.findFirstValidVideoStream();

    auto codec_params = format_ctx.getCodecParams(video_stream_index);
    auto codec = format_ctx.findDecoder(video_stream_index);

    // Set up the codec contex for the decoder
    AV::CodecContext codec_ctx(codec);
    codec_ctx.initFromParam(codec_params);
    codec_ctx.setThreading(8, FF_THREAD_FRAME);
    codec_ctx.open(codec);

    AVFrame* av_frame = av_frame_alloc();
    if (!av_frame) {
      printf("Couldn't allocate AVFrame\n");
      return false;
    }
    AV::Packet packet;

    int response;
    int framenum = 0;
    bool frame_timer = false;
    auto prev = high_resolution_clock::now();
    auto curr = high_resolution_clock::now();
    auto start = curr;

    while ((av_read_frame(format_ctx.get(), packet.get()) >= 0) && (framenum < NB_FRAMES)) {
      if (packet.get()->stream_index != video_stream_index) {
        continue;
      }
      response = avcodec_send_packet(codec_ctx.get(), packet.get());
      if (response < 0) {
        char error_message[255];
        av_strerror(response, error_message, 255);
        printf("Failed to decode packet: %s \n", error_message);
        return false;
      }
      response = avcodec_receive_frame(codec_ctx.get(), av_frame);
      if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
        continue;
      }
      else if (response < 0) {
        char error_message[255];
        av_strerror(response, error_message, 255);
        printf("Failed to decode packet: %s", error_message);
        return false;
      }

      if (frame_timer) {
        curr = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(curr - prev);
        printf("got frame %d \n", framenum);
        std::cout << "took " << duration.count() / 1000.0f << " ms" << std::endl;
        prev = curr;
      }

      framenum++;
      packet.unRef();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << " decode first " << NB_FRAMES << " frames took " << duration.count() / 1000.0f << " ms" << std::endl;

    unsigned char* data = new unsigned char[av_frame->width*av_frame->height * 3];
    for (int x = 0; x < av_frame->width; ++x) {
      for (int y = 0; y < av_frame->height; ++y) {
        data[y*av_frame->width * 3 + x * 3] = (unsigned char)0xff;
        data[y*av_frame->width * 3 + x * 3 + 1] = (unsigned char)0x00;
        data[y*av_frame->width * 3 + x * 3 + 2] = (unsigned char)0x00;
      }
    }
    *width_out = av_frame->width;
    *height_out = av_frame->height;
    *data_out = data;

    av_frame_free(&av_frame);

    return true;
  }
  catch (AV::AVException except) {
    std::cerr << "Exceptions:" << except.what() << std::endl;
    return false;
  }


}

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
      fprintf(stderr, "Failed to get HW surface format.\n");
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

static int decode_write(AVCodecContext *avctx, AVPacket *packet, const enum AVPixelFormat& hw_pix_fmt)
{
  AVFrame *frame = NULL, *sw_frame = NULL;
  AVFrame *tmp_frame = NULL;
  uint8_t *buffer = NULL;
  int size;
  int ret = 0;

  ret = avcodec_send_packet(avctx, packet);
  if (ret < 0) {
    fprintf(stderr, "Error during decoding\n");
    return ret;
  }

  while (1) {
    if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
      fprintf(stderr, "Can not alloc frame\n");
      ret = AVERROR(ENOMEM);
      goto fail;
    }

    ret = avcodec_receive_frame(avctx, frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
      av_frame_free(&frame);
      av_frame_free(&sw_frame);
      return 0;
    }
    else if (ret < 0) {
      fprintf(stderr, "Error while decoding\n");
      goto fail;
    }

    if (frame->format == hw_pix_fmt) {
      /* retrieve data from GPU to CPU */
      //auto prev = high_resolution_clock::now();
      if ((ret = av_hwframe_transfer_data(sw_frame, frame, 0)) < 0) {
        fprintf(stderr, "Error transferring the data to system memory\n");
        goto fail;
      }
      //auto curr = high_resolution_clock::now();
      //auto duration = duration_cast<microseconds>(curr - prev);
      //std::cout << " GPU --> CPU took " << duration.count() / 1000.0f << " ms" << std::endl;
      tmp_frame = sw_frame;
    }
    else
      tmp_frame = frame;

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

  fail:
    //std::cout << "fail" << std::endl;
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_freep(&buffer);
    if (ret < 0)
      return ret;
  }
}

bool load_frame_hw(const char* device_type_name, const char* filename)
{
  try {

    const AVCodec *codec = NULL;
    AV::Packet packet;
    AV::FormatContext format_ctx(filename);
    format_ctx.findStreamInfo();

    /* find the video stream information */
    int video_stream_index = format_ctx.findBestVideoStream(&codec);

    static enum AVPixelFormat hw_pix_fmt;

    auto hw_device_type = AV::HW::get_device_type(device_type_name);
    if (hw_device_type == AV_HWDEVICE_TYPE_NONE)  return false;

    auto* hwconfig = AV::HW::get_codec_hwconfig(codec, hw_device_type);
    if (hwconfig == nullptr) return false;
    hw_pix_fmt = hwconfig->pix_fmt;

    AV::CodecContext codec_ctx(codec);

    AVStream *video = format_ctx.get()->streams[video_stream_index];
    codec_ctx.initFromParam(video->codecpar);

    // Use lambda to create a callback that captures hw_pix_fmt
    codec_ctx.get()->get_format = [](AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {return AV::HW::get_hw_format(ctx, pix_fmts, hw_pix_fmt); };
    codec_ctx.initHw(hw_device_type);
    codec_ctx.setThreading(4, FF_THREAD_FRAME); // 4, best value ?
    codec_ctx.open(codec);

    /* open the file to dump raw data */
    //output_file = fopen(argv[3], "w+b");

    /* actual decoding and dump the raw data */
    int framenum = 0;
    bool frame_timer = false;
    auto prev = high_resolution_clock::now();
    auto curr = high_resolution_clock::now();
    auto start = curr;

    while ((av_read_frame(format_ctx.get(), packet.get()) >= 0) && (framenum < NB_FRAMES)) {
      if (packet.get()->stream_index != video_stream_index) 
        continue;

      decode_write(codec_ctx.get(), packet.get(), hw_pix_fmt);
      if (frame_timer) {
        curr = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(curr - prev);
        printf("got frame %d \n", framenum);
        std::cout << "took " << duration.count() / 1000.0f << " ms" << std::endl;
        prev = curr;
      }
      framenum++;

      packet.unRef();
    }
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    std::cout << " decode first " << NB_FRAMES << " frames took " << duration.count() / 1000.0f << " ms" << std::endl;



    /* flush the decoder */
    decode_write(codec_ctx.get(), NULL, hw_pix_fmt);

    //if (output_file)
    //  fclose(output_file);


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

//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  if ((argc <= 1) || (argc > 3)) {
    fprintf(stderr, "Usage: %s [hw_device_type] <input file>\n"
      , argv[0]);
    exit(0);
  }
  if (argc == 2) {
    std::string filename(argv[1]);
    int frame_width, frame_height;
    unsigned char* frame_data;
    if (!load_frame(filename.c_str(), &frame_width, &frame_height, &frame_data))
    {
      printf("Couldn't load video frame\n");
      return 1;
    }
  }
  else if (argc == 3)
  {
    std::string hwdevice(argv[1]);
    std::string filename(argv[2]);
    if (!load_frame_hw(hwdevice.c_str(), filename.c_str()))
    {
      printf("Couldn't load video frame with hw decoder\n");
      return 1;
    }
  }

}

