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
#define NB_FRAMES 200
using namespace std::chrono;

namespace AV
{
  //-----------------------------------------------------------------------------------------------
  class Packet
  {
  public:
    Packet()
    {
      _packet = av_packet_alloc();
      if (!_packet) {
        printf("Couldn't allocate AVPacket\n");
        _packet = nullptr;
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
        printf("Couldn't create AVFormatContext\n");
        _format_ctx = nullptr;
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
        printf("Couldn't open video file\n");
        _file_opened = false;
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
      _error_msg = "";
      _codec_ctx = avcodec_alloc_context3(codec);
      if (_codec_ctx == nullptr)
        _error_msg = "Couldn't create AVCodecContext\n";
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
        _error_msg = "AV::CodeContext: "+ std::string(err) +"\n" ;
      }
      return averror;
    }

    AVCodecContext* get() {
      return _codec_ctx;
    }

    std::string errorMsg() { return _error_msg;  }

  private:
    AVCodecContext* _codec_ctx;
    std::string _error_msg;
  };

} // end namespace AV

#define CHECK_RETURN(condition, message, returnvalue) if(!(condition)) { fprintf(stderr, message); return returnvalue; }


//-------------------------------------------------------------------------------------------------
// decode SW
//-------------------------------------------------------------------------------------------------

bool load_frame(const char* filename, int* width_out, int* height_out, unsigned char** data_out) {

  // Open the file using libavformat
  AV::FormatContext av_format_ctx(filename);
  CHECK_RETURN(av_format_ctx.fileOpened(), "Couldn't open video file\n", false);

  // Find the first valid video stream
  int video_stream_index = av_format_ctx.findFirstValidVideoStream();
  CHECK_RETURN(video_stream_index != -1, "Couldn't find a video stream \n", false);

  auto av_codec_params = av_format_ctx.getCodecParams(video_stream_index);
  auto av_codec = av_format_ctx.findDecoder(video_stream_index);

  // Set up the codec contex for the decoder
  AV::CodecContext codec_ctx(av_codec);
  CHECK_RETURN(codec_ctx.get(), codec_ctx.errorMsg().c_str(), false);

  int res = codec_ctx.initFromParam(av_codec_params);
  CHECK_RETURN(res >= 0, codec_ctx.errorMsg().c_str(), false);

  codec_ctx.get()->thread_count = 8;
  codec_ctx.get()->thread_type = FF_THREAD_FRAME;
  //codec_ctx->pix_fmt = mpeg_get_pixelformat(codec_ctx);
  //codec_ctx->hwaccel = ff_find_hwaccel(codec_ctx->codec->id, codec_ctx->pix_fmt);

  if (avcodec_open2(codec_ctx.get(), av_codec, NULL) < 0) {
    printf("Couldn't open codec\n");
    return false;
  }

  AVFrame* av_frame = av_frame_alloc();
  if (!av_frame) {
    printf("Couldn't allocate AVFrame\n");
    return false;
  }
  AV::Packet av_packet;

  int response;
  int framenum = 0;
  bool frame_timer = false;
  auto prev  = high_resolution_clock::now();
  auto curr  = high_resolution_clock::now();
  auto start = curr;

  while ((av_read_frame(av_format_ctx.get(), av_packet.get()) >= 0)&&(framenum<NB_FRAMES)) {
    if (av_packet.get()->stream_index != video_stream_index) {
      continue;
    }
    response = avcodec_send_packet(codec_ctx.get(), av_packet.get());
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
    av_packet.unRef();
  }
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  std::cout << " decode first " << NB_FRAMES << " frames took " << duration.count() / 1000.0f << " ms" << std::endl;

  unsigned char* data = new unsigned char[av_frame->width*av_frame->height * 3];
  for (int x = 0; x < av_frame->width; ++x) {
    for (int y = 0; y < av_frame->height; ++y) {
      data[y*av_frame->width * 3 + x * 3     ] = (unsigned char) 0xff;
      data[y*av_frame->width * 3 + x * 3 + 1 ] = (unsigned char) 0x00;
      data[y*av_frame->width * 3 + x * 3 + 2 ] = (unsigned char) 0x00;
    }
  }
  *width_out  = av_frame->width;
  *height_out = av_frame->height;
  *data_out   = data;

  av_frame_free(&av_frame);

  return true;
}

//-------------------------------------------------------------------------------------------------
// decode HW
//-------------------------------------------------------------------------------------------------

static AVBufferRef *hw_device_ctx = NULL;
static enum AVPixelFormat hw_pix_fmt;


static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
  const enum AVPixelFormat *pix_fmts)
{
  const enum AVPixelFormat *p;

  for (p = pix_fmts; *p != -1; p++) {
    if (*p == hw_pix_fmt)
      return *p;
  }

  fprintf(stderr, "Failed to get HW surface format.\n");
  return AV_PIX_FMT_NONE;
}

static int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type)
{
  int err = 0;

  if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
    NULL, NULL, 0)) < 0) {
    fprintf(stderr, "Failed to create specified HW device.\n");
    return err;
  }
  ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

  return err;
}

static int decode_write(AVCodecContext *avctx, AVPacket *packet)
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
    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_freep(&buffer);
    if (ret < 0)
      return ret;
  }
}

bool load_frame_hw(const char* device_type, const char* filename)
{
  int video_stream, ret;
  AVStream *video = NULL;
  const AVCodec *decoder = NULL;
  enum AVHWDeviceType type;
  int i;

    //fprintf(stderr, "Usage: %s <device type> <input file> <output file>\n", argv[0]);

  type = av_hwdevice_find_type_by_name(device_type);
  if (type == AV_HWDEVICE_TYPE_NONE) {
    fprintf(stderr, "Device type %s is not supported.\n",device_type);
    fprintf(stderr, "Available device types:");
    while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
      fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
    fprintf(stderr, "\n");
    return false;
  }

  AV::Packet packet;
  CHECK_RETURN(packet.get(), "Failed to allocate AVPacket\n", false);

  /* open the input file */
  AV::FormatContext input_ctx(filename);
  CHECK_RETURN(input_ctx.fileOpened(), "Cannot open input file", false);

  if (avformat_find_stream_info(input_ctx.get(), NULL) < 0) {
    fprintf(stderr, "Cannot find input stream information.\n");
    return false;
  }

  /* find the video stream information */
  ret = av_find_best_stream(input_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
  if (ret < 0) {
    fprintf(stderr, "Cannot find a video stream in the input file\n");
    return false;
  }
  video_stream = ret;

  for (i = 0;; i++) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
    if (!config) {
      fprintf(stderr, "Decoder %s does not support device type %s.\n",
        decoder->name, av_hwdevice_get_type_name(type));
      return false;
    }
    if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
      config->device_type == type) {
      hw_pix_fmt = config->pix_fmt;
      break;
    }
  }

  AV::CodecContext codec_ctx(decoder);
  CHECK_RETURN(codec_ctx.get(), codec_ctx.errorMsg().c_str(), false);

  video = input_ctx.get()->streams[video_stream];
  int res = codec_ctx.initFromParam(video->codecpar);
  CHECK_RETURN(res >= 0, codec_ctx.errorMsg().c_str(), false);

  codec_ctx.get()->get_format = get_hw_format;

  if (hw_decoder_init(codec_ctx.get(), type) < 0)
    return false;

  codec_ctx.get()->thread_count = 4; // Best value???
  codec_ctx.get()->thread_type = FF_THREAD_FRAME;
  if ((ret = avcodec_open2(codec_ctx.get(), decoder, NULL)) < 0) {
    fprintf(stderr, "Failed to open codec for stream #%u\n", video_stream);
    return false;
  }

  /* open the file to dump raw data */
  //output_file = fopen(argv[3], "w+b");

  /* actual decoding and dump the raw data */
  int framenum = 0;
  bool frame_timer = false;
  auto prev = high_resolution_clock::now();
  auto curr = high_resolution_clock::now();
  auto start = curr;

  while ((ret >= 0)&&(framenum <NB_FRAMES)) {
    if ((ret = av_read_frame(input_ctx.get(), packet.get())) < 0)
      break;

    if (video_stream == packet.get()->stream_index) {
      ret = decode_write(codec_ctx.get(), packet.get());
      if (frame_timer) {
        curr = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(curr - prev);
        printf("got frame %d \n", framenum);
        std::cout << "took " << duration.count() / 1000.0f << " ms" << std::endl;
        prev = curr;
      }
      framenum++;
    }
    packet.unRef();
  }
  auto end = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(end - start);
  std::cout << " decode first " << NB_FRAMES << " frames took " << duration.count() / 1000.0f << " ms" << std::endl;



  /* flush the decoder */
  ret = decode_write(codec_ctx.get(), NULL);

  //if (output_file)
  //  fclose(output_file);

  av_buffer_unref(&hw_device_ctx);

  return true;
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

