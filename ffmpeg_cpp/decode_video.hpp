
#pragma once

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


namespace AV
{

  //-----------------------------------------------------------------------------------------------
  class AVException : public std::exception {
  public:
    AVException(const char* msg);
    AVException(const std::string& msg);
    const char * what() const;
  private:
    std::string _message;
  };

  //-----------------------------------------------------------------------------------------------
  class Packet
  {
  public:
    Packet();
    ~Packet();
    void unRef();
    AVPacket* get() { return _packet;  }
  private:
    AVPacket* _packet;
  };

  //-----------------------------------------------------------------------------------------------
  class FormatContext
  {
  public:
    FormatContext(const char* filename=nullptr);
    ~FormatContext();
    AVFormatContext*  get()    { return _format_ctx; }
    AVFormatContext** getPtr() { return &_format_ctx; }
    bool fileOpened() const    { return _file_opened; }

    void openFile(const char* filename);
    int findFirstValidVideoStream();
    AVCodecParameters* getCodecParams(const int& stream_index);
    const AVCodec* findDecoder(const int& stream_index);
    void findStreamInfo();
    int findBestVideoStream(const AVCodec** codec);
    bool readVideoFrame(AVPacket* pkt, int video_stream_index);
  private:
    AVFormatContext* _format_ctx;
    bool             _file_opened;
  };

  //-----------------------------------------------------------------------------------------------
  class CodecContext
  {
  public:
    CodecContext(const AVCodec* codec);
    ~CodecContext();
    AVCodecContext* get() { return _codec_ctx; }
    int initFromParam(const AVCodecParameters* params);
    void setThreading( int count=8, int type = FF_THREAD_FRAME);
    int initHw(const enum AVHWDeviceType type);
    void open(const AVCodec* codec, AVDictionary** options=nullptr);
    int receiveFrame(const AVPacket* pkt, AVFrame* frame);

  private:
    AVCodecContext* _codec_ctx;
  };

  //-----------------------------------------------------------------------------------------------
  class Frame
  {
  public:
    Frame();
    ~Frame();
    AVFrame* get() { return _frame; }
    /**
      if the current frame is of hw_pix_format, convert it to CPU frame in the cpu_frame argument variable
      return the converted or initial frame pointer
    */
    AVFrame* gpu2Cpu(const AVPixelFormat& hw_pix_fmt, AV::Frame& cpu_frame);
  private:
    AVFrame* _frame;
  };

} // end namespace AV

#define CHECK_RETURN(condition, message, returnvalue) if(!(condition)) { fprintf(stderr, message); return returnvalue; }


//-------------------------------------------------------------------------------------------------
// decode HW
//-------------------------------------------------------------------------------------------------

//
// Add AV::HW namespace
//
namespace AV {
  namespace HW {

    // TODO: Check if this callback is really needed
    AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts, 
    const enum AVPixelFormat& pixel_format);

    AVHWDeviceType get_device_type(const char* device_type_name);

    const AVCodecHWConfig* get_codec_hwconfig(const AVCodec *codec, const AVHWDeviceType & device_type);
  }
}
