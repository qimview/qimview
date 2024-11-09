
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
#include <libavutil/pixfmt.h>
}

#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <tuple>

namespace py = pybind11;

namespace AV
{

  //-----------------------------------------------------------------------------------------------
  class AVException : public std::exception {
  public:
    AVException(const char* msg);
    AVException(const std::string& msg);
    const char * what() const noexcept;
  private:
    std::string _message;
  };

  std::string averror2str(int averror);

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
    int getStreamIndex(const int& video_stream_index);
    AVCodecParameters* getCodecParams(const int& stream_index);
    const AVCodec* findDecoder(const int& stream_index);
    void findStreamInfo();
    int findBestVideoStream(const AVCodec** codec);
    int readVideoFrame(AVPacket* pkt, int video_stream_index);
  private:
    AVFormatContext* _format_ctx;
    bool             _file_opened;
  };

  //-----------------------------------------------------------------------------------------------
  class CodecContext
  {
  public:
    CodecContext();
    CodecContext(const AVCodec* codec);
    ~CodecContext();
    void alloc(const AVCodec* codec);
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
    AVFrame* get() { 
        return _frame; 
    }
    uint64_t pts() { 
        // std::cout << "pts: " << _frame->pts << "  " << AV_NOPTS_VALUE << std::endl;
        // std::cout << "pkt_dts: " << _frame->pkt_dts <<  std::endl;
        // std::cout << "best effort_timestamp: " << _frame->best_effort_timestamp <<  std::endl;
        // std::cout << "best effort pts: " << av_frame_get_best_effort_timestamp ( _frame ) << std::endl;
        return _frame->pts; 
    }

    int getFlags()
    {
        return _frame->flags;
    }
    void setFlags(const int& flags)
    {
        _frame->flags = flags;
    }

    bool key_frame()
    {
        return _frame->flags && AV_FRAME_FLAG_KEY;
    }

    bool interlaced_frame()
    {
        return _frame->flags && AV_FRAME_FLAG_KEY;
    }

    AVPixelFormat getFormat() { return (AVPixelFormat) _frame->format;  }
    std::vector<int> getLinesize() {
      std::vector<int> res;
      int pos = 0;
      while ((_frame->linesize[pos] != 0) && (pos < AV_NUM_DATA_POINTERS))
      {
        res.push_back(_frame->linesize[pos]);
        pos++;
      }
      return res;
    }

    std::tuple<int, int> getShape() {
      return std::tuple<int, int>(_frame->height, _frame->width);
    }
    /**
      if the current frame is of hw_pix_format, convert it to CPU frame in the cpu_frame argument variable
      return the converted or initial frame pointer
    */
    Frame*   gpu2Cpu(const AVPixelFormat& hw_pix_fmt, Frame* cpu_frame);
    /**
      Returns python memoryview from the frame data
      currently only the Y channel
    */
    py::object getData(const int& channel, const int& height, const int& width,  bool verbose=false);
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

namespace AV {
  class VideoDecoder
  {
  public:
    VideoDecoder(): _stream_index(-1), _video_stream_index(-1),_framenum(0) {}
    bool open(  const char* filename, const char* device_type_name = nullptr, 
                const int& video_stream_index=-1, const int& num_threads = 4);
    bool seek(int64_t timestamp);
    bool seek_file(int64_t timestamp);
    int  nextFrame(bool convert=true);
    int  frameNumber() { return _framenum; }

    AV::Frame* getFrame() const;

    AVStream* getStream( int idx=-1)
    { 
      idx = std::min(idx,(int)_format_ctx.get()->nb_streams-1);
      return _format_ctx.get()->streams[((idx<0)?_stream_index:idx)]; 
    }

    AVFormatContext* getFormatContext()
    {
      return _format_ctx.get();
    }

    static AVPixelFormat hw_pix_fmt;

  private:
    std::string       _filename;
    AV::FormatContext _format_ctx;
    AV::CodecContext  _codec_ctx;
    int               _video_stream_index; // Index among videos streams
    int               _stream_index;       // Index of the video stream among all streams
    AV::Frame         _frame;
    AV::Frame         _sw_frame;
    AV::Packet        _packet;
    int               _framenum;
    AV::Frame*        _current_frame;
  };

  bool load_frames( const char* filename,
                    const char* device_type_name = nullptr, 
                    int nb_frames = 200);
}
