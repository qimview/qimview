
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
#include <array>
#include <utility>

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
    void closeFile();
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
    std::pair<int,int> receiveFrame(const AVPacket* pkt, AVFrame* frame, bool send_packet=true);

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
        // std::cout << "pts: " << _frame->pts << "  vs ";
        // std::cout << "best effort_timestamp: " << _frame->best_effort_timestamp <<  std::endl;
        // //std::cout << "pkt_dts: " << _frame->pkt_dts <<  std::endl;
        return _frame->pts; 
    }

    uint64_t best_effort_timestamp() { 
        return  _frame->best_effort_timestamp; 
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

    std::vector<int> getLinesizeAll() const {
      std::vector<int> res;
      int pos = 0;
      while ((_frame->linesize[pos] != 0) && (pos < AV_NUM_DATA_POINTERS))
      {
        res.push_back(_frame->linesize[pos]);
        pos++;
      }
      return res;
    }

    int getLinesize( const int& pos) const {
        return _frame->linesize[pos];
    }

    std::tuple<int, int> getShape() const {
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

    std::vector<AVHWDeviceType> get_device_type(const char* device_type_name);

    std::vector<AVHWDeviceType> get_device_types();

    std::vector<std::string> get_device_type_names();

    const AVCodecHWConfig* get_codec_hwconfig(const AVCodec *codec, const AVHWDeviceType & device_type);
  }
}

namespace AV {
  class VideoDecoder
  {
  public:
    VideoDecoder( const int nb_frames = 8): 
      _stream_index(-1), 
      _video_stream_index(-1),
      _framenum(0),
      _current_frame_idx(0) 
    /*
      VideoDecoder constructor
      Its hold a given number of pre-allocated frames, used like a circular buffer
    */
    {
        if (nb_frames <= 0) {
            std::cout << "VideoDecoder constructor, invalid parameter nb_frames=" << nb_frames << " setting value to 2" << std::endl;
            _init_frames(2);
        }
        else {
            _init_frames(nb_frames);
        }
    }

    ~VideoDecoder()
    {
        std::cout << "~VideoDecoder()" << std::endl;
      _delete_frames();
    }

    bool check_hw_device(const char* device_type_name, const AVCodec *codec);
    bool open(  const char* filename, const char* device_type_name = nullptr, 
                const int& video_stream_index=-1, const int& num_threads = 4,
                const std::string&  thread_type = "FRAME");
    bool seek(int64_t timestamp);
    bool seek_file(int64_t timestamp);
    int  nextFrame(bool convert=true);
    int  frameNumber() { return _framenum; }

    int get_nb_frames() { return _nb_frames; }

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

    bool useHw() const { return this->_use_hw; }

    // TODO: check if we can avoid hw_pix_fmt to be static!
    static AVPixelFormat hw_pix_fmt;

  private:
    std::string       _filename;
    AV::FormatContext _format_ctx;
    AV::CodecContext  _codec_ctx;
    int               _video_stream_index; // Index among videos streams
    int               _stream_index;       // Index of the video stream among all streams
    // Use an array of frames to feed the Frame Buffer without memory overlap issues
    int               _nb_frames;
    std::vector<AV::Frame*> _frames;
    bool              _use_hw;
    AV::Frame         _gpu_frame;
    int               _current_frame_idx;
    AV::Packet        _packet;
    int               _framenum;
    AV::Frame*        _current_frame;

    void _init_frames(const int nb_frames)
    {
      _nb_frames = nb_frames;
      _frames.resize(_nb_frames);
      for(auto &fptr: _frames)
        fptr = new AV::Frame(); 
    }

    void _delete_frames()
    {
      for(auto &fptr: _frames)
        delete fptr;
      _frames.clear(); 
    }
  };

  bool load_frames( const char* filename,
                    const char* device_type_name = nullptr, 
                    int nb_frames = 200);
}
