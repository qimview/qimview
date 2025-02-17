import os
ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import numpy as np
import decode_video_py as decode_lib
from qimview.video_player.video_frame_provider     import VideoFrameProvider
from qimview.video_player.video_frame_provider_cpp import VideoFrameProviderCpp
from time import perf_counter
from qimview.video_player.video_player_config      import VideoConfig
from qimview.video_player.video_frame              import VideoFrame


def main():

    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video',      help='input video')
    parser.add_argument('-c','--codec', type=str, default="", help='use hardware decoder device name')
    parser.add_argument('-n','--nb_frames', type=int, default=100, help=' nb of frames to decode')
    parser.add_argument('-t','--timings',   action='store_true', help='logs decoding time per frame')
    args = parser.parse_args()

    test_buffer = False

    def getFrames(fp, nb, log_timings:bool = False):
        if test_buffer:
            res = []
        total_time_start = perf_counter()
        # record start time
        for n in range(nb):
            if log_timings:
                time_start = perf_counter()
            # f = fp._frame_buffer.get_nothread()
            fp.get_next_frame()
            f = fp.frame
            vf = VideoFrame(fp.frame)
            im = vf.toViewerImage()
            # if im.y is not None:
            #     print(f"{np.average(im.y)=}")
            # if n==0: print(f"{f.getFormat()=}")
            if test_buffer:
                res.append(f)
                print(f"{f.pts=}")
            if log_timings:
                time_end = perf_counter()
                print(f' Frame {n} took {(time_end - time_start)*1000:0.1f} msec', end="; ")
        total_time_end = perf_counter()
        print(f'\nTotal time Took {(total_time_end - total_time_start)} msec')
        if test_buffer:
            return res
        else:
            return total_time_end - total_time_start


    filename = args.input_video
    hw_device_names = decode_lib.HW.get_device_type_names()
    device_type = args.codec if args.codec!="" else None
    stream_number = 0
    hw_device_names_ok = []
    for device_name in hw_device_names:
        vd = decode_lib.VideoDecoder(VideoConfig.framebuffer_max_size)
        vd.open(filename, device_name, stream_number, 
                num_threads = VideoConfig.decoder_thread_count,
                thread_type = VideoConfig.decoder_thread_type
                )
        print(f" --- {device_name} --> {vd.useHw()=}")
        if vd.useHw():
            print(f"open ok {args=}")
            fp = VideoFrameProviderCpp()
            fp.set_input_container(vd, stream_number)
            fp.set_time(0)
            if test_buffer:
                frames = getFrames(fp,4, log_timings=args.timings)
                for f in frames:
                    print(f"{f.pts=}")
                hw_device_names_ok.append(device_name)
            else:
                timing = getFrames(fp,nb=args.nb_frames, log_timings=args.timings)
                hw_device_names_ok.append((device_name, timing))
    
    print(f"{hw_device_names_ok=}")


if __name__ == '__main__':
    main()
