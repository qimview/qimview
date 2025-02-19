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

test_buffer = False

# Skip decoders that have never been tested
device_skip = ['vdpau','vaapi']

def getFrames(fp, nb, log_timings:bool = False):
    if test_buffer:
        res = []
    total_time_start = perf_counter()
    error = False
    # record start time
    for n in range(nb):
        if log_timings:
            time_start = perf_counter()
        # f = fp._frame_buffer.get_nothread()
        fp.get_next_frame()
        f = fp.frame
        if f is None:
            error = True
            break
        vf = VideoFrame(fp.frame)
        # Check conversion to ViewerImage
        if n==0:
            im = vf.toViewerImage()
            if im is None:
                error = True
                break
        # if im.y is not None:
        #     print(f"{np.average(im.y)=}")
        # if n==0: print(f"{f.getFormat()=}")
        if test_buffer:
            res.append(f)
            print(f"{f.pts=}")
        # print(f"{f.pts=}")
        if log_timings:
            time_end = perf_counter()
            print(f' Frame {n} took {(time_end - time_start)*1000:0.1f} msec', end="; ")
    if error:
        print("Error while decoding")
        return -1
    total_time_end = perf_counter()
    if log_timings:
        print(f'\nTotal time Took {(total_time_end - total_time_start):0.3f} msec FPS {nb/(total_time_end - total_time_start):0.2f}')
    if test_buffer:
        return res
    else:
        return total_time_end - total_time_start


def get_best_device(input_video: str, codec: str, threads: int = -1, nb_frames: int = 6, timings: bool = False) -> str:
    script_start = perf_counter()
    filename = input_video
    hw_device_names = decode_lib.HW.get_device_type_names()
    device_type = codec if codec!="" else None
    if device_type not in hw_device_names:
        hw_device_names.append(device_type)
    stream_number = 0
    hw_device_names_ok = []
    fp = VideoFrameProviderCpp()
    for device_name in hw_device_names:
        dev_time_start = perf_counter()
        if device_name in device_skip:
            continue
        vd = decode_lib.VideoDecoder(VideoConfig.framebuffer_max_size)
        open_ok = vd.open(filename, device_name, stream_number, 
                num_threads = VideoConfig.decoder_thread_count if threads==-1 else threads,
                thread_type = VideoConfig.decoder_thread_type
                )
        open_time = perf_counter()-dev_time_start
        if open_ok:
            fp.set_input_container(vd, stream_number)
            init_time = perf_counter()-dev_time_start
            # fp.set_time(0)
            rewind_time = perf_counter()-dev_time_start
            if test_buffer:
                frames = getFrames(fp,4, log_timings=timings)
                for f in frames:
                    print(f"{f.pts=}")
                hw_device_names_ok.append(device_name)
            else:
                timing = getFrames(fp,nb=nb_frames, log_timings=timings)
                hw_device_names_ok.append((device_name, timing))
            dev_time = perf_counter()-dev_time_start
            print(f"{open_time=:0.3f}, {init_time=:0.3f}, {rewind_time=:0.3f} {dev_time=:0.3f}")
        else:
            print(f" --- {open_ok=} {device_name} --> {vd.useHw()=}")

    script_end = perf_counter()
    print(f" time spent {script_end-script_start:0.3f} sec.")
    print(f"{hw_device_names_ok=}")
    best_device =  min(range(len(hw_device_names_ok)), key=lambda x : hw_device_names_ok[x][1] if hw_device_names_ok[x][1]>0 else 1000 )
    return hw_device_names_ok[best_device][0]


def main():

    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video',      help='input video')
    parser.add_argument('-c','--codec', type=str, default="", help='use hardware decoder device name')
    parser.add_argument('-n','--nb_frames', type=int, default=100, help=' nb of frames to decode')
    parser.add_argument('-t','--timings',   action='store_true', help='logs decoding time per frame')
    parser.add_argument('--threads',   type = int, default=-1,  help='number of threads (-1: use default value)')
    args = parser.parse_args()

    dev = get_best_device(args.input_video, args.codec, args.threads, args.nb_frames, args.timings)
    print(f"best device found is *** {dev} ***")

if __name__ == '__main__':
    main()
