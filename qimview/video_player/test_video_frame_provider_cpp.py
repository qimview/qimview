import os
ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib
from qimview.video_player.video_frame_provider     import VideoFrameProvider
from qimview.video_player.video_frame_provider_cpp import VideoFrameProviderCpp
from time import perf_counter
from qimview.video_player.video_player_config      import VideoConfig


def main():

    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_video',      help='input video')
    parser.add_argument('-c','--cuda',      action='store_true', help='use cuda hardware')
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
            f = fp._frame
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

    vd = decode_lib.VideoDecoder(VideoConfig.framebuffer_max_size)

    filename = args.input_video
    device_type = "cuda" if args.cuda else None
    stream_number = 0
    vd.open(filename, device_type, stream_number, 
            num_threads = VideoConfig.decoder_thread_count,
            thread_type = VideoConfig.decoder_thread_type
            )
    print(f"open ok {args=}")
    fp = VideoFrameProviderCpp()
    fp.set_input_container(vd, stream_number)
    fp.set_time(0)
    if test_buffer:
        frames = getFrames(fp,4, log_timings=args.timings)
        for f in frames:
            print(f"{f.pts=}")
    else:
        getFrames(fp,nb=args.nb_frames, log_timings=args.timings)

if __name__ == '__main__':
    main()
