import os
ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib
from qimview.video_player.video_frame_provider_cpp import VideoFrameProviderCpp
from qimview.video_player.video_frame_buffer_cpp import VideoFrameBufferCpp
from time import perf_counter


def getFrames(fp, nb):
    total_time_start = perf_counter()
    # record start time
    for n in range(nb):
        # time_start = perf_counter()
        fp.get_next_frame()
        # print(f"{fp._frame.pts=} {decode_lib.AV_NOPTS_VALUE=}")
        # time_end = perf_counter()
        # print(f'Took {(time_end - time_start)*1000:0.1f} msec', end="; ")
    total_time_end = perf_counter()
    print(f'\nTotal time Took {(total_time_end - total_time_start)} msec')

vd = decode_lib.VideoDecoder()

filename = "C:/Users/karl/Videos/GX010296.MP4"
device_type = None # = "cuda"
vd.open(filename, device_type)
print("open ok")
fp = VideoFrameProviderCpp()
fp.set_input_container(vd)
getFrames(fp,200)
