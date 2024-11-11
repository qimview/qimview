import os
ffmpeg_path = os.path.join(os.environ.get('FFMPEG_ROOT', ''),'bin')
if os.name == 'nt' and os.path.isdir(ffmpeg_path):
    os.add_dll_directory(ffmpeg_path)

import decode_video_py as decode_lib
from qimview.video_player.video_frame_provider     import VideoFrameProvider
from qimview.video_player.video_frame_provider_cpp import VideoFrameProviderCpp
from time import perf_counter


def getFrames(fp, nb):
    # res = []
    total_time_start = perf_counter()
    # record start time
    for n in range(nb):
        # time_start = perf_counter()
        # f = fp._frame_buffer.get_nothread()
        fp.get_next_frame()
        f = fp._frame
        # res.append(f)
        # print(f"{f.pts=}")
        # time_end = perf_counter()
        # print(f'Took {(time_end - time_start)*1000:0.1f} msec', end="; ")
    total_time_end = perf_counter()
    print(f'\nTotal time Took {(total_time_end - total_time_start)} msec')
    # return res

vd = decode_lib.VideoDecoder()

filename = "C:/Users/karl/Videos/GX010296.MP4"
device_type = "cuda"
vd.open(filename, device_type, 0, num_threads = 4)
print("open ok")
fp = VideoFrameProviderCpp()
fp.set_input_container(vd)
getFrames(fp,100)
# for f in frames:
#     print(f"{f.pts=}")