"""
    Sets video player configuration variables
        either from the configuration file if available
        or sets default values
"""

import os
import configparser
from dataclasses import dataclass

config = configparser.ConfigParser()
res = config.read([os.path.expanduser('~/.qimview.cfg')])

@dataclass
class VideoConfig:
    """
        Configuration parameters for video player
    """
    # Default values
    mipmap_max_level     : int = 0
    decoder_thread_type  : str = "FRAME"
    decoder_thread_count : int = 4
    framebuffer_max_size : int = 10

if res:
    VideoConfig.mipmap_max_level     = config.getint('VIDEOPLAYER', 'mipmap_max_level',
                                                   fallback=VideoConfig.mipmap_max_level)
    VideoConfig.decoder_thread_type = config.get('VIDEOPLAYER', 'decoder_thread_type',
                                                   fallback=VideoConfig.decoder_thread_type)
    VideoConfig.decoder_thread_count = config.getint('VIDEOPLAYER', 'decoder_thread_count',
                                                   fallback=VideoConfig.decoder_thread_count)
    VideoConfig.framebuffer_max_size = config.getint('VIDEOPLAYER', 'framebuffer_max_size',
                                                   fallback=VideoConfig.framebuffer_max_size)
    print(f"{VideoConfig.mipmap_max_level=}")
    print(f"{VideoConfig.decoder_thread_type=}")
    print(f"{VideoConfig.decoder_thread_count=}")
    print(f"{VideoConfig.framebuffer_max_size=}")
