
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "decode_video.hpp"

//-------------------------------------------------------------------------------------------------
// main
//-------------------------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  if ((argc <= 1) || (argc > 3)) {
    fprintf(stderr, "Usage: %s <input file> [hw_device_type]\n"
      , argv[0]);
    exit(0);
  }
  std::string filename(argv[1]);
  if (argc == 2) {
    //int frame_width, frame_height;
    //unsigned char* frame_data;
    if (!AV::load_frames(filename.c_str()))
    {
      printf("Couldn't load video frame\n");
      return 1;
    }
  }
  else if (argc == 3)
  {
    std::string hwdevice(argv[2]);
    if (!AV::load_frames(filename.c_str(), hwdevice.c_str()))
    {
      printf("Couldn't load video frame with hw decoder\n");
      return 1;
    }
  }

}

