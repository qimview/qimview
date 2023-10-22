# Description
_qimview_ (qt-based image view) is a set of classes that are designed to visualize and compare images. 
It uses Python as the main language, and include some code in C++ or OpenGL shading language.
Main features are:
* **image reader**: read type image format (like jpeg and png) and also raw image format (bayer images)
* **image cache**: save image reading/uncompressing time by loading previously read image into a buffer
* **image viewer**: different image viewers are available
  * QT based without OpenGL
  * OpenGL based 
* image viewer can have many **features**:
  * zomm/pan
  * display cursor and image information
  * display histogram
* **multiple image viewer** to compare a set of images
* **video viewer** is experimental and based on python vlc binding
* image **filters** can be applied:
  * black/white level
  * saturation
  * white balance

## cppimport
When displaying images with Qt, C++ code is used to speed-up the processing.
The code is compiled and bound to Python using pybind11 and the module cppimport.
However, the first time you use it, it may not be able to compile the code automatically (I don't know why).
In this case, you can run manually:

  python -m cppimport build ./qimview/CppBind

from the qimview folder. Even of the command ends up with the error 'No module named qimview_cpp', it has probably built the library correctly.

## Installation issue:

PyYAML version 5.4 required by CppBind can create installation errors, I was able to get around this issue by running:

pip3 install wheel -v
pip3 install "cython<3.0.0" pyyaml==5.4 --no-build-isolation -v
