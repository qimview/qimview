# Description
_ImComp_ (Image Comparison) is a set of classes that are designed to visualize and compare images. 
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