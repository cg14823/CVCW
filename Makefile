make:
  g++  face.cpp  /usr/local/opencv-2.4/lib/libopencv_core.so.2.4
    /usr/local/opencv-2.4/lib/libopencv_highgui.so.2.4
    /usr/local/opencv-2.4/lib/libopencv_imgproc.so.2.4
    /usr/local/opencv-2.4/lib/libopencv_objdetect.so. -0 face.exe

clean:
  rm *.exe
