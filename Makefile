# Makefile
v = 0
exe = face.exe
make:
	g++ -o $(exe) face.cpp /usr/local/opencv-2.4/lib/libopencv_core.so.2.4 /usr/local/opencv-2.4/lib/libopencv_highgui.so.2.4 /usr/local/opencv-2.4/lib/libopencv_imgproc.so.2.4 /usr/local/opencv-2.4/lib/libopencv_objdetect.so.2.4
run:
	g++ -o $(exe) face.cpp /usr/local/opencv-2.4/lib/libopencv_core.so.2.4 /usr/local/opencv-2.4/lib/libopencv_highgui.so.2.4 /usr/local/opencv-2.4/lib/libopencv_imgproc.so.2.4 /usr/local/opencv-2.4/lib/libopencv_objdetect.so.2.4
	./$(exe) testImages/dart$(v).jpg
clean:
	rm *.out
	rm *.exe
	rm detected.jpg
