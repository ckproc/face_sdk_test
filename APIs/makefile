CC = g++
CFLAGS = -fPIC -std=c++11 -DDLIB_JPEG_SUPPORT -O3
INCLUDES = -I./src -I/workspace1/ckp/dlib-19.7 ../lib/detect_embed.pic.o -I/usr/include -I/usr/include/opencv -I/usr/include/opencv2 -Iconfig/include
LIBS = -L../lib -Lconfig/lib/ -ldetect_embed `pkg-config --libs opencv` -lconfig4cpp -lpthread -ljpeg -lX11
SRC = ./src/*.cpp /workspace1/ckp/dlib-19.7/dlib/all/source.cpp test.cpp
EXEC = test
$(EXEC):$(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC) $(INCLUDES) $(LIBS)
clean:
	rm test
#all:
#	g++ -std=c++11 -fPIC test.cpp -I. -O3 -o runtest1 -loff_video_face_recognition1 -L../lib -lpthread
