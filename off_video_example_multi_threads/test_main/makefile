CC = g++
CFLAGS = -fPIC -std=c++11 -O3
INCLUDES = -I./include -Iconfig/include
LIBS = -L../../lib -Lconfig/lib/ -loff_video_face_recognition1 -lconfig4cpp -lpthread
SRC = ./src/*.cpp
EXEC = runtest1
$(EXEC):$(SRC)
	$(CC) $(CFLAGS) -o $(EXEC) $(SRC) $(INCLUDES) $(LIBS)
clean:
	rm runtest1
#all:
#	g++ -std=c++11 -fPIC test.cpp -I. -O3 -o runtest1 -loff_video_face_recognition1 -L../lib -lpthread
