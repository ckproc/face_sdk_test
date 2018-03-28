#all:
#	g++ -std=c++11 main.cpp KalmanTracker.cpp Hungarian.cpp -o test -I. -I/home/ckp/tensorflow /home/ckp/tensorflow/bazel-bin/tensorflow/came_mtcnn/_objs/libcame_mtcnn.so/tensorflow/came_mtcnn/detect_faces.pic.o -I/usr/include -I/usr/include/opencv -I/usr/include/opencv2 `pkg-config --libs opencv` -lcame_mtcnn -L/home/ckp/lib -lpthread -lprotobuf
all:
	g++ -std=c++11 -fPIC off_video_face_recognition.cpp KalmanTracker.cpp Hungarian.cpp -shared -O3 -o liboff_video_face_recognition1.so -I. ../../lib/face_recognition.pic.o -I/usr/include -I/usr/include/opencv -I/usr/include/opencv2 `pkg-config --libs opencv` -lface_recognition1 -L../../lib 
#g++ -std=c++11 -fPIC test.cpp -I. -o runtest -loff_video_face_recognition -L. -lprotobuf -lpthread

#-I/usr/include -I/usr/include/opencv -I/usr/include/opencv2 `pkg-config --libs opencv`	
