all:
	g++-4.8 -std=c++11 -fPIC -mavx -DDLIB_JPEG_SUPPORT detect_embed_search_verification.cpp /workspace1/ckp/dlib-19.7/dlib/all/source.cpp -shared -O3 -o libembed.so -I. -I/workspace1/ckp/dlib-19.7 /workspace1/ckp/mclab_test/lib/detect_embed.pic.o -I/usr/include -I/usr/include/opencv -I/usr/include/opencv2 `pkg-config --libs opencv` -ldetect_embed -L/workspace1/ckp/mclab_test/lib
