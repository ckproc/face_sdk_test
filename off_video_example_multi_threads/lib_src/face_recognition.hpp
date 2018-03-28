//#include "tensorflow/core/public/session.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <glob.h>
#include <vector>
#include <math.h>
#include "time.h"

//using namespace std;
//using namespace cv;

struct BoundingBox{
        //rect two points
        float x1, y1;
        float x2, y2;
        //regression
        float dx1, dy1;
        float dx2, dy2;
        //cls
        float score;
        //inner points
        float points_x[5];
        float points_y[5];
    };

void load_face_detector(std::string model, int port);

void load_face_embedding(std::string model, int port);

int face_detect(cv::Mat frame  , std::vector<BoundingBox> &faces, int port);

void face_embedding(std::vector<cv::Mat> &aligned_img, std::vector<cv::Mat> &embed, int port);

class Face{
public:
	Face(cv::Rect_<float> bbox, int faceid = -1, float detection_score=-1){
		Bbox = bbox;
		Faceid = faceid;
		Detection_score = detection_score;
	}
	~Face(){}
	void crop_face(cv::Mat img, cv::Rect_<float> bbox,int width,int height ,int target_img_size=160, int margin=0 ){
		cv::Rect_<float> img_rect(0.0, 0.0, float(width), float(height));
		bbox = bbox & img_rect;
		aligned_face = img(bbox);
		cv::resize(aligned_face, aligned_face,cv::Size(target_img_size,target_img_size));
		
	}

	cv::Rect_<float> Bbox;
	int Faceid;
	float Detection_score;
	cv::Mat aligned_face;
};

/*
class mtcnn{
public:
    mtcnn(const string model){
		
	}
	~mtcnn(){
		
	}
	






private:












int align_mtcnn(cv::Mat &image, std::unique_ptr<tensorflow::Session> &session1, std::unique_ptr<tensorflow::Session> &session2,
   std::unique_ptr<tensorflow::Session> &session3, std::vector<BoundingBox> &faces);
   
int Load_model(std::string model,std::unique_ptr<tensorflow::Session> &session1, std::unique_ptr<tensorflow::Session> &session2,
   std::unique_ptr<tensorflow::Session> &session3);
};*/