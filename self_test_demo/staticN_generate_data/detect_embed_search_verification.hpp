#include <iostream>
#include <vector>
#include <string>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv.h>
struct search_item{
 int index;
 float img_similarity;
};
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
//using namespace dlib;
//void photo_detect(std::string &img_file, cv::Mat &raw_photo, std::vector<cv::Mat> &img_array);
//void load_detector(std::string model);
//void load_embed(std::string model);

//oid photo_detect(cv::Mat &raw_photo, std::vector<cv::Mat> &img_array);
bool photo_detect(cv::Mat &raw_photo, std::vector<BoundingBox> &faceinfos);
//void detect_one_face(cv::Mat raw_photo, cv::Mat &align_img);
bool detect_one_face(cv::Mat raw_photo, BoundingBox &face);
void face_feature_extract(cv::Mat &face_img, cv::Mat &float_feature);

float  face_verification(cv::Mat &feature_a, cv::Mat &feature_b);

std::vector<search_item>  face_retrieval(cv::Mat &query_array, std::vector<cv::Mat> &db, int top_rank);



void load_face_detector(std::string model);

void load_face_embedding(std::string model);

bool face_detect(cv::Mat frame  , std::vector<BoundingBox> &faces);

bool face_embedding(std::vector<cv::Mat> &aligned_img, std::vector<cv::Mat> &embed);

bool single_face_embedding(cv::Mat &face_img, cv::Mat &float_feature);

void load_align(std::string model,  dlib::shape_predictor &sp);

void face_align(dlib::shape_predictor &sp, cv::Mat &frame, std::vector<BoundingBox> &faceinfos, std::vector<cv::Mat> &align_imgs);