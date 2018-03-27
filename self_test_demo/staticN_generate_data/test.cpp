#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <string>
//#include <map>
#include <glob.h>
#include "detect_embed_search_verification.hpp"
using namespace std;
using namespace cv;

void getname(string &match_face){
	const size_t last_slash_idx = match_face.find_last_of("/");
	if (std::string::npos != last_slash_idx)
	{
		match_face.erase(0, last_slash_idx + 1);
		}
		const size_t period_idx = match_face.rfind('.');
		if (std::string::npos != period_idx)
	{
		match_face.erase(period_idx);
	}
	
}


inline std::vector<std::string> glob(const std::string& pat){
    
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    std::vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}


int main(int argc, const char *argv[]){
	
	std::string db_dir = std::string(argv[1]);//query/  db/
	std::string output_file = std::string(argv[2]);//1k_test_data/align
	std::string model_path = std::string(argv[3]);
	std::string model_align_path = std::string(argv[4]);
	//float threshold = atof(argv[5]);
	float tp=0.0;
	float fp=0.0;
	float tn=0.0;
	int p=0;
	int num=0;
	float accuracy;

	//std::vector<std::vector<string>> lfw_data_list;
	//load_lfw_data_path(pair_path, lfw_data_path, lfw_data_list);
    dlib::shape_predictor sp;
	load_align(model_align_path,sp);
	load_face_detector(model_path);
	load_face_embedding(model_path);
    //string img ="/home/ckp/data/db/Angela_Bassett_0005.jpg";
    db_dir=db_dir+"/*.*";
    vector<string> db_path=glob(db_dir);
	cout<<"Running forward pass on images"<<endl;
	//cout<<lfw_data_list.size()<<endl;
	//ofstream infile;
	//string write_file ="testdata.txt";
	//infile.open(write_file,ios::trunc);
	//int ndt=0;
	std::vector<std::string> embed_path;
	cv::Mat emb_array;//=Mat(db_path.size(), 128, CV_32FC1, Scalar::all(0.0));
	double total_time1 = 0.0;
	double total_time2 = 0.0;
	double total_time3 = 0.0;
	for(int i=0;i<db_path.size();++i){
		//cout<<"pair"<<i<<endl;
		string image_path=db_path[i];
		cout<<image_path<<endl;
		Mat image = imread(image_path,-1);
		if(image.empty()){
        cout<<"Error:Image cannot be loaded!"<<endl;
        continue;                       
		}
		
       else if(image.channels()<2){
        cout<<"this is gray image"<<endl;
		//cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
        continue;                       
		}
		
	  else if(image.channels()==4){
		
		  cv::cvtColor(image, image, CV_BGRA2BGR);
	  }
		//cout<<"p1"<<endl;
	
		//int width_std = 300;
		//if ( image.cols > width_std )
		//{
			//cv::resize(image,image,cv::Size(width_std, (int)((float)(image.rows*width_std/image.cols))));	
		//}
		
		//cv::resize(image_a, image,cv::Size(160,160));
		BoundingBox face_a;
		//BoundingBox face_b;
		//cout<<"p1"<<endl;
		double t=(double)cv::getTickCount();
		bool isdetect = detect_one_face(image, face_a); 
		t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
		total_time1 +=t;
		
		//cout<<"p2"<<endl;
	    //cv::rectangle(image, cv::Point(face_a.x1, face_a.y1), cv::Point(face_a.x2, face_a.y2), cv::Scalar(255, 0, 0), 2);
	     
		//getname(image_path);
		//string wp="./align/"+image_path+".jpg";
	    //cv::imwrite(wp, image);
		//continue;
    //cv::waitKey(0);;
		//t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
		//cout<<"detect time:"<<t<<endl;
		//detect_one_face(image_b, face_b);
		if(!isdetect){
			continue;
		}
		//cout<<face_a.x1<<","<<face_a.y1<<endl;
		//cout<<face_b<<endl;
		std::vector<cv::Mat> align_imgs_a;
		
		std::vector<BoundingBox> faceinfos_a;
		faceinfos_a.push_back(face_a);

		//cv::resize(align_imgs_a[0], align_imgs_a[0],cv::Size(160,160));
		//cout<<"p3"<<endl;
		double t0=(double)cv::getTickCount();
		face_align(sp, image, faceinfos_a, align_imgs_a);
		//cout<<align_imgs_a[0].rows<<" "<<align_imgs_a[0].cols<<endl;
		//cv::imshow("1",align_imgs_a[0]);
		//getname(image_path);
		//string wp="./align/"+image_path+".jpg";
		//cv::imwrite(wp,align_imgs_a[0]);
		//continue;//cv::waitKey(0);
		t0=((double)cv::getTickCount()-t0)/cv::getTickFrequency();
		total_time2+=t0;
		//cv::Rect_<float> img_rect(0.0, 0.0, float(image.cols), float(image.rows));
	    //cv::Rect_<float> face_i = Rect_<float>(Point_<float>(face_a.y1, face_a.x1), Point_<float>(face_a.y2, face_a.x2));
	    //face_i=face_i&img_rect;
	    //cv::Mat align_img = image(face_i);
		
		//cout<<"align time:"<<t0<<endl;
		if(align_imgs_a.size()==0){
			cout<<"can not align:"<<image_path<<endl;
			continue;
		}
		Mat float_feature_a;	
		//double t1=(double)cv::getTickCount();
		//getname(image_path);
		//string wp = "./align/"+image_path+".jpg";
		//cout<<wp<<endl;
		//cv::imwrite(wp,align_imgs_a[0]);
		//continue;
		double t1=(double)cv::getTickCount();
		face_feature_extract(align_imgs_a[0], float_feature_a);
		t1=((double)cv::getTickCount()-t1)/cv::getTickFrequency();
		total_time3 +=t1;
		
		//face_feature_extract(align_img, float_feature_a);
		
		embed_path.push_back(image_path);
		emb_array.push_back(float_feature_a);
		
	}
	cout<<"avg detect time:"<<total_time1/double(db_path.size())<<endl;
	cout<<"avg sp time:"<<total_time2/double(db_path.size())<<endl;
	cout<<"avg embed time:"<<total_time3/double(db_path.size())<<endl;
	cout<<embed_path.size()<<endl;
	cout<<emb_array.rows<<endl;
	cout<<"Save embeddings of all images into "<<output_file<<endl;
    FileStorage fs(output_file,FileStorage::WRITE);
    fs <<"embed"<<emb_array;
    fs <<"path"<<embed_path;
    fs.release();
	return 0;
}
