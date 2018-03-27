//#include "detect_embed.hpp"
#include "detect_embed_search_verification.hpp"
#include <algorithm>
#include <math.h>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;
using namespace dlib;

typedef pair<float, int> P;

bool cmp1(const P &a, const P &b){
	return a.first > b.first;
}

bool isNaN(double x) { 
  return x != x;
}
/*
void face_alignment(cv::Mat frame, BoundingBox faceinfo, cv::Mat &align_img){
	faceinfo.x1=std::max(faceinfo.x1, float(0.0));
	faceinfo.y1=std::max(faceinfo.y1, float(0.0));
	faceinfo.x2=std::min(faceinfo.x2, float(frame.cols));
	faceinfo.y2=std::min(faceinfo.y2, float(frame.rows));
	cv::Rect roi;
	roi.x = faceinfo.y1;
    roi.y = faceinfo.x1;
    roi.width = faceinfo.y2-faceinfo.y1;
    roi.height = faceinfo.x2-faceinfo.x1;
	
	Point2f leftEye=Point2f(faceinfo.points_y[0], faceinfo.points_x[0]);
	Point2f rightEye=Point2f(faceinfo.points_y[1], faceinfo.points_x[1]);
	Point2f eyesCenter=Point2f(faceinfo.points_y[2], faceinfo.points_x[2]);
	
	double dy = (rightEye.y - leftEye.y);
    double dx = (rightEye.x - leftEye.x);
    double len = sqrt(dx*dx + dy*dy);
    double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.
    double scale = 1;
	Mat roImg;
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
	if (rot_mat.empty())
		{
			std::cout << "NULL" << std::endl;
			return;
		}
	roImg = Mat::zeros(frame.rows, frame.cols, frame.type());
	warpAffine(frame, roImg, rot_mat, roImg.size());
	align_img = roImg(roi);
	//cv::resize(croprec,croprec,cv::Size(160,160));
}*/


bool photo_detect(cv::Mat &raw_photo, std::vector<BoundingBox> &faceinfos){
	//vector<BoundingBox> faces;
    if(raw_photo.empty()){
        cout<<"Error:Image cannot be loaded!"<<endl;
        return false;                       
		}
      else if(raw_photo.channels()<2){
        return false;                       
		}	
	  else if(raw_photo.channels()==4){
		  cv::cvtColor(raw_photo, raw_photo, CV_BGRA2BGR);
	  }
	//load_face_detector(model);
	bool isdetect = face_detect(raw_photo, faceinfos);
	if(isdetect==false || faceinfos.size()==0){
		return false;
	}
	return true;
/*
	for(int i=0;i<faces.size();++i){
		Mat align_image;
		face_alignment(raw_photo, faces[i], align_image);
		img_array.push_back(align_image);
	}
*/			
}

bool detect_one_face(cv::Mat raw_photo, BoundingBox &face){
	std::vector<BoundingBox> faces;
	//Mat raw_photo=imread(img,-1);
	//cout<<11<<endl;
    if(raw_photo.empty()){
        cout<<"Error:Image cannot be loaded!"<<endl;
        return false;                       
		}
      else if(raw_photo.channels()<2){
		cout<<"channels small than 2."<<endl;
        return false;                       
		}	
	  else if(raw_photo.channels()==4){
		  cv::cvtColor(raw_photo, raw_photo, CV_BGRA2BGR);
	  }
	  
	//Mat c_image=raw_photo.t();
	float img_cw=float(raw_photo.cols)/2.0;
	float img_ch=float(raw_photo.rows)/2.0;
	//load_face_detector(model);
	//cout<<"p4"<<endl;
	bool isdetect = face_detect(raw_photo, faces);
	//cout<<true<<endl;
	//cout<<false<<endl;
	if(isdetect==false || faces.size()==0){
		cout<<"detect failed"<<endl;
		return false;
	}
	
	
	//cout<<"p5"<<endl;
	BoundingBox cand_rect;
	float box_size=(faces[0].x2-faces[0].x1)*(faces[0].y2-faces[0].y1);
	float off_sq = (pow((faces[0].x1+faces[0].x2)/2.0-img_cw, 2.0)+pow((faces[0].y1+faces[0].y2)/2.0-img_ch, 2.0))*2.0;
	float temp_value = box_size-off_sq;
	//cout<<"value_1:"<<temp_value<<"    "<<"size_1:"<<box_size<<"  "<<"sq_1:"<<off_sq<<endl;
	int temp_index=0;
	if(faces.size()>1){
		for(int index=1;index<faces.size();index++){
			
			box_size=(faces[index].x2-faces[index].x1)*(faces[index].y2-faces[index].y1);
			off_sq = (pow((faces[index].x1+faces[index].x2)/2.0-img_cw, 2.0)+pow((faces[index].y1+faces[index].y2)/2.0-img_ch, 2.0))*2.0;
			float index_value=box_size-off_sq;
			//cout<<"value_"<<index<<":"<<index_value<<"    "<<"size_:"<<index<<":"<<box_size<<"  "<<"sq_"<<index<<":"<<off_sq<<endl;
			if(index_value>temp_value){
				temp_index=index;
				temp_value=index_value;
			}
		}
		
	}
	//cout<<"p6"<<endl;
	cand_rect=faces[temp_index];
	cand_rect.x1=std::max(cand_rect.x1, float(0.0));
	cand_rect.y1=std::max(cand_rect.y1, float(0.0));
	cand_rect.x2=std::min(cand_rect.x2, float(raw_photo.cols));
	cand_rect.y2=std::min(cand_rect.y2, float(raw_photo.rows));
	//shape_predictor sp;
	//cout<<"p7"<<endl;
	face = cand_rect;
	return true;
	//face_align(shape_predictor sp, cv::Mat frame, std::vector<BoundingBox> faceinfos, std::vector<cv::Mat> &align_imgs
	//face_alignment(raw_photo, cand_rect, align_img);
	
	//Rect_<float> img_rect(0.0, 0.0, float(raw_photo.cols), float(raw_photo.rows));
	//Rect_<float> face_i = Rect_<float>(Point_<float>(cand_rect.y1, cand_rect.x1), Point_<float>(cand_rect.y2, cand_rect.x2));
	//face_i=face_i&img_rect;
	//align_img = raw_photo(face_i);
	//imshow("1",align_img);
	//waitKey(0);
	//img_array.push_back(aligned_face);
	
}

void face_feature_extract(cv::Mat &face_img, cv::Mat &float_feature){
	//load_face_embedding(model);
	//cout<<"e?"<<endl;
	//cout<<"pp1"<<endl;
	//cv::imshow("1",face_img);
	//cv::waitKey(0);
	cv::resize(face_img, face_img, cv::Size(160,160));
	//cout<<"pp2"<<endl;
	float_feature = Mat(1, 128, CV_32FC1, Scalar::all(0.0) );
	//cout<<"pp3"<<endl;
	//cout<<"p?"<<endl;
	//cout<<face_img.empty()<<endl;
	bool embed = single_face_embedding(face_img, float_feature);
	//cout<<"pp4"<<endl;
	//cout<<"12?"<<endl;
}



float  face_verification(cv::Mat &feature_a, cv::Mat &feature_b){
	double sumA=0;
	double sumB=0;
	double cosine = 0;
	for(int j=0; j<feature_a.cols; ++j){
			sumA+=feature_a.ptr<float>(0)[j]*feature_a.ptr<float>(0)[j];
			sumB+=feature_b.ptr<float>(0)[j]*feature_b.ptr<float>(0)[j];
			cosine+=feature_a.ptr<float>(0)[j]*feature_b.ptr<float>(0)[j];
	}
		sumA=sqrt(sumA);
		sumB=sqrt(sumB);
		cosine/=sumA*sumB;
		cosine=float(cosine);
	return cosine;
}

std::vector<search_item>  face_retrieval(cv::Mat &query_array, std::vector<cv::Mat> &db, int top_rank){
	std::vector<P> cdists;
	std::vector<search_item> result;
	for(int i=0; i<db.size(); ++i){
		double sumA=0;
		double sumB=0;
		double cosine = 0;
		for(int j=0; j<query_array.cols; ++j){
			sumA+=db[i].ptr<float>(0)[j]*db[i].ptr<float>(0)[j];
			sumB+=query_array.ptr<float>(0)[j]*query_array.ptr<float>(0)[j];
			cosine+=db[i].ptr<float>(0)[j]*query_array.ptr<float>(0)[j];
		}
		sumA=sqrt(sumA);
		sumB=sqrt(sumB);
		cosine/=sumA*sumB;
		if(isNaN(cosine)){
			cdists.push_back(make_pair(0.0,i));
		}
		else{
			cdists.push_back(make_pair(cosine,i));
		}
		//cdists.push_back(make_pair(cosine,i));
	}
	
	sort(cdists.begin(),cdists.end(),cmp1);
	//match_idx = cdists[0].second;
	//identification_score = cdists[0].first;
	for(int k=0;k<top_rank;++k){	
		search_item temp;
		temp.index=cdists[k].second;
		temp.img_similarity=cdists[k].first;
		result.push_back(temp);
	}
	return result;
}


void load_align(std::string model,  shape_predictor &sp){
	deserialize(model) >> sp;
}

void face_align(shape_predictor &sp, cv::Mat &frame, std::vector<BoundingBox> &faceinfos, std::vector<cv::Mat> &align_imgs){

	std::vector<dlib::rectangle> dets;
	//double tt=(double)cv::getTickCount();
	for(int k=0;k<faceinfos.size();++k){
		faceinfos[k].x1=std::max(faceinfos[k].x1, float(0.0)); //cout<<faceinfos[k].x1<<endl;
		faceinfos[k].y1=std::max(faceinfos[k].y1, float(0.0)); //cout<<faceinfos[k].y1<<endl;
		faceinfos[k].x2=std::min(faceinfos[k].x2, float(frame.rows)); //cout<<faceinfos[k].x2<<endl;
		faceinfos[k].y2=std::min(faceinfos[k].y2, float(frame.cols)); //cout<<faceinfos[k].y2<<endl;
        //dlib::rectangle det(faceinfos[k].y1,faceinfos[k].x1,faceinfos[k].y2,faceinfos[k].x2);
        dlib::rectangle det(faceinfos[k].x1,faceinfos[k].y1,faceinfos[k].x2,faceinfos[k].y2);
		dets.push_back(det);
		
	}
	//t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
	//cout<<"copy time:"<<t<<endl;
	//cout<<"p1"<<endl;
	array2d<rgb_pixel> img;
	//array2d<uchar> gray;
	//cv::Mat frame_gray;
	//cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
	//assign_image(gray,dlib::cv_image<uchar>(frame_gray));
	assign_image(img,dlib::cv_image<bgr_pixel>(frame));
    std::vector<full_object_detection> shapes;
	//double t=(double)cv::getTickCount();
	for (unsigned long j = 0; j < dets.size(); ++j)
    {
                full_object_detection shape = sp(img, dets[j]);
                shapes.push_back(shape);
    }
	
	
		//t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
	    //cout<<"sp time:"<<t<<"		";	
			dlib::array<array2d<rgb_pixel> > face_chips;
			//double t1=(double)cv::getTickCount();
            extract_image_chips(img, get_face_chip_details(shapes), face_chips);
			//t1=((double)cv::getTickCount()-t1)/cv::getTickFrequency();
			//cout<<"extract time:"<<t1<<endl;
			
			//double t2=(double)cv::getTickCount();
		for(unsigned long p=0;p<face_chips.size();++p){
				cv::Mat face_image = dlib::toMat(face_chips[p]).clone();
				cv::cvtColor(face_image, face_image, CV_RGB2BGR);
				align_imgs.push_back(face_image);
				//cv::imshow("1",face_image);
				//cv::waitKey(0);
				
		}	
		//t2=((double)cv::getTickCount()-t2)/cv::getTickFrequency();
	//cout<<"push time:"<<t2<<endl;
	//cout<<"p2"<<endl;
	//tt=((double)cv::getTickCount()-tt)/cv::getTickFrequency();
	   // cout<<"what time:"<<tt<<endl;	
}
