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

std::vector<string> &split(const string &str, char delim, std::vector<string> &elems) {
    istringstream iss(str);
    for (string item; getline(iss, item, delim); )
        elems.push_back(item);
    return elems;
}
string some_function(int n, int len)
{
    string result(len--, '0');
    for (int val=(n<0)?-n:n; len>=0&&val!=0; --len,val/=10)
       result[len]='0'+val%10;
    if (len>=0&&n<0) result[0]='-';
    return result;
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
void load_lfw_data_path(string pair_path,string lfw_data_path, vector<vector<string>> &lfw_data_list){
	ifstream fin(pair_path);
	string s;
	while(getline(fin,s)){
		std::vector<string> result;
		split(s,' ', result);
		//match_query[result[0]]=atoi(result[1].c_str());
		std::vector<string> temp(3);
		temp[0]=lfw_data_path+result[0].substr(1,result[0].length()-1);
		temp[1]=lfw_data_path+result[1].substr(1,result[1].length()-1);
		temp[2]=result[2];
		lfw_data_list.push_back(temp);
		/*
		if(result.size()==3){
			//vector<string> temp(3);
			string num_a=some_function(atoi(result[1].c_str()),4);
			string num_b=some_function(atoi(result[2].c_str()),4);
			temp[0]=lfw_data_path+"/"+result[0]+"/"+result[0]+"_"+num_a+".jpg";
			temp[1]=lfw_data_path+"/"+result[0]+"/"+result[0]+"_"+num_b+".jpg";
			temp[2]="1";
			lfw_data_list.push_back(temp);
		}
		else{
			vector<string> temp(4);
			string num_a=some_function(atoi(result[1].c_str()),4);
			string num_b=some_function(atoi(result[3].c_str()),4);
			temp[0]=lfw_data_path+"/"+result[0]+"/"+result[0]+"_"+num_a+".jpg";
			temp[1]=lfw_data_path+"/"+result[2]+"/"+result[2]+"_"+num_b+".jpg";
			temp[2]="0";
			lfw_data_list.push_back(temp);
		}
		*/
	}
}

int main(int argc, const char *argv[]){
	
	std::string pair_path = std::string(argv[1]);//query/  db/
	std::string lfw_data_path = std::string(argv[2]);//1k_test_data/align
	std::string model_path = std::string(argv[3]);
	std::string model_align_path = std::string(argv[4]);
	float threshold = atof(argv[5]);
	float tp=0.0;
	float fp=0.0;
	float tn=0.0;
	int p=0;
	int num=0;
	float accuracy;

	std::vector<std::vector<string>> lfw_data_list;
	load_lfw_data_path(pair_path, lfw_data_path, lfw_data_list);
    dlib::shape_predictor sp;
	load_align(model_align_path,sp);
	load_face_detector(model_path);
	load_face_embedding(model_path);
    //string img ="/home/ckp/data/db/Angela_Bassett_0005.jpg";
	//string db_dir=pair_path+"/*.jpg";
    //vector<string> db_path=glob(db_dir);
	//cout<<lfw_data_list.size()<<endl;
	ofstream infile;
	string write_file ="testdata.txt";
	infile.open(write_file,ios::trunc);
	int ndt=0;
	for(int i=0;i<lfw_data_list.size();++i){
		cout<<"pair"<<i<<endl;
		string img_a=lfw_data_list[i][0];;
		string img_b=lfw_data_list[i][1];
		cout<<img_a<<endl;
		cout<<img_b<<endl;
		//cout<<img_b<<endl;
		string match=lfw_data_list[i][2];
		//cout<<match<<endl;
		Mat image_a=cv::imread(img_a,-1);
		//cout<<image_a.empty()<<endl;
		//imshow("1",image_a);
		//waitKey(0);
		Mat image_b=cv::imread(img_b,-1);
		if(image_a.empty()){
			cout<<img_a<<" read error"<<endl;
			continue;
		}
		else if(image_b.empty()){
			cout<<img_b<<" read error"<<endl;
			continue;
		}
		//Mat align_img_a;
		//Mat align_img_b;
		//cout<<"p3"<<endl;
		int width_std = 160;
		if ( image_a.cols > width_std )
		{
			//cv::resize(image_a,image_a,cv::Size(width_std, (int)((float)(image_a.rows*width_std/image_a.cols))));	
		}
		if ( image_b.cols > width_std )
		{
			//cv::resize(image_b,image_b,cv::Size(width_std, (int)((float)(image_b.rows*width_std/image_b.cols))));	
		}
		//cv::resize(image_a, image,cv::Size(160,160));
		BoundingBox face_a;
		BoundingBox face_b;
		double t=(double)cv::getTickCount();
		bool a = detect_one_face(image_a, face_a); 	
		t=((double)cv::getTickCount()-t)/cv::getTickFrequency();
		cout<<"detect time:"<<t<<endl;
		bool b = detect_one_face(image_b, face_b);
		if(!a || !b){
			continue;
		}
		//cout<<face_a.x1<<","<<face_a.y1<<endl;
		//cout<<face_b<<endl;
		std::vector<cv::Mat> align_imgs_a;
		std::vector<cv::Mat> align_imgs_b;
		std::vector<BoundingBox> faceinfos_a;
		std::vector<BoundingBox> faceinfos_b;
		faceinfos_a.push_back(face_a);
		faceinfos_b.push_back(face_b);
		//cout<<"s a"<<endl;
		
		//cv::resize(align_imgs_a[0], align_imgs_a[0],cv::Size(160,160));
		
		double t0=(double)cv::getTickCount();
		face_align(sp, image_a, faceinfos_a, align_imgs_a);
		face_align(sp, image_b, faceinfos_b, align_imgs_b);
		t0=((double)cv::getTickCount()-t0)/cv::getTickFrequency();
		cout<<"align time:"<<t0<<endl;
		if(align_imgs_a.size()==0){
			cout<<"can not detect:"<<img_a<<endl;

			continue;
		}
		else if(align_imgs_b.size()==0){
			cout<<"can not detect:"<<img_b<<endl;
	       	//ndt++;
			continue;
		}
		//cout<<align_imgs_a.size()<<":"<<align_imgs_b.size()<<endl;
		//imshow("1",align_imgs_a[0]);
		//waitKey(0);
		
		//cout<<align_img_a.empty()<<endl;
		//imshow("1",image_a);
		//waitKey(0);
		Mat float_feature_a;
		Mat float_feature_b;
		//cout<<"p3"<<endl;
		//cv::resize(align_imgs_a[0], align_imgs_a[0],cv::Size(160,160));
		//cv::resize(align_imgs_b[0], align_imgs_b[0],cv::Size(160,160));
		double t1=(double)cv::getTickCount();
		face_feature_extract(align_imgs_a[0], float_feature_a);
		t1=((double)cv::getTickCount()-t1)/cv::getTickFrequency();
		cout<<"embed time:"<<t1<<endl;
		face_feature_extract(align_imgs_b[0], float_feature_b);
		//t1=((double)cv::getTickCount()-t1)/cv::getTickFrequency();
		//cout<<"embed time:"<<t1<<endl;
		//cout<<"p4"<<endl;
		//continue;
		float cosine=face_verification(float_feature_a, float_feature_b);
		//cout<<"p5"<<endl;
		getname(img_a);
		getname(img_b);
		//cout<<"p6"<<endl;
		infile<<cosine<<" "<<match<<" "<<img_a<<" "<<img_b<<" "<<endl;
		if(match=="1"){
			p+=1;
		}
		if(cosine>threshold){
			if(match=="1"){
				tp+=1;
			}
			else{
				fp+=1;
			}
		}
		else{
			if(match=="0"){
				tn+=1;
			}
		}
		num+=1;
		
		
	}
	//cout<<"total:"<<db_path.size()<<endl;
	//cout<<"hit:"<<ndt<<endl;
		accuracy=(tp+tn)/num;
        cout<<"threshold = "<<threshold<<endl;	
		cout<<"num = "<<num<<endl;
		cout<<"p = "<<p<<endl;
		cout<<"tp = "<<tp<<endl;
		cout<<"fp = "<<fp<<endl;
		cout<<"TPR = "<<tp/float(p)<<endl;
		cout<<"FPR = "<<fp/float(num-p)<<endl;
		cout<<"accuracy = "<<accuracy<<endl;
		infile.close();

}
