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
	std::string model_path = std::string(argv[2]);
	load_filter_face(model_path);
	db_dir=db_dir+"/*.*";
    vector<string> db_path=glob(db_dir);
	ofstream infile;
	infile.open("binary_list.txt",ios::trunc);
	for(int i=0;i<db_path.size();++i){
		string image_path=db_path[i];
		cout<<image_path<<endl;
		Mat image = imread(image_path,-1);
		if(image.empty()){
			cout<<"Error:Image cannot be loaded!"<<endl;
			continue;                       
		}
		getname(image_path);
		int isface = face_filter(image);
		if(isface==1){
			infile<<image_path<<" "<<1<<endl;
		}
			
		else{
				infile<<image_path<<" "<<0<<endl;
			}
		
	}
	infile.close();

}
