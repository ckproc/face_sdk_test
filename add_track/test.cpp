#include "detect_embed_search_verification.hpp"
#include <config4cpp/Configuration.h>
//#include "detect_embed.hpp"
#include<iostream>
#include <vector>
#include <string>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "Tracker.h"
#include <sys/time.h>
#include <time.h>
using namespace std;
using namespace cv;
using namespace config4cpp;
typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;

struct DetectionConfig{
    //int frame_persec;  //控制每秒检测帧数
	float jump_ratio;
	//int M_sec;         //连续M_sec秒； 与 N_frame互斥使用，M_sec不为0时，以M_sec为准；M_sec为0时，以N_frame为准
	int N_frame;		  //N_frame帧上报
	int min_face_size;
    DetectionConfig(){
		jump_ratio = 0.5;
		N_frame = 3;
		min_face_size = 50;
	}	//能检测到的最小目标大小（定为50像素）
};


#define CNUM 20

double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}
/*
int compare_face(cv::Mat db_embs, cv::Mat embed, int &match_idx, float &identification_score){
	vector<P> cdists;
	for(int i=0; i<db_embs.rows; ++i){
		double sumA=0;
		double sumB=0;
		double cosine = 0;
		for(int j=0; j<db_embs.cols; ++j){
			sumA+=db_embs.ptr<float>(i)[j]*db_embs.ptr<float>(i)[j];
			sumB+=embed.ptr<float>(0)[j]*embed.ptr<float>(0)[j];
			cosine+=db_embs.ptr<float>(i)[j]*embed.ptr<float>(0)[j];
		}
		sumA=sqrt(sumA);
		sumB=sqrt(sumB);
		cosine/=sumA*sumB;
		cdists.push_back(make_pair(cosine,i));
	}
	sort(cdists.begin(),cdists.end(),cmp);
	match_idx = cdists[0].second;
	identification_score = cdists[0].first;
}*/

typedef void (*PTR_IMAGE_PROCESS_FUN)(struct timeval capture_time, cv::Mat &whole_img, vector<cv::Mat> &faces, cv::Mat db_embs, 
								vector<string> db_paths,  unsigned int custom_Id);

int person =0;
void process_image(struct timeval capture_time, cv::Mat &whole_img, vector<cv::Mat> &faces,cv::Mat db_embs, 
                                 vector<string> db_paths, unsigned int custom_Id){
	cout<<"callback run"<<endl;//num i<<faces.size()<<endl;
	
	for(int i=0;i<faces.size();++i){
		// imshow("1",faces[i]);
		// waitKey(0);
		// vector<int> compression_params;
		// compression_params.push_back( CV_IMWRITE_JPEG_QUALITY );
		// compression_params.push_back( 100 );
		string path="./result/"+to_string(person)+"_"+to_string(i)+".bmp";
		cout<<path<<endl;
		//cv::imwrite(path,faces[i]);
		//cout<<w<<endl;
	}
	// cout<<"1"<<endl;
	vector<cv::Mat> embeds;
	bool isembed = face_embedding(faces, embeds);
	//cout<<"p2"<<endl;
	ofstream infile;
	infile.open("./result/match.txt",ios::app);
	for(int k=0;k<faces.size();k++){
		//vector<search_item> result = face_retrieval(embeds[k], db_embs, 1);
		//compare_face(db_embs, embeds[k], match_idx, identification_score);
		//int index = result[0].index;
		//float cosine = result[0].img_similarity;
		//cout<<"name:"<<db_paths[index]<<"	"<<"cosine_similarity:"<<cosine<<endl;
		//infile<<to_string(person)+"_"+to_string(k)<<" "<<db_paths[index]<<" "<<cosine<<endl;
	}
	infile.close();
	person++;
}

void off_video_detect(string &video_path, const DetectionConfig &detec_Config, string model_path, string db_path, 
				PTR_IMAGE_PROCESS_FUN image_Process_fun, unsigned int custom_Id, bool save_video, bool show_video){
	
	FileStorage fs;
	fs.open(db_path,FileStorage::READ);
	cv::Mat db_embs;
	fs["embed"]>>db_embs;
	vector<string> db_paths;
    fs["path"]>>db_paths;

	//Dict embs_res;
	vector<int> face_id_list;
	
    VideoCapture cap(video_path);
    if(!cap.isOpened()) {
        cerr << "video cannot be opened." << endl;
        return ;
    }
	 
	cv::VideoWriter output_cap("demo.avi", 
               CV_FOURCC('D', 'I', 'V', '3'),
               10,
               cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
               cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
			    
	if (!output_cap.isOpened())
	{
        std::cout << "!!! Output video could not be opened" << std::endl;
        return ;
	}
	 
	int total_frames = 0;
	int run_frames = 0;
	int img_W;
	int img_H;
    Mat frame;
	double fps;
	double t=0;
	char ss[10];
	
	int frame_count = 0;
	int max_age = 3;
	//int min_hits = 2;
	int min_hits = detec_Config.N_frame;
	double iouThreshold = 0.3;
	vector<KalmanTracker> trackers;// trackers one for one id
	KalmanTracker::kf_count = 0;
	vector<Rect_<float>> predictedBoxes;
	vector<vector<double>> iouMatrix;
	vector<int> assignment;
	set<int> unmatchedDetections;
	set<int> unmatchedTrajectories;
	set<int> allItems;
	set<int> matchedItems;
	vector<cv::Point> matchedPairs;
	vector<TrackingBox> frameTrackingResult;// trackers one for one box in a frame
	unsigned int trkNum = 0;
	unsigned int detNum = 0;
	//int width = 160;
	double totaltime=0.0;
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
	

	Tracker tracker;
	Mat img_prev;
	vector<Rect_<float>> rects_prev;
	vector<float> rects_score;
	bool on_tracking = false;

	load_filter_face(model_path);
	load_face_detector(model_path);
	load_face_embedding(model_path);
    for(;;) {
		//double t1=(double)getTickCount();
		
		total_frames++;
		frame_count++;
		
		
        cap >> frame;
		bool bSuccess = cap.read(frame);
		
		//double nowtime=(double)getTickCount();
		//if(total_frames%int(1.0/detec_Config.jump_ratio)==0){
		//	continue;
		//}
		//lasttime=nowtime;
		run_frames+=1;
		 if (!bSuccess) //if not success, break loop
        {
             cout << "ERROR: Cannot read a frame from video file" << endl;
             break;
        }
		
		
		
		//continue;
		if(total_frames==1){
			img_W=frame.cols;
			img_H=frame.rows;
		}

        
		if(total_frames%6==0){
			on_tracking=false;
		}
        vector< BoundingBox> faces;
        faces.clear();
        cv::Mat grayImg;
		cv::cvtColor(frame, grayImg, CV_RGB2GRAY);
		double t0=(double)cv::getTickCount();

		if (on_tracking && rects_prev.size() > 0) {
			cout<<"track"<<endl;
			vector<Rect_<float>> rects_tracked;
			vector<bool> status_tracked;
			tracker.track(img_prev, grayImg, rects_prev, rects_tracked, status_tracked);

			for (size_t i = 0; i < rects_prev.size(); i++) {
				if (status_tracked[i]) {
					BoundingBox box;
					box.x1 = rects_tracked[i].x ;
					box.y1 = rects_tracked[i].y ;
					box.x2 = (rects_tracked[i].x + rects_tracked[i].width) ;
					box.y2 = (rects_tracked[i].y + rects_tracked[i].height) ;
					box.score = rects_score[i];

					faces.push_back(box);
				}
			}

			if (faces.size() <= 0)
				//cout<<"get no track"<<endl;
				on_tracking = false;
		} else {
			bool isdetect = face_detect(frame, faces);
			cout<<"detect"<<endl;
			if(!isdetect){
				cout<<"fail detected!"<<endl;
				continue;
			}
			else{
				on_tracking = save_video;
			}
		}

		
		t0=((double)cv::getTickCount()-t0)/cv::getTickFrequency();
		//cout<<"detect time:"<<t0<<" ";
		totaltime+=t0;
		
		
		
		
		
		
		vector<Rect_<float>> detections;
		for(int i = 0; i < faces.size(); i++) {
				//Rect_<float> face_i = Rect_<float>(Point_<float>(faces[i].y1, faces[i].x1), Point_<float>(faces[i].y2, faces[i].x2));
				Rect_<float> face_i = Rect_<float>(Point_<float>(faces[i].x1, faces[i].y1), Point_<float>(faces[i].x2, faces[i].y2));
				//cv::Rect_<float> cropp;
				detections.push_back(face_i);
				
				
		}
		// initialize kalman trackers using first detections.
		
        if (trackers.size() == 0) // the first frame met
		{
			// initialize kalman trackers using first detections.
			for (unsigned int i = 0; i < detections.size(); i++)
			{
				//KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box);
				//Rect_<float> bb = Rect_<float>(Point_<float>(faces[i].x1, faces[i].y1), Point_<float>(faces[i].x2, faces[i].y2));
				KalmanTracker trk = KalmanTracker(detections[i]);
				trackers.push_back(trk);
				
			}
			continue;
		}
		// 3.1. get predicted locations from existing trackers.
		predictedBoxes.clear();
        for (auto it = trackers.begin(); it != trackers.end();)
		{
			Rect_<float> pBox = (*it).predict();
			if (pBox.x >= 0 && pBox.y >= 0)
			{
				predictedBoxes.push_back(pBox);
				it++;
			}
			else
			{
				it = trackers.erase(it);
				//cerr << "Box invalid at frame: " << frame_count << endl;
			}
		}
		// 3.2. associate detections to tracked object (both represented as bounding boxes)
		trkNum = predictedBoxes.size();
		detNum = detections.size();
		
		iouMatrix.clear();
		iouMatrix.resize(trkNum, vector<double>(detNum, 0));

		for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
		{
			for (unsigned int j = 0; j < detNum; j++)
			{
				// use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
				iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detections[j]);
			}
		}
		// solve the assignment problem using hungarian algorithm.
		HungarianAlgorithm HungAlgo;
		assignment.clear();
		HungAlgo.Solve(iouMatrix, assignment);

		// find matches, unmatched_detections and unmatched_predictions
		unmatchedTrajectories.clear();
		unmatchedDetections.clear();
		allItems.clear();
		matchedItems.clear();

		if (detNum > trkNum) //	there are unmatched detections
		{
			for (unsigned int n = 0; n < detNum; n++)
				allItems.insert(n);

			for (unsigned int i = 0; i < trkNum; ++i)
				matchedItems.insert(assignment[i]);

			set_difference(allItems.begin(), allItems.end(),
				matchedItems.begin(), matchedItems.end(),
				insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
		}
		else if (detNum < trkNum) // there are unmatched trajectory/predictions
		{
			for (unsigned int i = 0; i < trkNum; ++i)
				if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
					unmatchedTrajectories.insert(i);
		}
		else
			;
		// filter out matched with low IOU
		matchedPairs.clear();
		for (unsigned int i = 0; i < trkNum; ++i)
		{
			if (assignment[i] == -1) // pass over invalid values
				continue;
			if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
			{
				unmatchedTrajectories.insert(i);
				unmatchedDetections.insert(assignment[i]);
			}
			else
				matchedPairs.push_back(cv::Point(i, assignment[i]));
		}
		
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detections[detIdx]);
		}

		
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detections[umd]);
			//cout<<tracker.m_id<<" ";
			trackers.push_back(tracker);
		}

		frameTrackingResult.clear();
		for (auto it = trackers.begin(); it != trackers.end();)
		{
			if (((*it).m_time_since_update < 1) &&
				((*it).m_hit_streak >= min_hits || frame_count <= min_hits))
			{
				TrackingBox res;
				res.box = (*it).get_state();
				res.id = (*it).m_id + 1;
				res.frame = frame_count;
				frameTrackingResult.push_back(res);
				it++;
			}
			else
				it++;

			// remove dead tracklet
			if (it != trackers.end() && (*it).m_time_since_update > max_age)
				it = trackers.erase(it);
		}
        //Facenet
		vector<Face> faces_info;
		faces_info.clear();
		
		//double t3=(double)cv::getTickCount();
		//double ad=(t3-tt)/cv::getTickFrequency();
		//cout<<"time:"<<ad<<"		";
		//bool report;
		vector<int> report_id;
		vector<cv::Mat> report_faces;
		vector<int> temp_id;
		cv::Rect_<float> img_rect(0.0, 0.0, float(img_W), float(img_H));
		for(int i = 0; i < frameTrackingResult.size(); i++) { 
			//Rect_<float> face_i = frameTrackingResult[i].box;
			Rect_<float> face_i; //= frameTrackingResult[i].box;
			face_i.x = frameTrackingResult[i].box.x;
			face_i.y = frameTrackingResult[i].box.y;
			face_i.width = frameTrackingResult[i].box.width;
			face_i.height = frameTrackingResult[i].box.height;
			int faceid = frameTrackingResult[i].id;
			//Mat cropface=frame(face_i&img_rect);
				temp_id.push_back(faceid);
				float detection_score = faces[i].score;
				//if(detection_score<0.9)
				//	continue;
				Face t_face(face_i, faceid, detection_score);
				t_face.crop_face(frame, face_i,img_W,img_H);
				faces_info.push_back(t_face);
				//bool id_exists = (std::find(face_id_list.begin(), face_id_list.end(), faceid) != face_id_list.end());
				//if(!id_exists){
					
					//string path="/home/ckp/zz/"+to_string(faceid)+".bmp";
					//cout<<path<<endl;
					//cv::imwrite(path,t_face.aligned_face);
					//faces_info.push_back(t_face);
					//int isface = face_filter(t_face.aligned_face);
					//if(isface==1){
						//face_id_list.push_back(faceid);
						//report_id.push_back(faceid);
						//report_faces.push_back(t_face.aligned_face);
					//}
					//else{
					//	cout<<"reject"<<faceid<<" "<<isface<<endl;
					//}
				//}
		}
		
		
		//for(int i=0;i<face_id_list.size();++i){
			
				//vector<int>::iterator   iter   =   vec.begin()+5;
				//vec.erase(iter);
				//bool id_exists = (std::find(temp_id.begin(), temp_id.end(), face_id_list[i]) != temp_id.end());
				//if(!id_exists){
				//	face_id_list.erase(face_id_list.begin()+i);
				//}
				
				//report_faces.erase(report_faces.begin()+i);
				//report_id.erase(report_id.begin()+i);
				//temp_id.erase(temp_id.begin()+i);
			
			
		//}

		//if(report_faces.size()>0){
		//	frame_count++;
		//	for(int i=0;i<report_id.size();++i){
			//	cout<<report_id[i]<<" ";
			//}
			//cout<<endl;
			//struct timeval tv;
			//gettimeofday(&tv, NULL);
			//image_Process_fun(tv, frame, report_faces, db_embs, db_paths, custom_Id);//CALLBACk Function
		//}

		
		//double t4=(double)cv::getTickCount();
		//double rd=(t4-t1)/cv::getTickFrequency();
		//fps=1.0/rd;
		//cout<<"sort:"<<bd<<" "
		//cout<<"fps:"<<fps<<endl;
		img_prev = grayImg;
		rects_prev.clear();
		for(int i=0; i<faces_info.size(); i++){
			rects_prev.push_back(faces_info[i].Bbox);
			rects_score.push_back(faces_info[i].Detection_score);
		}

		double t1=(double)cv::getTickCount();
		for(int i=0; i<faces_info.size(); i++){
			Face temp_face = faces_info[i];
			Rect_<float> face_i = temp_face.Bbox;
			int prediction = temp_face.Faceid;
			
			rectangle(frame, face_i, randColor[prediction % CNUM] , 1);
			string box_text = format("%d",prediction);//format("id = %d", prediction);
			//format("%.4f",temp_face.Detection_score)+format("%.4f",identification_score)
			int pos_x = std::max(face_i.tl().x - 10, float(0));
			int pos_y = std::max(face_i.tl().y - 10, float(0));
			putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, randColor[prediction % CNUM], 2.0);
			
			
		
		}
		
        
		
		
		sprintf(ss,"%.2f",fps);
		std::string fpsString("FPS:");
		fpsString += ss;
		putText(frame, fpsString, cv::Point(5, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255,0), 2.0);
		
		if(save_video){
			//output_cap.write(frame);
		}
		if(show_video){
		//int width1 = 480;
		//float ratio1 = win.cols/float(width1);
		//int height1 = (int)(float(win.rows/ratio1));
		//
		//resize(win, win , Size(width1,height1));
		//cout<<"p1"<<endl;
        imshow("face_recognizer", frame);
		
		//cout<<"p2"<<endl;
        char key = (char) waitKey(1);
		t1=((double)cv::getTickCount()-t1)/cv::getTickFrequency();		
		//cout<<"plot time:"<<t1<<endl;
        if(key == 27)
            break;
		}
    }
	//cout<<endl;
	//cout<<total_frames<<":"<<run_frames<<":"<<frame_count<<endl;
	cout<<"avgtime:"<<totaltime/double(run_frames)<<endl;
	cout<<"fps:"<<double(run_frames)/totaltime<<endl;
}

int main(int argc, const char* argv[])
{
	//int video_number=std::stoi((std::string(argv[1])));
	const char *configFile = argv[1];
	//vector<run_video_param> params;
	string video_file,model_path,db_path;
	bool save_video,show_video;
	unsigned int custom_Id;
	//cout<<"p1"<<endl;
	Configuration *cfg = Configuration::create();
	const char *scope ="";
	//cout<<"p2"<<endl;
	//cout<<configFile<<endl;
	
	
	DetectionConfig detec_Config;
	//cout<<"start"<<endl;
	try{
		//run_video_param par;
		//string v_path = "video_"+to_string(i);
		//cout<<"start"<<endl;
		cfg->parse(configFile);
		video_file=string(cfg->lookupString(scope,"video_name"));
		model_path=string(cfg->lookupString(scope,"model_path"));
		db_path=string(cfg->lookupString(scope,"db_path"));
		save_video=cfg->lookupBoolean(scope, "save_video");
	    show_video=cfg->lookupBoolean(scope, "show_video");
		detec_Config.jump_ratio=cfg->lookupFloat(scope, "jump_ratio");
		detec_Config.N_frame=cfg->lookupInt(scope, "N_frame");
		detec_Config.min_face_size=cfg->lookupInt(scope, "min_face_size");
		custom_Id=cfg->lookupInt(scope, "Id");
		//cout<<"start"<<endl;
	}
	//cout<<"start"<<endl;
	catch(const ConfigurationException & ex) {
	cerr << ex.c_str() << endl;
	cfg->destroy();
	}
	//cout<<"start"<<endl;
	PTR_IMAGE_PROCESS_FUN image_Process_fun;
	image_Process_fun = process_image;
	off_video_detect(video_file, detec_Config, model_path, db_path, image_Process_fun, custom_Id, save_video, show_video);
	

}