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
#include "face_recognition.hpp"
#include "Hungarian.h"
#include "KalmanTracker.h"
#include "off_video_face_recognition.hpp"
//#include <omp.h>
using namespace cv;
using namespace std;

typedef struct TrackingBox
{
	int frame;
	int id;
	Rect_<float> box;
}TrackingBox;
typedef std::map<int, std::pair<string, float>> Dict;
typedef Dict::const_iterator It;
typedef pair<float, int> P;

bool cmp(const P &a, const P &b){
	return a.first > b.first;
}
#define CNUM 20
// Computes IOU between two bounding boxes
double GetIOU(Rect_<float> bb_test, Rect_<float> bb_gt)
{
	float in = (bb_test & bb_gt).area();
	float un = bb_test.area() + bb_gt.area() - in;

	if (un < DBL_EPSILON)
		return 0;

	return (double)(in / un);
}

//int total_frames = 0;
//int img_W;
//int img_H;
//double total_time = 0.0;
//vector<TrackingBox> sort(){}
//double cosine_simi(float )

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
}
int off_video_face_recognition(int port, std::string &video_file, std::string &model_path, std::string &db_path, int save_video, int show_video) 
{
	
	FileStorage fs;
	fs.open(db_path,FileStorage::READ);
	cv::Mat db_embs;
	fs["embed"]>>db_embs;
	vector<string> db_paths;
    fs["path"]>>db_paths;

	Dict embs_res;
	//vector<string> face_list;
	string outputname=video_file;
	getname(outputname);
    VideoCapture cap(video_file);
    if(!cap.isOpened()) {
        //cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        cerr << "video cannot be opened." << endl;
        return -1;
    }
	//cout<<video_file<<endl;
	//cout<<"p1"<<endl;
	string output=outputname+"_output"+".avi";
	cv::VideoWriter output_cap(output, 
               CV_FOURCC('D', 'I', 'V', '3'),
               15,
               cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH),
               cap.get(CV_CAP_PROP_FRAME_HEIGHT)));
			   //cout<<"p2"<<endl;
	if (!output_cap.isOpened())
	{
        std::cout << "!!! Output video could not be opened" << std::endl;
        return -1;
	}
	//cout<<"p3"<<endl;
    // Holds the current frame from the Video device:
    Mat frame;
	double fps;
	double t=0;
	char ss[10];
	
	int total_frames = 0;
	int img_W;
	int img_H;
	int frame_count = 0;
	int max_age = 3;
	int min_hits = 3;
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

	//double cycle_time = 0.0;
	//int64 start_time = 0;
	int width = 160;
	double totaltime=0.0;
	RNG rng(0xFFFFFFFF);
	Scalar_<int> randColor[CNUM];
	for (int i = 0; i < CNUM; i++)
		rng.fill(randColor[i], RNG::UNIFORM, 0, 256);
	
	//cout<<"p1"<<endl;
	//cout<<model_path<<endl;
	load_face_detector(model_path, port);
	load_face_embedding(model_path, port);
	//cout<<"p2"<<endl;
	ofstream infile;
	string write_file = to_string(port)+".txt";
	infile.open(write_file,ios::trunc);
	//#pragma omp parallel for
    for(;;) {
		total_frames++;
		
		frame_count++;
		
		//cout<< "p3"<<endl;
        cap >> frame;
		//cout<<"p4"<<endl;
		bool bSuccess = cap.read(frame);
		
		
		 if (!bSuccess) //if not success, break loop
        {
             cout << "Cannot read a frame from video file" << endl;
             break;
        }
		
		
	
		if(total_frames==1){
			img_W=frame.cols;
			img_H=frame.rows;
		}

		//double uu=((double)getTickCount()-t1)/cv::getTickFrequency();
		//cout<<"uu"<<uu-ju<<"	";
		//cout<<"r:"<<frame.rows<<endl;
		//cout<<"c:"<<frame.cols<<endl;
		
        Mat original = frame.clone();
		//cout<<"r:"<<original.rows<<endl;
		//cout<<"c:"<<original.cols<<endl;
		//Mat win=original.clone();
		
		//double copy=((double)getTickCount()-t1)/cv::getTickFrequency();
		//cout<<"cp"<<copy<<"	";

			
		float ratio = original.cols/float(width);
		//float ratio = img_W/float(width);
		int height = (int)(float(original.rows/ratio));
		resize(original, original , Size(width,height));	
		//cout<<"p1"<<endl;
		//double re =((double)getTickCount()-t1)/cv::getTickFrequency();
		
		//cout<<"re"<<(re-copy)<<"	";
		double t1=(double)getTickCount();
        vector< BoundingBox> faces;
		
		int isdetect = face_detect(original, faces, port);
		//cout<<"p2"<<endl;
		//double t2=(double)cv::getTickCount();
		//double mt=(t2-t1)/cv::getTickFrequency();
		//cout<<"mtcnn_time"<<mt<<"	";
		vector<Rect_<float>> detections;
		for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect_<float> face_i = Rect_<float>(Point_<float>(faces[i].y1, faces[i].x1), Point_<float>(faces[i].y2, faces[i].x2));
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
		//double tt=(double)cv::getTickCount();
		//double mtt=(tt-t2)/cv::getTickFrequency();
		//cout<<"time:"<<mtt<<"		";
		// 3.3. updating trackers

		// update matched trackers with assigned detections.
		// each prediction is corresponding to a tracker
		int detIdx, trkIdx;
		for (unsigned int i = 0; i < matchedPairs.size(); i++)
		{
			trkIdx = matchedPairs[i].x;
			detIdx = matchedPairs[i].y;
			trackers[trkIdx].update(detections[detIdx]);
		}

		// create and initialise new trackers for unmatched detections
		for (auto umd : unmatchedDetections)
		{
			KalmanTracker tracker = KalmanTracker(detections[umd]);
			trackers.push_back(tracker);
		}

		// get trackers' output
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

		for(int i = 0; i < frameTrackingResult.size(); i++) { 
			
			//Rect_<float> face_i = frameTrackingResult[i].box;
			Rect_<float> face_i; //= frameTrackingResult[i].box;
			face_i.x = frameTrackingResult[i].box.x*ratio;
			face_i.y = frameTrackingResult[i].box.y*ratio;
			face_i.width = frameTrackingResult[i].box.width*ratio;
			face_i.height = frameTrackingResult[i].box.height*ratio;
			int faceid = frameTrackingResult[i].id;
			float detection_score = faces[i].score;
			if(detection_score<0.9)
				continue;
			Face t_face(face_i, faceid, detection_score);
			//cout<<"p3"<<endl;
			t_face.crop_face(frame, face_i,img_W,img_H);
			//cout<<"p4"<<endl;
			faces_info.push_back(t_face);
			//Mat crop = win(face_i);
			//cv::resize(crop, crop,cv::Size(160,160));
		    //Mat embed = Mat(1, 128, CV_32FC1, Scalar::all(0.0));
		    //facenet(crop, embed, model_path)
			//cout<<faces_info[i].Faceid<<endl;
		}
		double t2=(double)cv::getTickCount();
		totaltime+=(t2-t1)/cv::getTickFrequency();
		
		if(total_frames%5==1){
			if(faces_info.size()==0)
				continue;
			vector<cv::Mat> aligned_faces;
			for(int j=0;j<faces_info.size();j++){
				aligned_faces.push_back(faces_info[j].aligned_face);
				//cv::imshow("1",faces_info[j].aligned_face);
				//waitKey(0);
			}
			vector<cv::Mat> embeds;
			//Mat embed = Mat(1, 128, CV_32FC1, Scalar::all(0.0));
			//cout<<"p1"<<endl;
      //cout<<"p1"<<endl;
			face_embedding(aligned_faces, embeds, port);
      //cout<<"p2"<<endl;
			//cout<<"p2"<<endl;
			embs_res.clear();
			for(int k=0;k<faces_info.size();k++){
				int match_idx;
				float identification_score;
				compare_face(db_embs, embeds[k], match_idx, identification_score);
				//cout<<match_idx<<"-"<<k<<":"<<embeds[k].ptr<float>(0)[0]<<","<<embeds[k].ptr<float>(0)[1]<<","<<embeds[k].ptr<float>(0)[2]<<","<<embeds[k].ptr<float>(0)[3]<<endl;
				embs_res[faces_info[k].Faceid] = std::make_pair(db_paths[match_idx], identification_score);
			}
			
			
		}
		
		double t4=(double)cv::getTickCount();
		double rd=(t4-t1)/cv::getTickFrequency();
		fps=1.0/rd;
		//cout<<"sort:"<<bd<<" "
		cout<<"fps:"<<fps<<endl;
		infile<<fps<<endl;
		
		
		for(int i=0; i<faces_info.size(); i++){
			Face temp_face = faces_info[i];
			bool key_exists = embs_res.find(temp_face.Faceid) == embs_res.end();
			if(key_exists)
				continue;
			string match_face = embs_res[temp_face.Faceid].first;
			
			float identification_score = embs_res[temp_face.Faceid].second;
			if(identification_score<0.5){
				Rect_<float> face_i = temp_face.Bbox;
				int prediction = temp_face.Faceid;
				rectangle(frame, face_i, randColor[prediction % CNUM] , 1);
				//string dscore,iscore;
				//gcvt(temp_face.Detection_score,6,dscore.c_str());
				//gcvt(identification_score,6,iscore.c_str());
				string box_text = std::string("unknow")+"("+format("%.4f",temp_face.Detection_score)+","+format("%.4f",identification_score)+")";//format("id = %d", prediction);
				int pos_x = std::max(face_i.tl().x - 10, float(0));
				int pos_y = std::max(face_i.tl().y - 10, float(0));
				putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, randColor[prediction % CNUM], 2.0);
				continue;
			}
			//bool face_exists = (std::find(face_list.begin(), face_list.end(), match_face) != face_list.end());
			//if(!face_exists)
			//	face_list.push_back(match_face);
		
			getname(match_face);
			
			
			Rect_<float> face_i = temp_face.Bbox;
			int prediction = temp_face.Faceid;
			rectangle(frame, face_i, randColor[prediction % CNUM] , 1);
			string box_text = match_face+"("+format("%.4f",temp_face.Detection_score)+","+format("%.4f",identification_score)+")";//format("id = %d", prediction);
			int pos_x = std::max(face_i.tl().x - 10, float(0));
			int pos_y = std::max(face_i.tl().y - 10, float(0));
			putText(frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, randColor[prediction % CNUM], 2.0);
			
			
		
		}

		
		
		sprintf(ss,"%.2f",fps);
		std::string fpsString("FPS:");
		fpsString += ss;
		putText(frame, fpsString, cv::Point(5, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0, 255,0), 2.0);
		//double t5=(double)cv::getTickCount();
		//double cd=(t5-t4)/cv::getTickFrequency();
		
		//cout<<"time3:"<<cd<<endl;
		if(save_video){
			output_cap.write(frame);
		}
		if(show_video){
		//int width1 = 480;
		//float ratio1 = win.cols/float(width1);
		//int height1 = (int)(float(win.rows/ratio1));
		//
		//resize(win, win , Size(width1,height1));
		
        imshow("face_recognizer", frame);
        char key = (char) waitKey(1);
        if(key == 27)
            break;
		}
    }
	
	//cout<<endl;
	cout<<"totaltime:"<<totaltime<<endl;
	infile<<"totaltime:"<<totaltime<<endl;
	infile.close();
	return 0;
}

