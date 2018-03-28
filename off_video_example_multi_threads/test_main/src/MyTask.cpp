#include "MyTask.h"
#include <thread>
#include "off_video_face_recognition.hpp"
MyTask::MyTask()
{
}


MyTask::~MyTask()
{
}
void MyTask::setdata(run_video_param param)
{
	port = param.port;
	video = param.video_name;
	model = param.model_path;
	db = param.db_path;
	save = param.save_video;
	show = param.show_video;
}
void MyTask::Run()
{
	
	std::cout<<"start to track video"<<std::endl;
	int is_track = off_video_face_recognition(port, video, model, db, save, show);
	if(is_track==-1){
		std::cout<<"run tracking failed"<<std::endl;
		return ;
	}
	std::cout<<"tracking done"<<std::endl;
	
	std::this_thread::sleep_for(std::chrono::seconds(1));
}