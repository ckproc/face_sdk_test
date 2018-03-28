#include "off_video_face_recognition.hpp"
#include "MyThreadPool.h"
#include "MyTask.h"
#include <config4cpp/Configuration.h>
#include<iostream>
#include <vector>
#include <string>
#include <thread>
#include <stdio.h>

//#include <thread>
//#include <functional>
using namespace std;
using namespace config4cpp;
int main(int argc, const char *argv[]){
	
	//std::string video_name = std::string(argv[1]);
	//std::string model_path = std::string(argv[2]);
	//std::string db_path = std::string(argv[3]);
	//int save_video = std::stoi((std::string(argv[4])));
	//int show_video = std::stoi((std::string(argv[5])));
	int video_number=std::stoi((std::string(argv[1])));
	const char *configFile = argv[2];
	vector<run_video_param> params;
	//run_video_param param;
	
	Configuration *cfg = Configuration::create();

		cfg->parse(configFile);
		for(int i=0;i<video_number;i++){
			try{
			run_video_param par;
			string v_path = "video_"+to_string(i);
			const char *scope =v_path.c_str();
			par.port=cfg->lookupInt(scope, "port");
			par.video_name=string(cfg->lookupString(scope,"video_name"));
			par.model_path=string(cfg->lookupString(scope,"model_path"));
			par.db_path=string(cfg->lookupString(scope,"db_path"));
			par.save_video=cfg->lookupBoolean(scope, "save_video");
			par.show_video=cfg->lookupBoolean(scope, "show_video");
			params.push_back(par);
			}
			catch(const ConfigurationException & ex) {
			cerr << ex.c_str() << endl;
			cfg->destroy();
			}
		}
	
	MyThreadPool mythreadPool(video_number);
	MyTask read_video[video_number];
	for(int i=0;i<video_number;++i){
		read_video[i].setdata(params[i]);
	}
	for(int i=0;i<video_number;++i){
		mythreadPool.AddTask(&read_video[i],20);
	}
	 while(getchar()){
		mythreadPool.EndMyThreadPool();
	}
	//std::cout<<"start to track video"<<std::endl;
	//std::thread t(off_video_face_recognition,std::ref(video_path), std::ref(model_path), std::ref(db_path), save_video,show_video);
	//t.join();
	//int is_track = off_video_face_recognition(video_path, model_path, db_path, save_video,show_video);
	//if(is_track==-1){
	//	std::cout<<"run tracking failed"<<std::endl;
	//	return -1;}
	//std::cout<<"tracking done"<<std::endl;
	//system("pause");
	//getchar();
	return 0;
}
