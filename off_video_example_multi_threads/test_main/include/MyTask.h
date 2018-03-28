#pragma once
#include "Task.h"
#include <string>
#include <iostream>
struct run_video_param{
	int port;
	std::string video_name;
	std::string model_path;
	std::string db_path;
	bool save_video;
	bool show_video;
};

class MyTask :public Task
{
	friend bool operator<(MyTask  &lv,MyTask &rv)
	{
		return lv.priority_ < rv.priority_;
	}
public:
	MyTask();
	~MyTask();
	virtual void Run();
	void setdata(int d, run_video_param param);
private:
	int port;
	std::string video;
	std::string model;
	std::string db;
	int save;
	int show;
};

