#include "IdleThreadContainer.h"
#include "MyThread.h"
#include <iostream>
using namespace std;
IdleThreadContainer::IdleThreadContainer()
{
}


IdleThreadContainer::~IdleThreadContainer()
{
	int i = 0;
	for (Iterator it = idle_thread_container_.begin(); it != idle_thread_container_.end(); it++)
	{
		cout << i++ << endl;
		delete *it;
	}
}

std::vector<MyThread*>::size_type IdleThreadContainer::size()
{
	return idle_thread_container_.size();
}
void IdleThreadContainer::push(MyThread *m)
{
	idle_thread_container_.push_back(m);
}
void IdleThreadContainer::pop()
{
	idle_thread_container_.pop_back();
}
void IdleThreadContainer::erase(MyThread *m)
{
	//idle_thread_container_.erase(find(idle_thread_container_.begin(), idle_thread_container_.end(), m));
	vector<MyThread*>::iterator it;
	for (it = idle_thread_container_.begin(); it != idle_thread_container_.end(); it++)
	{
		if (*it == m){
			idle_thread_container_.erase(it);
            break;
		}
	}
}
void IdleThreadContainer::assign(int number, MyThreadPool *m)
{
	for (int i = 0; i < number; i++)
	{
		MyThread *p = new MyThread(m);
		idle_thread_container_.push_back(p);
	}
}
MyThread* IdleThreadContainer::top()
{
	
	return idle_thread_container_.back();
}