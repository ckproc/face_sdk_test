#include "BusyThreadContainer.h"
#include "MyThread.h"

using namespace std;
BusyThreadContainer::BusyThreadContainer()
{
}
BusyThreadContainer::~BusyThreadContainer()
{
}
void BusyThreadContainer::push(MyThread *m)
{
	busy_thread_container_.push_back(m);
}

void  BusyThreadContainer::erase(MyThread *m)
{
	//busy_thread_container_.erase(find(busy_thread_container_.begin(),busy_thread_container_.end(),m));
	list<MyThread*>::iterator it;
	for (it=busy_thread_container_.begin(); it !=busy_thread_container_.end() ; it++)
	{
		if (*it == m){
			busy_thread_container_.erase(it);
            break;
		}
	}
}
std::list<MyThread*>::size_type BusyThreadContainer::size()
{
	return busy_thread_container_.size();
}
