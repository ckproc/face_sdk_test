#pragma once

namespace
{
	enum  PRIORITY
	{

		MIN = 1, NORMAL = 25, MAX = 50
	};
}

class Task
{
	
public:
	Task()
	{

	}
	void SetPriority(int priority)
	{
		if (priority>(PRIORITY::MAX))
		{
			priority = (PRIORITY::MAX);
		}
		else if (priority>(PRIORITY::MAX))
		{
			priority = (PRIORITY::MIN);
		}
	}	
	virtual void Run() = 0;
protected:
	int priority_;
};
