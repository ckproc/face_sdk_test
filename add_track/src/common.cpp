#include "common.h"

using std::nth_element;

//TODO: Check for even/uneven number of elements
//The order of the elements of A is changed
float median(vector<float> & A)
{

    if (A.size() == 0)
    {
        return numeric_limits<float>::quiet_NaN();
    }

    nth_element(A.begin(), A.begin() + A.size()/2, A.end());

    return A[A.size()/2];
}

bool isPointInRect(const Rect_<float> & rect, const cv::Point2f & pt) 
{
    return pt.x > rect.x && pt.x < (rect.x + rect.width) && pt.y > rect.y && pt.y < (rect.y + rect.height);
}

void getScaledRect(const vector<Rect_<float>> & rect, vector<Rect_<float>> & rect_scaled, float scale) 
{
	for (size_t i = 0; i < rect.size(); i++) {
		float x = rect[i].x * scale;
		float y = rect[i].y * scale;
		float width = rect[i].width * scale;
		float height = rect[i].height * scale;
		rect_scaled.push_back(Rect_<float>(x, y, width, height));
	}
}