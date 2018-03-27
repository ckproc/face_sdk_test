#ifndef COMMON_H

#define COMMON_H

#include <limits>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

float median(vector<float> & A);
bool isPointInRect(const Rect_<float> & rect, const cv::Point2f & pt);
void getScaledRect(const vector<Rect_<float>> & rect, vector<Rect_<float>> & rect_scaled, float scale);
#endif /* end of include guard: COMMON_H */
