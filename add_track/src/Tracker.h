#ifndef TRACKER_H

#define TRACKER_H

#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

class Tracker
{
public:
    Tracker() : thr_fb(10), min_size(10), img_ratio(0.5) {};
    void trackPoints(const Mat im_prev, const Mat im_gray, vector<Point2f> & points_prev,
            vector<Point2f> & points_tracked, vector<unsigned char> & status);

    void track(const Mat im_prev, const Mat im_gray, const vector<Rect_<float>> & rect_prev,
            vector<Rect_<float>> & rect_tracked, vector<bool> & status_tracked);

private:
    float thr_fb;
    float img_ratio;
    int min_size;

    void estimateScaleTranslation(const vector<Point2f> & points_prev, const vector<Point2f> & points_next, float & scale, Point2f & translation);
};

#endif /* end of include guard: TRACKER_H */
