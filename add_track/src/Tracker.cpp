#include "Tracker.h"
#include "common.h"

void Tracker::trackPoints(const Mat im_prev, const Mat im_gray, vector<Point2f> & points_prev,
        vector<Point2f> & points_tracked, vector<unsigned char> & status)
{
    if (points_prev.size() > 0)
    {
        vector<float> err; //Needs to be float

        //Calculate forward optical flow for prev_location
        calcOpticalFlowPyrLK(im_prev, im_gray, points_prev, points_tracked, status, err);

        vector<Point2f> points_back;
        vector<unsigned char> status_back;
        vector<float> err_back; //Needs to be float

        //Calculate backward optical flow for prev_location
        calcOpticalFlowPyrLK(im_gray, im_prev, points_tracked, points_back, status_back, err_back);

        //Traverse vector backward so we can remove points on the fly
        for (int i = points_prev.size()-1; i >= 0; i--)
        {
            float l2norm = norm(points_back[i] - points_prev[i]);

            bool fb_err_is_large = l2norm > thr_fb;

            if (fb_err_is_large || !status[i] || !status_back[i])
            {
                //Make sure the status flag is set to 0
                status[i] = 0;
            }
        }
    }
}

void Tracker::track(const Mat im_prev, const Mat im_gray, const vector<Rect_<float>> & rect_prev,
        vector<Rect_<float>> & rect_tracked, vector<bool> & status_tracked)
{
    Mat im2_prev, im2_gray;
    resize(im_prev, im2_prev, Size(), img_ratio, img_ratio);
    resize(im_gray, im2_gray, Size(), img_ratio, img_ratio);

    vector<Rect_<float>> r2_prev;
    getScaledRect(rect_prev, r2_prev, img_ratio);

    vector<int> points_split_index;
    vector<Point2f> points_prev;
    vector<Point2f> points_tracked;
    vector<KeyPoint> keypoints;
    cv::FAST(im2_prev, keypoints, 20);

    int start = 0;
    points_split_index.clear();
    points_split_index.push_back(start);
    for (size_t j = 0; j < r2_prev.size(); j++) {
        for (size_t i = 0; i < keypoints.size(); i++) {
            cv::Point2f pt = keypoints[i].pt;
            if (isPointInRect(r2_prev[j], pt)) {
                points_prev.push_back(pt);
            }
        }
        start = points_prev.size();
        points_split_index.push_back(start);
    }

    // cout << "corners size: " << points_prev.size() << "; face num: " << points_split_index.size() - 1 << endl;

    vector<unsigned char> status;
    trackPoints(im2_prev, im2_gray, points_prev, points_tracked, status);

    for (size_t i = 0; i < points_split_index.size() - 1; i++) {
        vector<Point2f> prev_match_points;
        vector<Point2f> next_match_points;

        for (int j = points_split_index[i]; j < points_split_index[i+1]; j++) {
            if (status[j]) {
                prev_match_points.push_back(points_prev[j]);
                next_match_points.push_back(points_tracked[j]);
            }
        }

        // cout << "face id: " << i << "; points size: " << prev_match_points.size() << endl;
        
        float scale;
        Point2f translation;
        estimateScaleTranslation(prev_match_points, next_match_points, scale, translation);

        // cout << "scale: " << scale << " translation: " << translation << endl;

        float x = r2_prev[i].x * scale + translation.x;
        float y = r2_prev[i].y * scale + translation.y;
        float width = r2_prev[i].width * scale;
        float height = r2_prev[i].height * scale;
        Rect_<float> rect = Rect_<float>(x / img_ratio, y / img_ratio, width / img_ratio, height / img_ratio);
        rect_tracked.push_back(rect);

        Rect_<float> img_rect(0.0, 0.0, float(im_prev.cols), float(im_prev.rows));
        rect = rect & img_rect;

        if (prev_match_points.size() < min_size || rect.width <= 0 || rect.height <= 0) {
            status_tracked.push_back(false);
        } else {
            status_tracked.push_back(true);
        }
        
        // cout << "prev rect: " << rect_prev[i] << " next rect: " << rect << endl;
    }
}

void Tracker::estimateScaleTranslation(const vector<Point2f> & points_prev, const vector<Point2f> & points_next, float & scale, Point2f & translation) 
{
    vector<float> scales;
    size_t points_size = points_prev.size();
    scales.reserve(points_size * points_size);
    
    for (size_t i = 0; i < points_size; i++) {
        for (size_t j = 0; j < points_size; j++) {
            if (i == j) continue;

            float d1 = norm(points_prev[i] - points_prev[j]);
            float d2 = norm(points_next[i] - points_next[j]);
            float s = d2 / d1;
            scales.push_back(s);
        }

        translation += points_next[i] - points_prev[i];
        //translation += points_prev[i] - points_next[i];
    }

    //Do not use scales, transes after this point as their order is changed by median()
    scale = median(scales);
    translation *= 1.0 / points_size;
}