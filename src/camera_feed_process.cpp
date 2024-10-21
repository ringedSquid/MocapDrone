#include <lcm/lcm-cpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

#include "project_types/image_points_t.hpp"

using namespace cv;
using namespace std;
using namespace project_types;
using namespace lcm;


#define GAUSSIAN_SD 0.001
#define GAUSSIAN_KERNEL_S Size(5, 5)

vector<Point2f> get_points(Mat source) {
    Mat img;

    vector<Vec4i> img_hierarchy; //needed for idk what
    vector<vector<Point>> img_contours;

    //Convert to gray -> threshold -> dilate -> find points
    cvtColor(source, img, COLOR_RGB2GRAY); 
    threshold(img, img, 255*0.6, 255, THRESH_BINARY); 

    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1), Point(0, 0));

    dilate(img, img, element);
    findContours(img, img_contours, img_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    //From contours -> moments -> (x, y) points
    vector<Moments> img_moments(img_contours.size());
    vector<Point2f> img_points(img_contours.size());

    for (int i=0; i<img_contours.size(); i++) {
        img_moments[i] = moments(img_contours[i]);
        img_points[i] = Point2f(
            //1e-5 to avoid div by 0
            static_cast<float>(img_moments[i].m10/(img_moments[i].m00 + 1e-5)),
            static_cast<float>(img_moments[i].m01/(img_moments[i].m00 + 1e-5))
        );
    }

    return img_points;
}

vector<vector<float>> points_to_float(vector<Point2f> src) {
    vector<vector<float>> out(src.size());
    for (int i=0; i<src.size(); i++) {
        out[i] = vector<float>(2);
        out[i][0] = src[i].x;
        out[i][1] = src[i].y;
    }
    return out;
}

int main() {

    LCM lcm;
    if (!lcm.good()) {
		return 1;
	}

    //This is supposed to be the camera stream 
    Mat feed = imread("img.png", IMREAD_COLOR);
    GaussianBlur(feed, feed, GAUSSIAN_KERNEL_S, GAUSSIAN_SD, GAUSSIAN_SD, BORDER_DEFAULT);

    //imshow("img", feed);
    //waitKey(0);

    image_points_t out_points;

    Mat image0 = Mat(feed, Rect(1, 1, 638, 510));
    Mat image1 = Mat(feed, Rect(641, 0, 638, 510));
    Mat image2 = Mat(feed, Rect(1, 513, 638, 510));
    Mat image3 = Mat(feed, Rect(641, 513, 638, 510));

    vector<Point2f> points0 = get_points(image0);
    vector<Point2f> points1 = get_points(image1);
    vector<Point2f> points2 = get_points(image2);
    vector<Point2f> points3 = get_points(image3);

    out_points.cam0_n = points0.size();
    out_points.cam1_n = points1.size();
    out_points.cam2_n = points2.size();
    out_points.cam3_n = points3.size();

    out_points.cam0_points = points_to_float(points0);
    out_points.cam1_points = points_to_float(points1);
    out_points.cam2_points = points_to_float(points2);
    out_points.cam3_points = points_to_float(points3);
    
    
    lcm.publish("POINTS", &out_points);

    return 0;
}