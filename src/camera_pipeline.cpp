#include <lcm/lcm-cpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

#include <iostream>

#include "project_types/image_points_t.hpp"

using namespace cv;
using namespace std;
using namespace project_types;
using namespace lcm;

#define NUM_CAMS 4
#define CAMERA_DATA_PATH ""

#define CORRESPONDANCE_RADIUS 5

#define GAUSSIAN_SD 0.001
#define GAUSSIAN_KERNEL_S Size(5, 5)

Mat draw_points(Mat source, vector<Point2d> points) {
    for (int i=0; i<points.size(); i++) {
        circle(source, points[i], 1, Scalar(0, 0, 255), -1);
        putText(source, format("{%.1f, %.1f}", points[i].x, points[i].y), Point2d(points[i].x, points[i].y-10), FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 255), 1);
    }
    return source;
}

Mat draw_lines(Mat source, vector<vector<double>> lines) {
    for (int i=0; i<lines.size(); i++) {
        Point2d p0 = Point2d(0, -lines[i][2]/lines[i][1]);
        Point2d p1 = Point2d(source.cols, -source.cols*(lines[i][0]/lines[i][1]) - lines[i][2]/lines[i][1]);
        line(source, p0, p1, Scalar(0, 0, 255), 1);
    }
    return source;
}

double dist_point_line(Point2d point, vector<double> line) {
    return abs(line[0]*point.x + line[1]*point.y + line[2])/(sqrt(line[0]*line[0] + line[1]*line[1]));
}

Point2d find_corresponding_point(vector<Point2d> points, vector<double> line) {
    double min_dist = 1e9;
    //negative points are invalid (2d screen!)
    Point2d out = Point2d(-1, -1);
    for (int i=0; i<points.size(); i++) {
        double dist = dist_point_line(points[i], line);
        if ((dist < CORRESPONDANCE_RADIUS) && (dist < min_dist)) {
            min_dist = dist;
            out = points[i];
        }
    }
    return out;
}

vector<Point2d> get_points(Mat source) {
    Mat img;

    vector<Vec4i> img_hierarchy; //needed for idk what
    vector<vector<Point>> img_contours;

    //Convert to gray -> threshold -> dilate -> find points
    cvtColor(source, img, COLOR_RGB2GRAY); 
    threshold(img, img, 255*0.7, 255, THRESH_BINARY); 

    /*
    Mat element = getStructuringElement(MORPH_RECT, Size(1, 1), Point(0, 0));
    dilate(img, img, element);
    */

    findContours(img, img_contours, img_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    drawContours(source, img_contours, -1, Scalar(0, 255, 0), 1);

    //From contours -> moments -> (x, y) points
    vector<Moments> img_moments(img_contours.size());
    vector<Point2d> img_points(img_contours.size());

    for (int i=0; i<img_contours.size(); i++) {
        img_moments[i] = moments(img_contours[i]);
        if (contourArea(img_contours[i]) > 0) {
            img_points[i] = Point2d(
                //1e-5 to avoid div by 0
                static_cast<float>((img_moments[i].m10 + 1e-5)/(img_moments[i].m00 + 1e-5)),
                static_cast<float>((img_moments[i].m01 + 1e-5)/(img_moments[i].m00 + 1e-5))
            );
        }
        else {
            //If the point is really small, (area 0), find where it is.
            img_points[i] = Point2d(
                static_cast<float>(img_contours[i][0].x),
                static_cast<float>(img_contours[i][0].y)
            );
        }
    }

    return img_points;
}



vector<vector<float>> points_to_float(vector<Point2d> src) {
    vector<vector<float>> out(src.size());
    for (int i=0; i<src.size(); i++) {
        out[i] = vector<float>(2);
        out[i][0] = src[i].x;
        out[i][1] = src[i].y;
    }
    return out;
}

int main() {
    /*
    Pipeline:
    Camera feed -> Isolating Points -> Finding Corresponding Points ->
    Finding the 3D Positions of these points. 
    */

    LCM lcm;
    if (!lcm.good()) {
		return 1;
    }
    
    //This is supposed to be the camera stream 
    Mat feed = imread("img.png", IMREAD_COLOR);
    //GaussianBlur(feed, feed, GAUSSIAN_KERNEL_S, GAUSSIAN_SD, GAUSSIAN_SD, BORDER_DEFAULT);

    FileStorage fs(CAMERA_DATA_PATH, FileStorage::READ);
    //Should be the same for all cameras (unless we use diff cameras from time to time)
    Mat intrinsic_matricies[NUM_CAMS];
    //From fundamental everything else can be derived
    //For a 4 cam system {0->1, 1->2, 2->3, 3->0}
    Mat fundamental_matricies[NUM_CAMS];
    Mat essential_matricies[NUM_CAMS];
    Mat rotation_matricies[NUM_CAMS];
    Mat translation_matricies[NUM_CAMS];

    Mat feed_images[NUM_CAMS];
    Rect feed_splits[NUM_CAMS] = {Rect(1, 1, 638, 510), Rect(641, 0, 638, 510), Rect(1, 513, 638, 510), Rect(641, 513, 638, 510)};
    vector<Point2d> u_points[NUM_CAMS];
    vector<vector<double>> epipolar_lines[NUM_CAMS];
    //0->1, 1->2, 2->3, 3->0
    vector<vector<Point2d>> corresponding_points[NUM_CAMS];
    
    //Read in intrinsic matricies, fundamental matricies, find other matricies
    for (int i=0; i<NUM_CAMS; i++) {
        fs[format("cam%d_intrinsice_matrix", i)] >> intrinsic_matricies[i];
        fs[format("cam%d_fundamental_matrix", i)] >> fundamental_matricies[i];
        fs[format("cam%d_essential_matrix", i)] >> essential_matricies[i];
        fs[format("cam%d_rotation_matrix", i)] >> rotation_matricies[i];
        fs[format("cam%d_translation_matrix", i)] >> translation_matricies[i];
    }

    fs.release();

    //Split images, get points.
    for (int i=0; i<NUM_CAMS; i++) {
        feed_images[i] = Mat(feed, feed_splits[i]);
        u_points[i] = get_points(feed_images[i]);
        feed_images[i] = draw_points(feed_images[i], u_points[i]);
    }

    //Create epipolar lines + find corresponding points
    for (int i=0; i<NUM_CAMS; i++) {
        int r = (i+1)%NUM_CAMS;
        epipolar_lines[i] = vector<vector<double>>(u_points[i].size());
        for (int k=0; k<u_points[i].size(); k++) {
            //Find epipolar line equation
            Mat p_l = Mat({u_points[i][k].x, u_points[i][k].y, 1.0}).t();
            p_l = p_l * fundamental_matricies[i];
            epipolar_lines[i][k] = vector<double>{p_l.at<double>(1,1), p_l.at<double>(1,2), p_l.at<double>(1,3)};
            //Search through right image for corresponding points
            Point2d p_r = find_corresponding_point(u_points[r], epipolar_lines[i][k]);
            if (p_r.x > 0) {
                corresponding_points[i].push_back(vector<Point2d>{u_points[i][k], p_r});
            }
        }
        feed_images[r] = draw_lines(feed_images[r], epipolar_lines[i]);
    }

    

    return 0;
}
