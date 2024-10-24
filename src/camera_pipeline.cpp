#include <lcm/lcm-cpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <math.h>

#include <iostream>

#include "project_types/corresponding_points_t.hpp"
#include "project_types/point2d_pairs_t.hpp"
#include "project_types/point2d_t.hpp"

using namespace cv;
using namespace std;
using namespace project_types;
using namespace lcm;

#define NUM_CAMS 4
#define CAMERA_DATA_PATH "../data/data.json"
#define CHANNEL_NAME "CORRESPONDING_POINTS"

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
        //printf("%lf, %lf\n", img_points[i].x, img_points[i].y);
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
    printf("lcm\n");
    VideoCapture cap(0);
    printf("cam\n");
    //namedWindow("pooP");
    if (!cap.isOpened()){ //This section prompt an error message if no video stream is found//
      cout << "No video stream detected" << endl;
      system("pause");
      return -1;
   }
    printf("success\n");
    //This is supposed to be the camera stream 
    //Mat feed = imread("img.png", IMREAD_COLOR);
    Mat feed;
    //GaussianBlur(feed, feed, GAUSSIAN_KERNEL_S, GAUSSIAN_SD, GAUSSIAN_SD, BORDER_DEFAULT);
    
    //FileStorage fs(CAMERA_DATA_PATH, FileStorage::READ);

    //Should be the same for all cameras (unless we use diff cameras from time to time)
    Mat intrinsic_matricies[NUM_CAMS];
    //From fundamental everything else can be derived
    //For a 4 cam system {0->1, 1->2, 2->3, 3->0}
    Mat fundamental_matricies[NUM_CAMS];
    Mat essential_matricies[NUM_CAMS];
    Mat rotation_matricies[NUM_CAMS];
    Mat translation_matricies[NUM_CAMS];
    
    /*
    //Read in intrinsic matricies, fundamental matricies, find other matricies
    for (int i=0; i<NUM_CAMS; i++) {
        fs[format("cam%d_intrinsic_matrix", i)] >> intrinsic_matricies[i];
        fs[format("cam%d_fundamental_matrix", i)] >> fundamental_matricies[i];
        fs[format("cam%d_essential_matrix", i)] >> essential_matricies[i];
        fs[format("cam%d_rotation_matrix", i)] >> rotation_matricies[i];
        fs[format("cam%d_translation_matrix", i)] >> translation_matricies[i];
    }
    */

    //fs.release();

    //Flags
    bool CALIBRATE_CAPTURE_POINTS = true;
    bool STREAM_FEED = true;
    bool DRAW_EPIPOLAR_LINES = false;
    bool DRAW_POINTS = true;

    while (true) {
        Mat feed_images[NUM_CAMS];
        Rect feed_splits[NUM_CAMS] = {Rect(1, 1, 958, 538), Rect(961, 1, 958, 538), Rect(1, 541, 958, 538), Rect(961, 541, 958, 538)};
        vector<Point2d> u_points[NUM_CAMS];
        vector<vector<double>> epipolar_lines[NUM_CAMS];
        //0->1, 1->2, 2->3, 3->0
        vector<vector<Point2d>> corresponding_points[NUM_CAMS];

        //Output for lcm
        corresponding_points_t out;
        out.n = NUM_CAMS;
        out.pairs.resize(out.n);

        //Read in frame
        cap >> feed;
        printf("cool\n");
        printf("%d, %d\n", feed.rows, feed.cols);

        if (feed.empty()) {
            break;
        }

        //Split images, get points.
        for (int i=0; i<NUM_CAMS; i++) {
            feed_images[i] = Mat(feed, feed_splits[i]);
            u_points[i] = get_points(feed_images[i]);
            if (DRAW_POINTS) feed_images[i] = draw_points(feed_images[i], u_points[i]);
        }
        printf("split\n");

        //Create epipolar lines + find corresponding points
        for (int i=0; i<NUM_CAMS; i++) {
            printf("size: %d\n", u_points[i].size());
            int r = (i+1)%NUM_CAMS;
            if ((u_points[i].size() == 0) || (u_points[r].size() == 0)) {
                printf("wow\n");
                continue;
            }
            epipolar_lines[i] = vector<vector<double>>(u_points[i].size());
            printf("init\n");
            for (int k=0; k<u_points[i].size(); k++) {
                Point2d p_r;
                if (CALIBRATE_CAPTURE_POINTS) p_r = u_points[r][0]; 
                else {
                    //Find epipolar line equation
                    Mat p_l = Mat({u_points[i][k].x, u_points[i][k].y, 1.0}).t();
                    p_l = p_l * fundamental_matricies[i];
                    epipolar_lines[i][k] = vector<double>{p_l.at<double>(1,1), p_l.at<double>(1,2), p_l.at<double>(1,3)};
                    //Search through right image for corresponding points
                    p_r = find_corresponding_point(u_points[r], epipolar_lines[i][k]);
                }
                printf("cap\n");

                //Add pair of corresponding points
                if (p_r.x > 0) corresponding_points[i].push_back(vector<Point2d>{u_points[i][k], p_r});
                
            }
            if (DRAW_EPIPOLAR_LINES) feed_images[r] = draw_lines(feed_images[r], epipolar_lines[i]);
        }

        printf("epipole\n");

        //Write to output datatype
        for (int i=0; i<NUM_CAMS; i++) {
            out.pairs[i].n = corresponding_points[i].size();
            out.pairs[i].points.resize(out.pairs[i].n);
            //Pairs of points
            for (int k=0; k<corresponding_points[i].size(); k++) {
                printf("poop2 %d\n", corresponding_points[i][k].size());
                if (corresponding_points[i].size() >= 2) {
                    printf("p1 %d, %d\n", out.pairs[i].n, corresponding_points[i].size());
                    out.pairs[i].points[k][0].x = corresponding_points[i][k][0].x;
                    out.pairs[i].points[k][0].y = corresponding_points[i][k][0].y;
                    printf("p2\n");
                    out.pairs[i].points[k][1].x = corresponding_points[i][k][1].x;
                    out.pairs[i].points[k][1].y = corresponding_points[i][k][1].y;
                }
                printf("poop4\n");
            }
            printf("poop3\n");
        }

        imshow("", feed);
        char c = (char)waitKey(5);//Allowing 25 milliseconds frame processing time and initiating break condition//
        if (c == 27){ //If 'Esc' is entered break the loop//
            break;
        }

        lcm.publish(CHANNEL_NAME, &out);
    }

    cap.release();
    return 0;
}
