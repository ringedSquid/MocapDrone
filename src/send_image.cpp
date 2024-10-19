#include <lcm/lcm-cpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "image_types/image_grey_t.hpp"

using namespace cv;
using namespace lcm;
using namespace image_types;

int main(int argc, char **argv) {
	LCM lcm;
	if (!lcm.good()) {
		return 1;
	}

	std::string image_path = samples::findFile("catTest.jpg");
	Mat img_open = imread(image_path, IMREAD_GRAYSCALE);

	image_grey_t img;

	img.rows = img_open.rows;
	img.cols = img_open.cols;
	img.size = img.rows * img.cols;
	//printf("r: %d, c: %d, s: %d\n", img.rows, img.cols, img.size);
	img.data.resize(img.size);
	std::memcpy(img.data.data(), img_open.data, img.size*sizeof(uint8_t));

	lcm.publish("IMAGE", &img);
	return 0;
}