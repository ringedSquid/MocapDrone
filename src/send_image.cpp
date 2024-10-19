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
	std::string image_path = samples::findFile("catTest.jpg");
	Mat img = imread(image_path, IMREAD_GRAYSCALE);

	image_grey_t img;


	LCM lcm;
	if (!lcm.good()) {
		return 1;
	}
