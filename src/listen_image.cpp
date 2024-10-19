#include <lcm/lcm-cpp.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

#include "image_types/image_grey_t.hpp"

using namespace cv;
using namespace lcm;
using namespace image_types;

class Handler
{
    public:
        ~Handler() {}

        void handleMessage(
            const ReceiveBuffer* rbuf,
            const std::string& chan,
            const image_grey_t* msg
        )
        {
            printf("WORKING\n");
            Mat img = Mat(msg->rows, msg->cols, CV_8UC1, (void*)msg->data.data(), Mat::AUTO_STEP);
            imshow("img", img);
            waitKey(0);
        }
};

int main(int argc, char** argv)
{
    LCM lcm;
    if(!lcm.good())
        return 1;

    Handler handlerObject;
    lcm.subscribe("IMAGE", &Handler::handleMessage, &handlerObject);
    
    while(0 == lcm.handle());
    return 0;
}