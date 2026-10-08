#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

void detectAndDrawCoins(
    Mat& frame,
    const Mat& blurred,
    const Mat& gray,
    const Mat& edges,
    float pixelsPerCm,
    bool isCalibrated);
