#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
using namespace cv;
using namespace std;

void detectAndDrawTriangles(
    Mat& frame,
    const vector<vector<Point>>& contours,
    float pixelsPerCm,
    bool isCalibrated);
