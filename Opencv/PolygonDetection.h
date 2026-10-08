// ============================================================
//  PolygonDetection.h
// ============================================================

#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace cv;
using namespace std;

struct DetectedPolygon
{
    string          name = "";       // "Pentagon" or "Hexagon"
    int             sides = 0;        
    vector<Point>   contour;
    Point2f         center = { 0, 0 };
    double          area = 0.0;      
    Rect            boundingBox = { 0,0,0,0 };
    vector<float>   sideLengths;
};


vector<DetectedPolygon> detectPolygons(const Mat& frame);

// Draws polygons with side length annotations
void drawPolygons(Mat& frame,
    const vector<DetectedPolygon>& polygons,
    float                          pixelsPerCm,
    bool                           isCalibrated);

// Full pipeline — detect + draw, returns annotated frame
Mat runPolygonDetection(const Mat& frame,
    float      pixelsPerCm,
    bool       isCalibrated);

// Classify shape by number of sides
string classifyPolygon(int sides, const vector<Point>& contour);
