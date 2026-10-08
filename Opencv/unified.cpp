// ============================================================
//  unified.cpp
//  Main vision pipeline for the Object Measurement System
//  Handles both live camera and static image input modes
//  Detects and measures: rectangles, coins, triangles, polygons
//  Calibration reference: ISO/IEC 7810 ID-1 card (85.6×54mm)
// ============================================================

#include "unified.h"
#include "RectangleMeasurement.h"
#include "Coinmeasurement.h"
#include "triangleDetection.h"
#include "PolygonDetection.h"
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

// ─── Colour scheme (BGR) ─────────────────────────────────────────────────────
static const Scalar COL_GREEN(0, 255, 0);
static const Scalar COL_BLUE(255, 128, 0);
static const Scalar COL_ORANGE(0, 165, 255);
static const Scalar COL_PURPLE(255, 0, 255);
static const Scalar COL_RED(0, 0, 255);
static const Scalar COL_WHITE(255, 255, 255);
static const Scalar COL_GRAY(200, 200, 200);

// ─── Unified text style ───────────────────────────────────────────────────────
static const int    UI_FONT = FONT_HERSHEY_SIMPLEX;
static const double UI_SCALE_MAIN = 0.7;
static const double UI_SCALE_SMALL = 0.5;
static const int    UI_THICK_MAIN = 2;
static const int    UI_THICK_SMALL = 1;

// ─── countValidContours ───────────────────────────────────────────────────────
static int countValidContours(const vector<vector<Point>>& contours)
{
    int count = 0;
    for (const auto& c : contours)
    {
        double area = contourArea(c);
        if (area < 5000) continue;

        Rect   br = boundingRect(c);
        double ar = (double)br.width / br.height;
        if (ar > 8.0 || ar < 0.12) continue;

        count++;
    }
    return count;
}

// ─── applyBrightness ─────────────────────────────────────────────────────────
static Mat applyBrightness(const Mat& frame, int trackbarValue)
{
    Mat output;
    frame.convertTo(output, -1, 1.0, (double)(trackbarValue - 100));
    return output;
}

// ─── preprocessFrame ─────────────────────────────────────────────────────────
static Mat preprocessFrame(const Mat& frame, Mat& grayOut, Mat& blurredOut)
{
    cvtColor(frame, grayOut, COLOR_BGR2GRAY);

    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(grayOut, grayOut);

    GaussianBlur(grayOut, blurredOut, Size(9, 9), 2);

    Mat edges;
    adaptiveThreshold(blurredOut, edges, 255,
        ADAPTIVE_THRESH_GAUSSIAN_C,
        THRESH_BINARY_INV, 15, 3);
    return edges;
}

// ─── extractContours ─────────────────────────────────────────────────────────
static vector<vector<Point>> extractContours(const Mat& edges)
{
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    sort(contours.begin(), contours.end(),
        [](const vector<Point>& a, const vector<Point>& b)
        { return contourArea(a) > contourArea(b); });
    return contours;
}

// ─── drawStatusInfo ───────────────────────────────────────────────────────────
static void drawStatusInfo(Mat& frame,
    bool   isCalibrated,
    float  pixelsPerCm,
    int    objectCount,
    bool   imageMode,
    double fps)
{
    if (isCalibrated)
    {
        string txt = "Calibrated: " + to_string(pixelsPerCm).substr(0, 5) + " px/cm";
        putText(frame, txt, Point(10, 30), UI_FONT, UI_SCALE_MAIN, COL_GREEN, UI_THICK_MAIN);
        string obj = "Objects detected: " + to_string(objectCount);
        putText(frame, obj, Point(10, 60), UI_FONT, UI_SCALE_MAIN, COL_WHITE, UI_THICK_MAIN);
    }
    else
    {
        putText(frame, "Show reference card to calibrate...",
            Point(10, 30), UI_FONT, UI_SCALE_MAIN, COL_RED, UI_THICK_MAIN);
    }

    if (!imageMode)
    {
        char fpsBuf[32];
        snprintf(fpsBuf, sizeof(fpsBuf), "FPS: %.1f", fps);
        putText(frame, fpsBuf,
            Point(frame.cols - 120, 30),
            UI_FONT, UI_SCALE_MAIN, COL_WHITE, UI_THICK_MAIN);
    }

    int y = frame.rows - 100;
    putText(frame, "Rectangles", Point(10, y), UI_FONT, UI_SCALE_SMALL, COL_GREEN, UI_THICK_SMALL);
    putText(frame, "Coins", Point(10, y + 20), UI_FONT, UI_SCALE_SMALL, COL_BLUE, UI_THICK_SMALL);
    putText(frame, "Triangles", Point(10, y + 40), UI_FONT, UI_SCALE_SMALL, COL_ORANGE, UI_THICK_SMALL);
    putText(frame, "Polygons", Point(10, y + 60), UI_FONT, UI_SCALE_SMALL, COL_PURPLE, UI_THICK_SMALL);

    putText(frame, imageMode ? "IMAGE MODE" : "LIVE MODE",
        Point(10, frame.rows - 10), UI_FONT, UI_SCALE_SMALL, COL_GRAY, UI_THICK_SMALL);
}

// ─── runUnifiedSystem ─────────────────────────────────────────────────────────
void runUnifiedSystem(const string& imagePath)
{
    bool         imageMode = !imagePath.empty();
    Mat          staticImage;
    VideoCapture cap;

    if (!imageMode)
    {
        cap.open(0);
        if (!cap.isOpened())
        {
            cerr << "Error: Could not open camera." << endl;
            return;
        }
        cout << "LIVE MODE | SPACE = snapshot | ESC = quit" << endl;
    }
    else
    {
        staticImage = imread(imagePath);
        if (staticImage.empty())
        {
            cerr << "Error: Could not load: " << imagePath << endl;
            return;
        }
        cout << "IMAGE MODE: " << imagePath << " | SPACE = snapshot | ESC = quit" << endl;
    }

    bool        isCalibrated = false;
    float       pixelsPerCm = 1.0f;
    RotatedRect refRect;
    int         snapshotIndex = 0;

    double fps = 0.0;
    int    frameCount = 0;
    double fpsTimer = (double)getTickCount();

    const string WINDOW = "Unified Measurement System";
    namedWindow(WINDOW, WINDOW_AUTOSIZE);

    int brightness = 100;
    createTrackbar("Brightness", WINDOW, &brightness, 200);

    while (true)
    {
        Mat frame;
        if (imageMode)
            frame = staticImage.clone();
        else
        {
            cap >> frame;
            if (frame.empty()) break;

            frameCount++;
            if (frameCount >= 30)
            {
                double elapsed = ((double)getTickCount() - fpsTimer) / getTickFrequency();
                fps = frameCount / elapsed;
                frameCount = 0;
                fpsTimer = (double)getTickCount();
            }
        }

        Mat brightFrame = applyBrightness(frame, getTrackbarPos("Brightness", WINDOW));

        // Preprocess — produces grayscale, blurred and edge map
        Mat gray, blurred;
        Mat edges = preprocessFrame(brightFrame, gray, blurred);

        // Extract contours
        vector<vector<Point>> contours = extractContours(edges);

        // Run all detectors on brightFrame
        detectAndDrawCoins(brightFrame, blurred, gray, edges, pixelsPerCm, isCalibrated);
        detectAndDrawRectangles(brightFrame, contours, pixelsPerCm, isCalibrated, refRect);
        detectAndDrawTriangles(brightFrame, contours, pixelsPerCm, isCalibrated);
        vector<DetectedPolygon> polygons = detectPolygons(brightFrame);
        drawPolygons(brightFrame, polygons, pixelsPerCm, isCalibrated);

        drawStatusInfo(brightFrame, isCalibrated, pixelsPerCm,
            countValidContours(contours), imageMode, fps);

        // ── Build 2×2 grid display ────────────────────────────────────────────
        // All four tiles are resized to half the original frame dimensions
        // Grayscale/blur/edges converted to BGR so all tiles share the same type
        Size tileSize(brightFrame.cols / 2, brightFrame.rows / 2);

        Mat grayBGR, blurBGR, edgesBGR;
        cvtColor(gray, grayBGR, COLOR_GRAY2BGR);
        cvtColor(blurred, blurBGR, COLOR_GRAY2BGR);
        cvtColor(edges, edgesBGR, COLOR_GRAY2BGR);

        Mat t1, t2, t3, t4;
        resize(brightFrame, t1, tileSize);   // top-left:     annotated output
        resize(grayBGR, t2, tileSize);   // top-right:    grayscale
        resize(blurBGR, t3, tileSize);   // bottom-left:  gaussian blur
        resize(edgesBGR, t4, tileSize);   // bottom-right: edge map

        // White tile labels so each panel is clearly identified
        putText(t1, "Output", Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        putText(t2, "Grayscale", Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        putText(t3, "Gaussian Blur", Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        putText(t4, "Edge Map", Point(8, 22), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);

        // Stitch tiles into a single 2×2 grid and display
        Mat topRow, bottomRow, grid;
        hconcat(t1, t2, topRow);
        hconcat(t3, t4, bottomRow);
        vconcat(topRow, bottomRow, grid);

        imshow(WINDOW, grid);

        int key = waitKey(imageMode ? 30 : 10) & 0xFF;

        if (key == 27) break;   // ESC — exit

        if (key == ' ')         // SPACE — save snapshot and show it
        {
            string fn = "snapshot_" + to_string(snapshotIndex++) + ".png";
            imwrite(fn, grid);   // saves the full grid so you see all four panels
            cout << "Snapshot saved: " << fn << endl;
            imshow("Captured Image", grid);
        }
    }

    if (!imageMode) cap.release();
    destroyAllWindows();
}
