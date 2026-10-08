// ============================================================
//  Coinmeasurement.cpp
//  Detects circular coins using Hough Circle Transform
//  Detects £1 coin (12-sided) using dedicated contour analysis
//  Measures diameter and identifies £1 / £2 / 2p coins by size
//  Requires calibration (pixelsPerCm) before measurements shown
// ============================================================

#include "Coinmeasurement.h"
#include <algorithm>
#include <cstdio>
using namespace cv;
using namespace std;

// ─── Global text style ───────────────────────────────────────────────────────
static const int    CM_FONT_FACE = FONT_HERSHEY_SIMPLEX;
static const double CM_FONT_SCALE_MAIN = 0.7;
static const double CM_FONT_SCALE_SMALL = 0.5;
static const int    CM_THICK_MAIN = 2;
static const int    CM_THICK_SMALL = 1;

// ─── Colours ─────────────────────────────────────────────────────────────────
static const Scalar CM_COLOUR(255, 128, 0);        // blue — round coins
static const Scalar CM_COLOUR_POUND1(0, 215, 255); // gold — £1 coin

// ─── Coin size definitions (diameter in mm) ──────────────────────────────────
static const float POUND1_MIN = 21.43f;   // £1 = 23.43mm ± 2mm
static const float POUND1_MAX = 25.43f;
static const float POUND2_MIN = 26.40f;   // £2 = 28.40mm ± 2mm
static const float POUND2_MAX = 30.40f;
static const float PENNY2_MIN = 23.90f;   // 2p = 25.9mm  ± 2mm
static const float PENNY2_MAX = 27.90f;

// ─── Helper: Classify round coin by diameter ─────────────────────────────────
static string classifyCoin(float diamMm)
{
    if (diamMm >= POUND2_MIN && diamMm <= POUND2_MAX) return "2 Pound Coin";
    if (diamMm >= POUND1_MIN && diamMm <= POUND1_MAX) return "1 Pound Coin";
    if (diamMm >= PENNY2_MIN && diamMm <= PENNY2_MAX) return "2p Coin";
    if (diamMm >= 40.0f && diamMm <= 80.0f)      return "Circle";
    return "Small Circle";
}

// ─── Helper: Draw label with black background box ────────────────────────────
static void drawLabel(Mat& frame,
    const string& text,
    Point         origin,
    double        scale,
    int           thickness,
    Scalar        colour)
{
    int  baseline = 0;
    Size ts = getTextSize(text, CM_FONT_FACE, scale, thickness, &baseline);
    rectangle(frame,
        origin + Point(-4, -ts.height - 4),
        origin + Point(ts.width + 4, baseline + 4),
        Scalar(0, 0, 0), FILLED);
    putText(frame, text, origin,
        CM_FONT_FACE, scale, colour, thickness, LINE_AA);
}

// ─── Helper: Draw round coin annotation ──────────────────────────────────────
static void drawCoinAnnotation(Mat& frame,
    Point         centre,
    int           radius,
    const string& label,
    float         diamMm,
    Scalar        colour)
{
    circle(frame, centre, radius, colour, 2, LINE_AA);
    circle(frame, centre, 4, colour, -1, LINE_AA);

    drawLabel(frame, label,
        Point(centre.x - 30, centre.y - radius - 10),
        CM_FONT_SCALE_MAIN, CM_THICK_MAIN, colour);

    char buf[64];
    snprintf(buf, sizeof(buf), "D: %.1f mm", diamMm);
    drawLabel(frame, buf,
        Point(centre.x - 35, centre.y),
        CM_FONT_SCALE_SMALL, CM_THICK_SMALL, colour);
}

// ─── £1 Coin Detector ────────────────────────────────────────────────────────
// The £1 coin is a 12-sided dodecagon — HoughCircles misses it because
// it is not a perfect circle. Instead this function uses a circularity
// window approach:
//   - Lower bound (0.82): rejects pentagons, hexagons, rough shapes
//   - Upper bound (0.92): rejects perfect circles already caught by Hough
//   The £1 sits in the 0.82–0.92 window because it is nearly-but-not-
//   quite circular, then a size check confirms it is the right diameter.
static void detectAndDrawPoundOne(Mat& frame,
    const Mat& edges,
    float      pixelsPerCm,
    bool       isCalibrated)
{
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (const auto& c : contours)
    {
        double area = contourArea(c);
        if (area < 1000) continue;   // too small — ignore noise

        // Circularity: 4π×area / perimeter²
        // Perfect circle = 1.0 | £1 dodecagon ≈ 0.85–0.91
        // Window 0.82–0.92 catches £1 without overlap with round coins or polygons
        double perim = arcLength(c, true);
        if (perim < 1.0) continue;
        double circularity = (4.0 * CV_PI * area) / (perim * perim);
        if (circularity < 0.82 || circularity > 0.92) continue;

        // Get enclosing circle for centre point, radius and drawing
        Point2f centre;
        float   radius;
        minEnclosingCircle(c, centre, radius);

        if (!isCalibrated)
        {
            circle(frame, Point(centre), (int)radius, Scalar(128, 128, 128), 1);
            continue;
        }

        // Convert from pixels to mm using calibration scale
        float diamMm = (2.0f * radius / pixelsPerCm) * 10.0f;

        // Final size gate — must match real £1 diameter (23.43mm ± 2mm)
        if (diamMm < POUND1_MIN || diamMm > POUND1_MAX) continue;

        // Draw the actual contour outline in gold (shows the 12-sided shape)
        drawContours(frame, vector<vector<Point>>{c}, -1, CM_COLOUR_POUND1, 2, LINE_AA);
        circle(frame, Point(centre), 4, CM_COLOUR_POUND1, -1, LINE_AA);

        drawLabel(frame, "1 Pound Coin",
            Point((int)centre.x - 40, (int)centre.y - (int)radius - 10),
            CM_FONT_SCALE_MAIN, CM_THICK_MAIN, CM_COLOUR_POUND1);

        char buf[64];
        snprintf(buf, sizeof(buf), "D: %.1f mm", diamMm);
        drawLabel(frame, buf,
            Point((int)centre.x - 35, (int)centre.y),
            CM_FONT_SCALE_SMALL, CM_THICK_SMALL, CM_COLOUR_POUND1);
    }
}

// ─── Main Detection Function ─────────────────────────────────────────────────
void detectAndDrawCoins(Mat& frame,
    const Mat& blurred,
    const Mat& gray,
    const Mat& edges,
    float      pixelsPerCm,
    bool       isCalibrated)
{
    // ── Step 1: Detect round coins (£2, 2p) via Hough ────────────────────────
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT,
        1,
        gray.rows / 6,   // minDist — allows two coins placed close together
        100,             // higher Canny threshold — filters weak logo edges
        65,              // higher accumulator — only confident circles pass
        28,              // minimum radius (pixels)
        250);            // maximum radius (pixels)


    sort(circles.begin(), circles.end(),
        [](const Vec3f& a, const Vec3f& b) { return a[2] > b[2]; });

    for (const auto& c : circles)
    {
        Point centre((int)c[0], (int)c[1]);
        int   radius = (int)c[2];

        // ── Circularity verification ──────────────────────────────────────
        // Mask the edges image to this circle's region, then confirm the
        // content inside is actually round — rejects pentagons/hexagons
        Mat mask = Mat::zeros(gray.size(), CV_8U);
        circle(mask, centre, radius + 5, Scalar(255), -1);
        Mat roi;
        bitwise_and(edges, mask, roi);

        vector<vector<Point>> localContours;
        findContours(roi, localContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool isCircular = false;
        for (const auto& lc : localContours)
        {
            double a = contourArea(lc);
            double p = arcLength(lc, true);
            if (p < 1.0) continue;
            if ((4.0 * CV_PI * a) / (p * p) > 0.75) { isCircular = true; break; }
        }
        if (!isCircular) continue;

        if (!isCalibrated)
        {
            circle(frame, centre, radius, Scalar(128, 128, 128), 1);
            continue;
        }

        float  diamMm = (2.0f * radius / pixelsPerCm) * 10.0f;
        string label = classifyCoin(diamMm);
        drawCoinAnnotation(frame, centre, radius, label, diamMm, CM_COLOUR);
    }

    // ── Step 2: Detect £1 coin via dedicated dodecagon detector ──────────────
    detectAndDrawPoundOne(frame, edges, pixelsPerCm, isCalibrated);
}
