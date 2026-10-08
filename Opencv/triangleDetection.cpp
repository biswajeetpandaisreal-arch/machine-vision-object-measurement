// ============================================================
//  triangleDetection.cpp
// ============================================================

#include "triangleDetection.h"
#include <cstdio>
#include <algorithm>
#include <cmath>
using namespace cv;
using namespace std;

// ─── Global text style ───────────────────────────────────────────────────────
static const int    TD_FONT_FACE = FONT_HERSHEY_SIMPLEX;
static const double TD_FONT_SCALE_MAIN = 0.7;
static const double TD_FONT_SCALE_SMALL = 0.5;
static const int    TD_THICK_MAIN = 2;
static const int    TD_THICK_SMALL = 1;

// ─── Orange (BGR) ─────────────────────────────────────────────────────────────
static const Scalar TD_COLOUR(0, 120, 200);

// ─── Helper: Euclidean distance ──────────────────────────────────────────────
static float pointDist(const Point2f& a, const Point2f& b)
{
    float dx = a.x - b.x, dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// ─── Helper: Classify triangle ───────────────────────────────────────────────
static string classifyTriangle(float s1, float s2, float s3)
{
    float tol = 0.10f;
    bool s12 = fabs(s1 - s2) / max(s1, s2) < tol;
    bool s23 = fabs(s2 - s3) / max(s2, s3) < tol;
    bool s13 = fabs(s1 - s3) / max(s1, s3) < tol;
    if (s12 && s23 && s13) return "Equilateral";
    if (s12 || s23 || s13) return "Isosceles";
    return "Scalene";
}

// ─── Helper: Draw text label (no background box) ─────────────────────────────
static void drawLabel(Mat& frame,
    const string& text,
    Point         origin,
    double        scale,
    int           thickness)
{
    putText(frame, text, origin,
        TD_FONT_FACE, scale, TD_COLOUR, thickness, LINE_AA);
}

// ─── Main Detection ───────────────────────────────────────────────────────────
void detectAndDrawTriangles(Mat& frame,
    const vector<vector<Point>>& contours,
    float                        pixelsPerCm,
    bool                         isCalibrated)
{
    for (const auto& cnt : contours)
    {
        double area = contourArea(cnt);
        if (area < 1000) continue;

        double        perimeter = arcLength(cnt, true);
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * perimeter, true);

        if (approx.size() != 3)        continue;
        if (!isContourConvex(approx))  continue;

        Point2f p0(approx[0]), p1(approx[1]), p2(approx[2]);

        float side1Px = pointDist(p0, p1);
        float side2Px = pointDist(p1, p2);
        float side3Px = pointDist(p2, p0);

        // ── Draw outline — orange ─────────────────────────────────────────────
        for (int j = 0; j < 3; j++)
            line(frame, approx[j], approx[(j + 1) % 3], TD_COLOUR, 2, LINE_AA);

        // ── Draw vertex dots — orange ─────────────────────────────────────────
        for (int j = 0; j < 3; j++)
            circle(frame, approx[j], 4, TD_COLOUR, -1, LINE_AA);

        Point centroid(
            (approx[0].x + approx[1].x + approx[2].x) / 3,
            (approx[0].y + approx[1].y + approx[2].y) / 3);

        if (isCalibrated)
        {
            float s1Mm = (side1Px / pixelsPerCm) * 10.0f;
            float s2Mm = (side2Px / pixelsPerCm) * 10.0f;
            float s3Mm = (side3Px / pixelsPerCm) * 10.0f;

            string triType = classifyTriangle(s1Mm, s2Mm, s3Mm);
            drawLabel(frame, triType,
                Point(centroid.x - 40, centroid.y - 10),
                TD_FONT_SCALE_MAIN, TD_THICK_MAIN);

            char buf[32];

            snprintf(buf, sizeof(buf), "%.1f mm", s1Mm);
            drawLabel(frame, buf,
                Point((int)((p0.x + p1.x) / 2) - 20, (int)((p0.y + p1.y) / 2) - 8),
                TD_FONT_SCALE_SMALL, TD_THICK_SMALL);

            snprintf(buf, sizeof(buf), "%.1f mm", s2Mm);
            drawLabel(frame, buf,
                Point((int)((p1.x + p2.x) / 2) - 20, (int)((p1.y + p2.y) / 2) - 8),
                TD_FONT_SCALE_SMALL, TD_THICK_SMALL);

            snprintf(buf, sizeof(buf), "%.1f mm", s3Mm);
            drawLabel(frame, buf,
                Point((int)((p2.x + p0.x) / 2) - 20, (int)((p2.y + p0.y) / 2) - 8),
                TD_FONT_SCALE_SMALL, TD_THICK_SMALL);
        }
        else
        {
            drawLabel(frame, "Triangle",
                Point(centroid.x - 30, centroid.y),
                TD_FONT_SCALE_MAIN, TD_THICK_MAIN);
        }
    }
}
