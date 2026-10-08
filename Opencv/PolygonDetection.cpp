// ============================================================
//  PolygonDetection.cpp
//  Detects pentagons and hexagons using contour approximation
//  Measures and displays the length of each side in mm
//  Uses calibration scale (pixelsPerCm) from reference card
//  Two-layer filter prevents coins being detected as hexagons:
//    Layer 1 — circularity check (4π×area / perimeter²)
//    Layer 2 — fill ratio check (area / enclosing circle area)
// ============================================================

#include "PolygonDetection.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
using namespace cv;
using namespace std;

// ─── Global text style ───────────────────────────────────────────────────────
static const int    PD_FONT_FACE = FONT_HERSHEY_SIMPLEX;
static const double PD_FONT_SCALE_MAIN = 0.7;
static const double PD_FONT_SCALE_SMALL = 0.5;
static const int    PD_THICK_MAIN = 2;
static const int    PD_THICK_SMALL = 1;

// ─── Filters ─────────────────────────────────────────────────────────────────
// Pentagon ≈ 0.72, Hexagon ≈ 0.79
// Rectangles ≈ 0.50–0.67 → rejected by MIN
// Coins ≈ 0.85–1.0       → rejected by MAX
static const double MIN_CIRCULARITY = 0.65;
static const double MAX_CIRCULARITY = 0.87;
static const double MAX_FILL_RATIO = 0.88;

// ─── Colour map ──────────────────────────────────────────────────────────────
static Scalar getColour(const string& name)
{
    if (name == "Pentagon") return Scalar(255, 0, 255);
    if (name == "Hexagon")  return Scalar(255, 0, 255);
    return Scalar(200, 200, 200);
}

// ─── Classify by vertex count ────────────────────────────────────────────────
string classifyPolygon(int sides, const vector<Point>&)
{
    switch (sides)
    {
    case 5: return "Pentagon";
    case 6: return "Hexagon";
    default: return "";
    }
}

// ─── tryApprox ───────────────────────────────────────────────────────────────
// Sweeps epsilon to find an approximation landing on exactly 5 or 6 sides
static bool tryApprox(const vector<Point>& contour, vector<Point>& bestApprox)
{
    double perim = arcLength(contour, true);
    for (double factor = 0.01; factor <= 0.08; factor += 0.005)
    {
        vector<Point> approx;
        approxPolyDP(contour, approx, factor * perim, true);
        int s = (int)approx.size();
        if (s == 5 || s == 6)
        {
            bestApprox = approx;
            return true;
        }
    }
    return false;
}

// ─── Main Detection ──────────────────────────────────────────────────────────
vector<DetectedPolygon> detectPolygons(const Mat& frame)
{
    vector<DetectedPolygon> results;

    // ── Preprocessing ─────────────────────────────────────────────────────────
    Mat gray, blurred, thresh;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // CLAHE normalises contrast for dark shapes on light backgrounds
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, gray);

    GaussianBlur(gray, blurred, Size(7, 7), 2);

    // Canny traces clean single-pixel outlines without filling interior noise
    Mat edges;
    Canny(blurred, edges, 50, 150);

    // Dilate once to close small gaps in the contour outline
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(edges, thresh, kernel, Point(-1, -1), 1);

    // ── Contour extraction ────────────────────────────────────────────────────
    vector<vector<Point>> contours;
    findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // ── Process each contour ──────────────────────────────────────────────────
    for (const auto& contour : contours)
    {
        double area = contourArea(contour);
        if (area < 3000) continue;

        double perimeter = arcLength(contour, true);
        if (perimeter < 1.0) continue;

        // ── Layer 1: Circularity window ───────────────────────────────────────
        double circularity = (4.0 * CV_PI * area) / (perimeter * perimeter);
        if (circularity < MIN_CIRCULARITY) continue;   // too low = rectangle
        if (circularity > MAX_CIRCULARITY) continue;   // too high = coin

        // ── Layer 2: Fill ratio filter ────────────────────────────────────────
        Point2f enclosingCenter;
        float   enclosingRadius;
        minEnclosingCircle(contour, enclosingCenter, enclosingRadius);
        double fillRatio = area / (CV_PI * enclosingRadius * enclosingRadius);
        if (fillRatio > MAX_FILL_RATIO) continue;

        // ── Epsilon sweep ─────────────────────────────────────────────────────
        vector<Point> approx;
        if (!tryApprox(contour, approx)) continue;

        // Convexity check — rejects L-shapes, concave cutouts
        if (!isContourConvex(approx)) continue;

        // ── Aspect ratio guard ────────────────────────────────────────────────
        // Pentagons/hexagons: 0.65–1.55
        // Wide/tall rectangles: outside this range
        Rect bb = boundingRect(approx);
        double aspectRatio = (double)bb.width / bb.height;
        if (aspectRatio < 0.55 || aspectRatio > 1.65) continue;

        // ── Side length uniformity check ──────────────────────────────────────
        // True pentagons/hexagons have roughly equal sides (ratio > 0.40)
        float minSide = FLT_MAX, maxSide = 0;
        vector<float> sideLengths;
        for (size_t s = 0; s < approx.size(); s++)
        {
            Point p1 = approx[s];
            Point p2 = approx[(s + 1) % approx.size()];
            float len = (float)norm(p1 - p2);
            sideLengths.push_back(len);
            if (len < minSide) minSide = len;
            if (len > maxSide) maxSide = len;
        }
        if (maxSide < 1.0f) continue;
        if ((minSide / maxSide) < 0.30f) continue;

        // Classify
        string name = classifyPolygon((int)approx.size(), approx);
        if (name.empty()) continue;

        // ── Centroid ──────────────────────────────────────────────────────────
        Point2f center;
        Moments M = moments(contour);
        if (M.m00 != 0)
            center = Point2f((float)(M.m10 / M.m00), (float)(M.m01 / M.m00));
        else
            center = Point2f(bb.x + bb.width / 2.0f, bb.y + bb.height / 2.0f);

        // ── Build result ──────────────────────────────────────────────────────
        DetectedPolygon poly;
        poly.name = name;
        poly.sides = (int)approx.size();
        poly.contour = approx;
        poly.center = center;
        poly.area = area;
        poly.boundingBox = bb;
        poly.sideLengths = sideLengths;
        results.push_back(poly);
    }

    return results;
}

// ─── Drawing ─────────────────────────────────────────────────────────────────
void drawPolygons(Mat& frame,
    const vector<DetectedPolygon>& polygons,
    float                          pixelsPerCm,
    bool                           isCalibrated)
{
    for (const auto& poly : polygons)
    {
        Scalar colour = getColour(poly.name);

        // Semi-transparent fill
        Mat overlay = frame.clone();
        vector<vector<Point>> c = { poly.contour };
        fillPoly(overlay, c,
            Scalar(colour[0] * 0.3, colour[1] * 0.3, colour[2] * 0.3));
        addWeighted(overlay, 0.25, frame, 0.75, 0, frame);

        // Draw sides with length labels
        for (int s = 0; s < poly.sides; s++)
        {
            Point p1 = poly.contour[s];
            Point p2 = poly.contour[(s + 1) % poly.sides];
            Point mid((p1.x + p2.x) / 2, (p1.y + p2.y) / 2);

            line(frame, p1, p2, colour, 2, LINE_AA);

            char buf[32];
            if (isCalibrated)
            {
                float lenMm = (poly.sideLengths[s] / pixelsPerCm) * 10.0f;
                snprintf(buf, sizeof(buf), "%.1f mm", lenMm);
            }
            else
                snprintf(buf, sizeof(buf), "%.0f px", poly.sideLengths[s]);

            int  baseline = 0;
            Size ts = getTextSize(buf, PD_FONT_FACE,
                PD_FONT_SCALE_SMALL, PD_THICK_SMALL, &baseline);
            putText(frame, buf,
                mid + Point(-ts.width / 2, ts.height / 2),
                PD_FONT_FACE, PD_FONT_SCALE_SMALL, colour, PD_THICK_SMALL, LINE_AA);
        }

        // Vertex dots
        for (const auto& pt : poly.contour)
            circle(frame, pt, 4, colour, -1, LINE_AA);

        // Shape name at centroid
        string label = poly.name + " (" + to_string(poly.sides) + " sides)";
        int  baseline = 0;
        Size textSize = getTextSize(label, PD_FONT_FACE,
            PD_FONT_SCALE_MAIN, PD_THICK_MAIN, &baseline);
        Point textOrigin((int)poly.center.x - textSize.width / 2,
            (int)poly.center.y + textSize.height / 2);
        putText(frame, label, textOrigin,
            PD_FONT_FACE, PD_FONT_SCALE_MAIN, colour, PD_THICK_MAIN, LINE_AA);

        circle(frame, poly.center, 5, colour, -1, LINE_AA);
    }
}

// ─── Full Pipeline ────────────────────────────────────────────────────────────
Mat runPolygonDetection(const Mat& frame, float pixelsPerCm, bool isCalibrated)
{
    Mat output = frame.clone();
    vector<DetectedPolygon> polygons = detectPolygons(frame);
    drawPolygons(output, polygons, pixelsPerCm, isCalibrated);
    return output;
}
