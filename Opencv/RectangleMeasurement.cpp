// ============================================================
//  RectangleMeasurement.cpp
//  Detects rectangular objects and uses a standard credit/debit
//  card (85.6mm × 54.0mm) as the calibration reference
//  Once calibrated, measures all other rectangles in mm
//  Supports multiple cards of the same size simultaneously
// ============================================================

#include "RectangleMeasurement.h"
#include <cstdio>
#include <algorithm>
using namespace cv;
using namespace std;

// ─── Global text style (matches all other shape files) ───────────────────────
static const int    RM_FONT_FACE = FONT_HERSHEY_SIMPLEX;
static const double RM_FONT_SCALE_MAIN = 0.7;   // dimension labels
static const double RM_FONT_SCALE_SMALL = 0.5;   // secondary labels
static const int    RM_THICK_MAIN = 2;
static const int    RM_THICK_SMALL = 1;

// ─── Colours (green for all rectangles) ──────────────────────────────────────
static const Scalar RM_COLOUR_CARD(0, 255, 0);   // green — reference card
static const Scalar RM_COLOUR_RECT(0, 255, 0);   // green — other rectangles

// ─── Reference card dimensions (ISO/IEC 7810 ID-1 standard) ─────────────────
static const float  CARD_WIDTH_CM = 8.56f;
static const float  CARD_HEIGHT_CM = 5.40f;
static const float  CARD_RATIO = CARD_WIDTH_CM / CARD_HEIGHT_CM;  // ~1.585

// ─── Detection thresholds ────────────────────────────────────────────────────
static const double MIN_CARD_AREA = 8000.0;  // minimum px² for a card-sized object
static const double MIN_RECT_AREA = 2000.0;  // minimum px² for other rectangles
static const float  RATIO_TOLERANCE = 0.10f;   // ±10% aspect ratio tolerance

// ─── Helper: Check if a 4-point contour is a valid rectangle ─────────────────
// A valid rectangle must be convex and have roughly 90° angles at all corners
static bool isValidRectangle(const vector<Point>& approx)
{
    if (approx.size() != 4)       return false;
    if (!isContourConvex(approx)) return false;

    for (int i = 0; i < 4; i++)
    {
        Point  o = approx[i];
        Point  a = approx[(i + 1) % 4];
        Point  b = approx[(i + 3) % 4];

        double dx1 = a.x - o.x, dy1 = a.y - o.y;
        double dx2 = b.x - o.x, dy2 = b.y - o.y;
        double dot = dx1 * dx2 + dy1 * dy2;
        double mag = sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));

        if (mag == 0) return false;

        double angle = acos(max(-1.0, min(1.0, dot / mag))) * 180.0 / CV_PI;

        // Each corner must be within 15° of 90°
        if (abs(angle - 90.0) > 15.0) return false;
    }
    return true;
}

// ─── Helper: Check if aspect ratio matches a standard card ───────────────────
static bool matchesCardRatio(float wPx, float hPx)
{
    float ratio = wPx / (hPx + 1e-6f);
    return (ratio >= CARD_RATIO * (1.0f - RATIO_TOLERANCE) &&
        ratio <= CARD_RATIO * (1.0f + RATIO_TOLERANCE));
}

// ─── Helper: Draw rectangle outline ──────────────────────────────────────────
static void drawRectOutline(Mat& frame,
    const Point2f pts[4],
    const Scalar& colour,
    int           thickness)
{
    for (int j = 0; j < 4; j++)
        line(frame, pts[j], pts[(j + 1) % 4], colour, thickness, LINE_AA);
}

// ─── Helper: Draw label with black background box ────────────────────────────
static void drawLabel(Mat& frame,
    const string& text,
    Point         origin,
    const Scalar& colour,
    double        scale,
    int           thickness)
{
    int  baseline = 0;
    Size ts = getTextSize(text, RM_FONT_FACE, scale, thickness, &baseline);

    rectangle(frame,
        origin + Point(-4, -ts.height - 4),
        origin + Point(ts.width + 4, baseline + 4),
        Scalar(0, 0, 0), FILLED);

    putText(frame, text, origin,
        RM_FONT_FACE, scale, colour, thickness, LINE_AA);
}

// ─── Helper: Draw dimension label at rectangle centre ────────────────────────
static void drawDimLabel(Mat& frame,
    Point2f       centre,
    float         wMm,
    float         hMm,
    const Scalar& colour)
{
    char buf[64];
    snprintf(buf, sizeof(buf), "%.1f x %.1f mm", wMm, hMm);
    drawLabel(frame, buf,
        Point((int)centre.x - 40, (int)centre.y),
        colour,
        RM_FONT_SCALE_MAIN,
        RM_THICK_MAIN);
}

// ─── Main Detection Function ──────────────────────────────────────────────────
bool detectAndDrawRectangles(Mat& frame,
    const vector<vector<Point>>& contours,
    float& pixelsPerCm,
    bool& isCalibrated,
    RotatedRect& refRect)
{
    bool cardFound = false;

    for (const auto& cnt : contours)
    {
        double area = contourArea(cnt);

        // Skip contours too small to be a real rectangle
        if (area < MIN_RECT_AREA) continue;

        // Approximate contour to a polygon
        double        perimeter = arcLength(cnt, true);
        vector<Point> approx;
        approxPolyDP(cnt, approx, 0.04 * perimeter, true);

        // Must pass strict rectangle check (4 sides + 90° angles)
        if (!isValidRectangle(approx)) continue;

        // Get oriented bounding box — handles rotated rectangles correctly
        RotatedRect rr = minAreaRect(cnt);
        float       wPx = max(rr.size.width, rr.size.height);
        float       hPx = min(rr.size.width, rr.size.height);

        Point2f pts[4];
        rr.points(pts);

        // ── Card-sized rectangle ──────────────────────────────────────────────
        if (area >= MIN_CARD_AREA && matchesCardRatio(wPx, hPx))
        {
            // Calibrate using the FIRST card only — subsequent cards
            // reuse the same pixelsPerCm so scale stays consistent
            if (!cardFound)
            {
                pixelsPerCm = ((wPx / CARD_WIDTH_CM) +
                    (hPx / CARD_HEIGHT_CM)) / 2.0f;
                isCalibrated = true;
                refRect = rr;
                cardFound = true;
            }

            // Draw ALL detected cards — not just the first one
            drawRectOutline(frame, pts, RM_COLOUR_CARD, 3);

            // Convert to mm for display
            float wMm = (wPx / pixelsPerCm) * 10.0f;
            float hMm = (hPx / pixelsPerCm) * 10.0f;
            drawDimLabel(frame, rr.center, wMm, hMm, RM_COLOUR_CARD);

            continue;   // card fully processed
        }

        // ── Other rectangles (non-card sized) ────────────────────────────────
        if (!isCalibrated) continue;

        // Convert to mm for display
        float wMm = (wPx / pixelsPerCm) * 10.0f;
        float hMm = (hPx / pixelsPerCm) * 10.0f;

        drawRectOutline(frame, pts, RM_COLOUR_RECT, 2);
        drawDimLabel(frame, rr.center, wMm, hMm, RM_COLOUR_RECT);
    }

    return cardFound;
}
