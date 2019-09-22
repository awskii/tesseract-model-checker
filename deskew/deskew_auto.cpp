#include <opencv2/opencv.hpp>

#define DEBUG

using namespace std;
using namespace cv;

const unsigned int HOUGH_STEP_SIZE = 1;
const unsigned int HOUGH_VOTE_TRESHOLD = 35;
const unsigned int HOUGH_MAX_LINE_GAP = 1;
const unsigned int HOUGH_MIN_LINE_LEN = 25;

double detect_skew(const char* filename) {
  Mat src = imread(filename, IMREAD_COLOR);
  Size size = src.size();
  cout << "img size is " << src.size() << endl;

  Mat src_gray;
  cvtColor(src, src_gray, COLOR_BGR2GRAY);
  bitwise_not(src_gray, src_gray);

#ifdef DEBUG
 imshow("gray", src_gray);
#endif // DEBUG

  vector<Vec4i> lines;
  HoughLinesP(src_gray, lines, HOUGH_STEP_SIZE, CV_PI/180, HOUGH_VOTE_TRESHOLD, HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP);

  Mat disp_lines(size, CV_8UC1, Scalar(0,0,0));
  double angle = 0.0;
  unsigned num_lines = lines.size();

  vector<double> v;
  for (unsigned i = 0; i < num_lines; i++) {
    line(disp_lines, Point(lines[i][0], lines[i][1]),
        Point(lines[i][2], lines[i][3]), Scalar(0xff, 0, 0));
    double w = atan2((double)lines[i][3] - lines[i][1],
        (double)lines[i][2] - lines[i][0]);
    angle += w;
    v.push_back(w);
  }
  angle *= 180/(CV_PI*num_lines);

  sort(v.begin(), v.end());
  int m = v.size()/2;
  double med_angle = v[m]*180/CV_PI;

  cout << "angle_avg=" << angle << " " << "angle_median=" << med_angle  << endl;

#ifdef DEBUG
 imshow(filename, disp_lines);
 waitKey(0);
 destroyWindow("gray");
 destroyWindow(filename);
#endif // debug
  return angle;
}

void deskew(const char* filename, double angle) {
 Mat src = imread(filename, IMREAD_COLOR);
 Mat src_gray;
 cvtColor(src, src_gray, COLOR_BGR2GRAY);
 bitwise_not(src_gray, src_gray);

 vector<Point> points;
 Mat_<uchar>::iterator start = src_gray.begin<uchar>();
 Mat_<uchar>::iterator end   = src_gray.end<uchar>();
 for (; start != end; ++start) {
   if (*start) points.push_back(start.pos());
 }

 RotatedRect box = minAreaRect(Mat(points));
 Mat rot_mx, rotated, cropped;
 rot_mx = getRotationMatrix2D(box.center, angle, 1);
 warpAffine(src_gray, rotated, rot_mx, src_gray.size(), INTER_CUBIC);

 Size box_sz = box.size;
 if (box.angle < -45.) swap(box_sz.width, box_sz.height);
 getRectSubPix(rotated, box_sz, box.center, cropped);

#ifdef DEBUG
 imshow("rotated", rotated);
 imshow("cropped", cropped);
 waitKey(0);
 destroyWindow("rotated");
 destroyWindow("cropped");
#endif // DEBUG
}

int main(int argc, const char** argv) {
  if (argc < 2) {
    cout << "pass image as second argument." << endl;
    return 1;
  }

  double skew = detect_skew(argv[1]);
  deskew(argv[1], skew);
  return 0;
}
