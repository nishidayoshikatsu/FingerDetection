#include "opencv2/opencv.hpp"

using namespace cv;

//#define MIN_HSVCOLOR Scalar(0, 60, 80)
//#define MAX_HSVCOLOR Scalar(10, 160, 240)
#define MIN_HSVCOLOR Scalar(0, 10, 0)
#define MAX_HSVCOLOR Scalar(50, 180, 255)

#define CLOSING 2	/* クロージングの回数 */
#define OPENING 2	/* オープニングの回数 */

#define BOUTYOU 2	/* 膨張処理回数 */


Mat RGBtoHSV(Mat);
Mat GenerateMask(Mat);
Mat RGBtoGray(Mat);
Mat opening(Mat frame_gray, int syori_num);
Mat closing(Mat frame_gray, int syori_num);
int Filter(Mat frame_gray, int x, int y, int Oper[]);
Mat sikaku_kuro(Mat frame_gray, int x, int y);
Mat sikaku_siro(Mat frame_gray, int x, int y);

Mat GaussianFilter(Mat img, double sigma, int kernel_size);

void CalcGravity(Mat, float*);
Mat LaplacianFilter(Mat img, int kernel_size);

Mat DetectFinger(Mat, float*);