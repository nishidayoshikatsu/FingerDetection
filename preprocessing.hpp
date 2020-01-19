#include "opencv2/opencv.hpp"

using namespace cv;

#define SIKISOU1 34	/* 色相の閾値 */
#define SIKISOU2 335

#define KIDO1 40	/* 輝度値の閾値 */
#define KIDO2 140

#define NICHIKA 50	/* 二値化の閾値 */

#define CLOSING 2	/* クロージングの回数 */
#define OPENING 10	/* オープニングの回数 */

#define BOUTYOU 2	/* 膨張処理回数 */

Mat main_process(Mat);
Mat RGBtoGray(Mat);
Mat Sikisou_tyuusyutu(Mat, Mat, int, int);
Mat kido_tyuusyutu(Mat, Mat, int, int);
Mat Nichika(Mat, int);
Mat opening(Mat frame_gray, int syori_num);
Mat closing(Mat frame_gray, int syori_num);
int Filter(Mat frame_gray, int x, int y, int Oper[]);
Mat sikaku_kuro(Mat frame_gray, int x, int y);
Mat sikaku_siro(Mat frame_gray, int x, int y);
Mat Filter_laplacian(Mat, Mat);