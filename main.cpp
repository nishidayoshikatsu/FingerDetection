#include "opencv2/opencv.hpp"
#include "preprocessing.hpp"
//#include "cpu.hpp"

using namespace cv;

Mat main_process(Mat);

int main(int argh, char* argv[])
{
    VideoCapture cap(0);//デバイスのオープン

    if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
    {
        return -1;
    }

    Mat frame; //取得したフレーム
	Mat frame_out;
	Rect rect(140, 60, 360, 360);	//x,y, width, height
	//int *gpx;
	//int *gpy;
	//printf("start");
	//Mat paper = imread("./input/paper_sample.jpg", 0);
	//printf("finish");
    while(cap.read(frame))//無限ループ
    {
		frame_out = main_process(frame);
		//printf("幅:%d, 高さ%d\n", frame.cols, frame.rows);
		//cvtColor(frame, frame_out, CV_BGR2GRAY);
		//paper = frame;
		//Mat dst = Mat::ones(frame.rows, frame.cols, CV_8U);

		//resize(paper, dst, dst.size(), INTER_CUBIC);
		//paper = scale(paper, dst, (double)dst.rows/frame.rows, (double)dst.cols/frame.cols);
		rectangle(frame, rect, Scalar(0, 0, 255), 3, cv::LINE_4);

		imshow("Rock Paper Scissors Origin", frame);
		imshow("Rock Paper Scissors", frame_out);

		frame_out = LaplacianFilter(frame_out, 3);
		imshow("Rock Paper Scissorsw", frame_out);
		//imshow("Paper", paper);

        const int key = waitKey(1);
        if(key == 'q'/*113*/){
            break;
        }
        else if(key == 's'/*115*/){
            imwrite("img.png", frame_out);
        }
    }
    destroyAllWindows();
    return 0;
}

Mat main_process(Mat frame){
	Mat frame_out = frame;
	Mat frame_gray, frame_gray2;
	Mat hsv_img, msk_img;
	int gpx, gpy;
	float x[] = {0.0, 0.0};

	/*frame = Sikisou_tyuusyutu(frame, frame_out, SIKISOU1 , SIKISOU2);
	frame_gray = RGBtoGray(frame);

	frame_gray2 = frame_gray;

	frame_gray = kido_tyuusyutu(frame_gray, frame_gray2, KIDO1, KIDO2);

	//frame_gray = Nichika(frame_gray2, NICHIKA);

	//frame_gray = opening(frame_gray, OPENING);
	//frame_gray = closing(frame_gray, CLOSING);
	frame_gray = Filter_laplacian(frame_gray, frame_gray2);

	//RockPaperScisors(1);
	*/
	frame = GaussianFilter(frame, 1.3, 3);

	frame = RGBtoHSV(frame);

	//Size ksize = cv::Size(5, 5);
    //GaussianBlur(frame, frame, ksize, 0);
	//cvtColor(frame, hsv_img, CV_BGR2HSV);
	//inRange(hsv_img, MIN_HSVCOLOR, MAX_HSVCOLOR, msk_img);
	frame_gray = GenerateMask(frame);

	//frame_gray = closing(frame_gray, CLOSING);
	//frame_gray = opening(frame_gray, OPENING);
	cv::erode(frame_gray, frame_gray, cv::Mat(), cv::Point(-1,-1), 2); // 収縮
	cv::dilate(frame_gray, frame_gray, cv::Mat(), cv::Point(-1,-1), 5); // 膨張

	CalcGravity(frame_gray, x);
	printf("gpx:%d, gpy:%d\n", (int)x[0], (int)x[1]);
	circle(frame_gray, Point((int)x[0], (int)x[1]), 10, 100, 3, 4);

	//frame_gray = LaplacianFilter(frame_gray, 3);

	//frame_gray2 = frame_gray;
	//frame_gray = Filter_laplacian(frame_gray, frame_gray2);

	frame_gray = DetectFinger(frame_gray, x);

	return frame_gray;
}