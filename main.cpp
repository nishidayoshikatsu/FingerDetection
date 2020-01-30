#include "opencv2/opencv.hpp"
#include <omp.h>
#include "preprocessing.hpp"
#include "cpu.hpp"

using namespace cv;

Mat main_process(Mat, int*);

int main(int argh, char* argv[])
{
    VideoCapture cap(0);

    if(!cap.isOpened())
    {
        return -1;
    }

    Mat frame;
	Mat frame_out, frame_out2;
	Rect rect(140, 60, 360, 360);	//x,y, width, height
	int success[1] = {};
	//int *gpx;
	//int *gpy;
	//printf("start");
	//printf("finish");
    while(cap.read(frame))
    {
		printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());
		frame_out = main_process(frame, success);
		//printf("幅:%d, 高さ%d\n", frame.cols, frame.rows);
		//cvtColor(frame, frame_out, CV_BGR2GRAY);
		//paper = frame;
		//Mat dst = Mat::ones(frame.rows, frame.cols, CV_8U);
		/*if(!paper.data) {
			fprintf(stderr, "no image\n");
			return -1;
		}*/

		//resize(paper, dst, dst.size(), INTER_CUBIC);
		//paper = scale(paper, dst, (double)dst.rows/frame.rows, (double)dst.cols/frame.cols);
		rectangle(frame, rect, Scalar(0, 0, 255), 2, cv::LINE_4);

		imshow("Rock Paper Scissors Origin", frame);
		imshow("Rock Paper Scissors", frame_out);

		//frame_out2 = LaplacianFilter(frame_out, 3);
		//imshow("Rock Paper Scissorsw", frame_out2);

		if(success[0] == 5)		RockPaperScisors(0);
		if(success[0] == 2)		RockPaperScisors(2);

        const int key = waitKey(1);
        if(key == 'q'/*113*/){
            break;
        }
        else if(key == 's'/*115*/){
            imwrite("img_origin.png", frame);
			imwrite("img1.png", frame_out);
			imwrite("img2.png", frame_out);
        }
    }
    destroyAllWindows();
    return 0;
}

Mat main_process(Mat frame, int* suceess){
	Mat frame_gray;
	float x[2] = {};

	frame = GaussianFilter(frame, 1.3, 3);

	frame = RGBtoHSV(frame);

	frame_gray = GenerateMask(frame);

	frame_gray = opening(frame_gray, OPENING);
	frame_gray = closing(frame_gray, CLOSING);
	//cv::erode(frame_gray, frame_gray, cv::Mat(), cv::Point(-1,-1), 2); // 収縮
	//cv::dilate(frame_gray, frame_gray, cv::Mat(), cv::Point(-1,-1), 5); // 膨張

	CalcGravity(frame_gray, x);
	printf("gpx:%d, gpy:%d\n", (int)x[0], (int)x[1]);
	circle(frame_gray, Point((int)x[0], (int)x[1]), 10, 100, 3, 4);

	frame_gray = DetectFinger(frame_gray, x, suceess);

	return frame_gray;
}