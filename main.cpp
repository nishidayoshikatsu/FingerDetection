#include "opencv2/opencv.hpp"
#include "preprocessing.hpp"
#include "cpu.hpp"

using namespace cv;

Mat main_process(Mat);
Mat scale(Mat frame, Mat frame_out, double rx, double ry);

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
	//printf("start");
	//Mat paper = imread("./input/paper_sample.jpg", 0);
	//printf("finish");
    while(cap.read(frame))//無限ループ
    {
		frame_out = main_process(frame);
		printf("幅:%d, 高さ%d\n", frame.cols, frame.rows);
		//cvtColor(frame, frame_out, CV_BGR2GRAY);
		//paper = frame;
		//Mat dst = Mat::ones(frame.rows, frame.cols, CV_8U);

		//resize(paper, dst, dst.size(), INTER_CUBIC);
		//paper = scale(paper, dst, (double)dst.rows/frame.rows, (double)dst.cols/frame.cols);
		rectangle(frame, rect, Scalar(0, 0, 255), 3, cv::LINE_4);
		
		imshow("Rock Paper Scissors Origin", frame);
		imshow("Rock Paper Scissors", frame_out);
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
	frame = RGBtoHSV(frame);

	//Size ksize = cv::Size(5, 5);
    //GaussianBlur(frame, frame, ksize, 0);
	//cvtColor(frame, hsv_img, CV_BGR2HSV);
	//inRange(hsv_img, MIN_HSVCOLOR, MAX_HSVCOLOR, msk_img);
	frame_gray = GenerateMask(frame);

	return frame_gray;
}

Mat scale(Mat frame, Mat frame_out, double rx, double ry){
	frame = RGBtoGray(frame);
	//printf("%d\t%d\n", img_in->cols, img_in->rows);
	//printf("%lf\t%lf\n", img_in->cols * rx, img_in->rows * ry);

	int X, Y;

	for(int y=0; y<frame_out.rows; y++){
		for(int x=0; x<frame_out.cols; x++){
			X = (int)(x / rx);
			Y = (int)(y / ry);

			frame_out.at<unsigned char>(y,x) = frame.at<unsigned char>(Y,X);
		}
	}

	return frame_out;
}