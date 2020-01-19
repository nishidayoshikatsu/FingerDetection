#include "opencv2/opencv.hpp"
#include "preprocessing.hpp"

using namespace cv;

int main(int argh, char* argv[])
{
    VideoCapture cap(0);//デバイスのオープン

    if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
    {
        return -1;
    }

    Mat frame; //取得したフレーム
	Mat frame_out;
    while(cap.read(frame))//無限ループ
    {
		frame_out = main_process(frame);
		//cvtColor(frame, frame_out, CV_BGR2GRAY);

		imshow("Rock Paper Scissors", frame_out);

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

	frame = Sikisou_tyuusyutu(frame, frame_out, SIKISOU1 , SIKISOU2);
	frame_gray = RGBtoGray(frame);

	frame_gray2 = frame_gray;

	frame_gray = kido_tyuusyutu(frame_gray, frame_gray2, KIDO1, KIDO2);

	//frame_gray = Nichika(frame_gray2, NICHIKA);

	//frame_gray = opening(frame_gray, OPENING);
	//frame_gray = closing(frame_gray, CLOSING);
	frame_gray = Filter_laplacian(frame_gray, frame_gray2);

	return frame_gray;
}