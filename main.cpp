#include "opencv2/opencv.hpp"

using namespace cv;

int main(int argh, char* argv[])
{
    VideoCapture cap(0);//デバイスのオープン

    if(!cap.isOpened())//カメラデバイスが正常にオープンしたか確認．
    {
        return -1;
    }

    Mat frame; //取得したフレーム
    while(cap.read(frame))//無限ループ
    {
		//frame = masking(frame);
		line(frame, Point(100, 300), Point(400, 300), Scalar(255,0,0), 10, CV_AA);
        imshow("win", frame);//画像を表示．
        const int key = waitKey(1);
        if(key == 'q'/*113*/){
            break;
        }
        else if(key == 's'/*115*/){
            imwrite("img.png", frame);
        }
    }
    cv::destroyAllWindows();
    return 0;
}