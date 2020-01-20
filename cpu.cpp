#include "opencv2/opencv.hpp"
#include "cpu.hpp"

using namespace cv;

Mat paper = imread("../img/pa.png", 1);
Mat rock = imread("../img/gu.png", 1);
Mat scissors = imread("../img/chiki.png", 1);

void RockPaperScisors(unsigned int player) {

	switch(player){
		case 0:
			printf("Rock\n");
			imshow("Rock", rock);
			break;
		case 1:
			printf("Paper\n");
			imshow("Paper", paper);
			break;
		case 2:
			printf("Scisors\n");
			break;
	}
}