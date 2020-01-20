#include "opencv2/opencv.hpp"
#include "cpu.hpp"

using namespace cv;

void RockPaperScisors(unsigned int player) {
	Mat paper = imread("./input/paper_sample.jpg", 1);

	switch(player){
		case 0:
			printf("Rock\n");
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