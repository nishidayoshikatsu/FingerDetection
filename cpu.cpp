#include "opencv2/opencv.hpp"
#include "cpu.hpp"
#include <time.h>
#include <stdlib.h>

using namespace cv;

Mat paper = imread("../img/pa.png", 1);
Mat rock = imread("../img/gu.png", 1);
Mat scissors = imread("../img/choki.png", 1);
Mat win = imread("../img/win.png", 1);
Mat lose = imread("../img/lose.png", 1);

void RockPaperScisors(int player) {

	switch(player){
		case 0:
			printf("Rock\n");
			imshow("Rock", rock);
			waitKey(4000);	// wait
			CPU(0);
			cvDestroyWindow("Rock");
			break;
		case 1:
			printf("Paper\n");
			imshow("Paper", paper);
			waitKey(4000);	// wait
			CPU(1);
			cvDestroyWindow("Paper");
			break;
		case 2:
			printf("Scisors\n");
			imshow("Scissors", scissors);
			waitKey(4000);	// wait
			CPU(2);
			cvDestroyWindow("Scissors");
			break;
	}
}

void CPU(int player) {
	srand((unsigned)time(NULL));
	int cpu = rand()%3;
	int winorlose;

	switch(cpu) {
		case 0:
			printf("Rock\n");
			imshow("Rock_CPU", rock);
			waitKey(2000);	// wait
			judge(player, 0);
			cvDestroyWindow("Rock_CPU");
			break;
		case 1:
			printf("Paper\n");
			imshow("Paper_CPU", paper);
			waitKey(2000);	// wait
			judge(player, 1);
			cvDestroyWindow("Paper_CPU");
			break;
		case 2:
			printf("Scisors\n");
			imshow("Scissors_CPU", scissors);
			waitKey(2000);	// wait
			judge(player, 2);
			cvDestroyWindow("Scissors_CPU");
			break;
	}
}

void judge(int player, int cpu){
	if(player == 1){	// パー
		if(cpu == 0){	// グー
			imshow("PLAYER WIN!!", win);
			waitKey(2000);	// wait
			cvDestroyWindow("PLAYER WIN!!");
		} else if(cpu == 1){
			imshow("PLAYER WIN!!", win);
			waitKey(2000);	// wait
			cvDestroyWindow("PLAYER WIN!!");
		} else {
			imshow("PLAYER LOSE", lose);
			waitKey(2000);	// wait
			cvDestroyWindow("PLAYER LOSE");
		}
	} else if(player == 2){	// ちょき
		if(cpu == 0){	// グー
			imshow("PLAYER LOSE", lose);
			waitKey(2000);	// wait
			cvDestroyWindow("PLAYER LOSE");
		} else if(cpu == 1){
			imshow("PLAYER WIN!!", win);
			waitKey(2000);	// wait
			cvDestroyWindow("PLAYER WIN!!");
		} else {
			imshow("PLAYER WIN!!", win);
			waitKey(2000);	// wait
			cvDestroyWindow("PLAYER WIN!!");
		}
	}
}