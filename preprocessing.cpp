#include <math.h>
#include "opencv2/opencv.hpp"
#include "preprocessing.hpp"

using namespace cv;

Mat RGBtoHSV(Mat img){
	int width = img.cols;
	int height = img.rows;
	float r, g, b;
	float h, s, v;
	float _max, _min;

	Mat hsv = Mat::zeros(height, width, CV_32FC3);

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			r = (float)img.at<Vec3b>(y, x)[2] / 255;
			g = (float)img.at<Vec3b>(y, x)[1] / 255;
			b = (float)img.at<Vec3b>(y, x)[0] / 255;

			_max = fmax(r, fmax(g, b));
			_min = fmin(r, fmin(g, b));

			if(_max == _min){
				h = 0;
			} else if (_min == b) {
				h = 60 * (g - r) / (_max - _min) + 60;
			} else if (_min == r) {
				h = 60 * (b - g) / (_max - _min) + 180;
			} else if (_min == g) {
				h = 60 * (r - b) / (_max - _min) + 300;
			}

			s = _max - _min;

			v = _max;

			hsv.at<Vec3f>(y, x)[0] = h;
			hsv.at<Vec3f>(y, x)[1] = s;
			hsv.at<Vec3f>(y, x)[2] = v;
		}
	}

	return hsv;
}

Mat GenerateMask(Mat frame){
	//#define MIN_HSVCOLOR cv::Scalar(0, 60, 80)
	//#define MAX_HSVCOLOR cv::Scalar(10, 160, 240)
	Mat frame_out = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);

	int width = frame.cols;
	int height = frame.rows;

	float h, s, v;

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			if(y >= 60 && y<=420 && x >=140 && x <=500){
				h = frame.at<Vec3f>(y, x)[0];
				s = frame.at<Vec3f>(y, x)[1];
				v = frame.at<Vec3f>(y, x)[2];

				//if(h >= 0 && h <= 10 && s >= 60 && s <= 160 && v >= 80 && v <= 240){
				if(h <= 50){
					frame_out.at<unsigned char>(y, x) = 255;
				}
			}
		}
	}

	return frame_out;
}

Mat GaussianFilter(Mat img, double sigma, int kernel_size){
	int height = img.rows;
	int width = img.cols;
	int channel = img.channels();

	Mat out = Mat::zeros(height, width, CV_8UC3);

	int pad = floor(kernel_size / 2);
	int _x = 0, _y = 0;
	double kernel_sum = 0;

	float kernel[kernel_size][kernel_size];

	for (int y = 0; y < kernel_size; y++){
		for (int x = 0; x < kernel_size; x++){
			_y = y - pad;
			_x = x - pad;
			kernel[y][x] = 1 / (2 * M_PI * sigma * sigma) * exp( - (_x * _x + _y * _y) / (2 * sigma * sigma));
			kernel_sum += kernel[y][x];
		}
	}

	for (int y = 0; y < kernel_size; y++){
		for (int x = 0; x < kernel_size; x++){
			kernel[y][x] /= kernel_sum;
		}
	}


	double v = 0;

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			for (int c = 0; c < channel; c++){
				v = 0;
				for (int dy = -pad; dy < pad + 1; dy++){
					for (int dx = -pad; dx < pad + 1; dx++){
						if (((x + dx) >= 0) && ((y + dy) >= 0)){
							v += (double)img.at<cv::Vec3b>(y + dy, x + dx)[c] * kernel[dy + pad][dx + pad];
						}
					}
				}
				out.at<Vec3b>(y, x)[c] = v;
			}
		}
	}

	return out;
}

void CalcGravity(Mat frame, float *p) {
	int sx = 0;
	int sy = 0;
	int mm = 0;
	int width = frame.cols;
	int height = frame.rows;

	int x, y;
	float gpx, gpy;

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			if (frame.at<unsigned char>(y, x) == 255) {
				sx += (float)x;
				sy += (float)y;
				++mm;
			}
		}
	}

	gpx = (float)sx/(float)mm;
	gpy = (float)sy/(float)mm;

	*p++ = gpx;
	*p = gpy;
}

Mat LaplacianFilter(Mat img, int kernel_size){
	int height = img.rows;
	int width = img.cols;
	int channel = img.channels();

	Mat out = Mat::zeros(height, width, CV_8UC1);

	double kernel[kernel_size][kernel_size] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};

	int pad = floor(kernel_size / 2);

	double v = 0;

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){
			v = 0;
			for (int dy = -pad; dy < pad + 1; dy++){
				for (int dx = -pad; dx < pad + 1; dx++){
					if (((y + dy) >= 0) && (( x + dx) >= 0) && ((y + dy) < height) && ((x + dx) < width)){
            			v += img.at<uchar>(y + dy, x + dx) * kernel[dy + pad][dx + pad];
					}
				}
			}
			v = fmax(v, 0);
			v = fmin(v, 255);
			out.at<uchar>(y, x) = (uchar)v;
		}
	}

	return out;
}

Mat DetectFinger(Mat frame, float* gp) {
	int height = frame.rows;
	int width = frame.cols;
	int cnt = 0;
	int pre_x = 0, pre_y = 0;
	int finger_cnt = 0;

	for (int y = 0+3; y < height-3; y++){
		for (int x = 0+3; x < width-3; x++){
			int cor[7][7] = {
				{frame.at<unsigned char>(y-3, x-3), frame.at<unsigned char>(y-3, x-2), frame.at<unsigned char>(y-3, x-1), frame.at<unsigned char>(y-3, x), frame.at<unsigned char>(y-3, x+1), frame.at<unsigned char>(y-3, x+2), frame.at<unsigned char>(y-3, x+3)},
				{frame.at<unsigned char>(y-2, x-3), frame.at<unsigned char>(y-2, x-2), frame.at<unsigned char>(y-2, x-1), frame.at<unsigned char>(y-2, x), frame.at<unsigned char>(y-2, x+1), frame.at<unsigned char>(y-2, x+2), frame.at<unsigned char>(y-2, x+3)},
				{frame.at<unsigned char>(y-1, x-3), frame.at<unsigned char>(y-1, x-2), frame.at<unsigned char>(y-1, x-1), frame.at<unsigned char>(y-1, x), frame.at<unsigned char>(y-1, x+1), frame.at<unsigned char>(y-1, x+2), frame.at<unsigned char>(y-1, x+3)},
				{frame.at<unsigned char>(y, x-3), frame.at<unsigned char>(y, x-2), frame.at<unsigned char>(y, x-1), frame.at<unsigned char>(y, x), frame.at<unsigned char>(y, x+1), frame.at<unsigned char>(y, x+2), frame.at<unsigned char>(y, x+3)},
				{frame.at<unsigned char>(y+1, x-3), frame.at<unsigned char>(y+1, x-2), frame.at<unsigned char>(y+1, x-1), frame.at<unsigned char>(y+1, x), frame.at<unsigned char>(y+1, x+1), frame.at<unsigned char>(y+1, x+2), frame.at<unsigned char>(y+1, x+3)},
				{frame.at<unsigned char>(y+2, x-3), frame.at<unsigned char>(y+2, x-2), frame.at<unsigned char>(y+2, x-1), frame.at<unsigned char>(y+2, x), frame.at<unsigned char>(y+2, x+1), frame.at<unsigned char>(y+2, x+2), frame.at<unsigned char>(y+2, x+3)},
				{frame.at<unsigned char>(y+3, x-3), frame.at<unsigned char>(y+3, x-2), frame.at<unsigned char>(y+3, x-1), frame.at<unsigned char>(y+3, x), frame.at<unsigned char>(y+3, x+1), frame.at<unsigned char>(y+3, x+2), frame.at<unsigned char>(y+3, x+3)}
			};

			if(cor[3][3] == 255){

				for(int i=0;i<3;i++){
					for(int j=0;j<7;j++){
						//if(i == 3 && j == 3)	continue;
						if(cor[i][j] == 0){
							cnt++;
						}
					}
				}

				if(cnt >= 21 && sqrt(pow(y-pre_y,2) +pow(x-pre_x,2)) >= 50){
					//printf("%f\n", sqrt(pow(y-*gpy,2) +pow(x-*gpx,2)));
					if(sqrt(pow(y-*(gp+1),2) +pow(x-*gp,2)) >= 150){
						circle(frame, Point((int)x, (int)y), 3, 100, 3, 4);
						line(frame, Point((int)*gp, (int)*(gp+1)), Point((int)x, (int)y), 100, 3, 4); 
						finger_cnt++;
						pre_y = y;
						pre_x = x;
					}
				}
				cnt = 0;
			}
		}
	}

	printf("finger count:%d\n", finger_cnt);

	return frame;
}

int Filter(Mat frame_gray, int x, int y, int Oper[]){
	return  Oper[0]*frame_gray.at<unsigned char>(y-1,x-1) + Oper[1]*frame_gray.at<unsigned char>(y-1,x) + Oper[2]*frame_gray.at<unsigned char>(y-1,x+1) +
                Oper[3]*frame_gray.at<unsigned char>(y,x-1)   + Oper[4]*frame_gray.at<unsigned char>(y,x)   + Oper[5]*frame_gray.at<unsigned char>(y,x+1) +
                Oper[6]*frame_gray.at<unsigned char>(y+1,x-1) + Oper[7]*frame_gray.at<unsigned char>(y+1,x) + Oper[8]*frame_gray.at<unsigned char>(y+1,x+1);
}

Mat sikaku_kuro(Mat frame_gray, int x, int y){
	frame_gray.at<unsigned char>(y-1,x-1) = 0; frame_gray.at<unsigned char>(y-1,x) = 0; frame_gray.at<unsigned char>(y-1,x+1) = 0;
        frame_gray.at<unsigned char>(y,x-1) = 0;   frame_gray.at<unsigned char>(y,x) = 0;   frame_gray.at<unsigned char>(y,x+1) = 0;
        frame_gray.at<unsigned char>(y+1,x-1) = 0; frame_gray.at<unsigned char>(y+1,x) = 0; frame_gray.at<unsigned char>(y+1,x+1) = 0;
	return frame_gray;
}

Mat sikaku_siro(Mat frame_gray, int x, int y){
	frame_gray.at<unsigned char>(y-1,x-1) = 255; frame_gray.at<unsigned char>(y-1,x) = 255; frame_gray.at<unsigned char>(y-1,x+1) = 255;
        frame_gray.at<unsigned char>(y,x-1) = 255;   frame_gray.at<unsigned char>(y,x) = 255;   frame_gray.at<unsigned char>(y,x+1) = 255;
        frame_gray.at<unsigned char>(y+1,x-1) = 255; frame_gray.at<unsigned char>(y+1,x) = 255; frame_gray.at<unsigned char>(y+1,x+1) = 255;
	return frame_gray;
}

Mat syuusyuku(Mat frame_gray, int syori_num){
	int x,y;
        int i = 0;
        int edge[frame_gray.cols * frame_gray.rows / 2][2];
        int edge_num = 0;
        int Oper1[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
        int p = 0;

	Mat frame_out = frame_gray;

	do{
                for(y = 0; y < frame_gray.rows; y++){
                        for(x = 0; x < frame_gray.cols; x++){
                                if(x > 10 && x < (frame_gray.cols - 11) && y > 10 && y < (frame_gray.rows - 11)){
                                        p = Filter(frame_out, x, y, Oper1);
                                        if(frame_out.at<unsigned char>(y,x) == 255 && p != 0){
                                                edge[edge_num][0] = x;
                                                edge[edge_num][1] = y;
                                                edge_num++;
                                        }
                                }else{
                                        frame_out.at<unsigned char>(y,x) = 0;
                                }
                        }
                }
                for(x = 0; x < edge_num; x++){
                        frame_out = sikaku_kuro(frame_out, edge[x][0], edge[x][1]);
                }
        edge_num = 0;
        i++;
        }while(i < syori_num);

	return frame_out;
}
Mat boutyou(Mat frame_gray, int syori_num){
	int x,y;
        int i = 0;
        int edge[frame_gray.cols * frame_gray.rows / 2][2];
        int edge_num = 0;
        int Oper1[9] = {-1,-1,-1,
						-1,8,-1,
						-1,-1,-1};
        int p = 0;

	Mat frame_out = frame_gray;

	do{
                for(y = 0; y < frame_gray.rows; y++){
                        for(x = 0; x < frame_gray.cols; x++){
                                if(x > 10 && x < (frame_gray.cols - 11) && y > 10 && y < (frame_gray.rows - 11)){
                                        p = Filter(frame_out, x, y, Oper1);
                                        if(frame_out.at<unsigned char>(y,x) == 0 && p != 0){
                                                edge[edge_num][0] = x;
                                                edge[edge_num][1] = y;
                                                edge_num++;
                                        }
                                }else{
                                        frame_out.at<unsigned char>(y,x) = 0;
                                }
                        }
                }
                for(x = 0; x < edge_num; x++){
                        frame_out = sikaku_siro(frame_out, edge[x][0], edge[x][1]);
                }
        edge_num = 0;
        i++;
        }while(i < syori_num);

	return frame_out;
}

Mat opening(Mat frame_gray, int syori_num){
	frame_gray = syuusyuku(frame_gray, syori_num);
	frame_gray = boutyou(frame_gray, syori_num);

	return frame_gray;
}
Mat closing(Mat frame_gray, int syori_num){
	frame_gray = boutyou(frame_gray, syori_num);
	frame_gray = syuusyuku(frame_gray, syori_num);

	return frame_gray;
}