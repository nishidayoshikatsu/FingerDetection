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

Mat RGBtoGray(Mat frame) {
	Mat frame_gray = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);

	for(int y = 0; y < frame.rows; y++){
                for(int x = 0; x < frame.cols; x++){
                        frame_gray.at<unsigned char>(y,x) =
								0.2126*frame.at<cv::Vec3b>(y,x)[2]
                                + 0.7152*frame.at<cv::Vec3b>(y,x)[1]
                                + 0.0722*frame.at<cv::Vec3b>(y,x)[0];
                }
    }

	return frame_gray;
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

  // prepare output
  Mat out = Mat::zeros(height, width, CV_8UC3);

  // prepare kernel
  int pad = floor(kernel_size / 2);
  int _x = 0, _y = 0;
  double kernel_sum = 0;
  
  // get gaussian kernel
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


  // filtering
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

	*p = gpx;
	p++;
	*p = gpy;
}

Mat LaplacianFilter(Mat img, int kernel_size){
	int height = img.rows;
	int width = img.cols;
	int channel = img.channels();

	Mat out = cv::Mat::zeros(height, width, CV_8UC1);

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
	//int cor[7][7];
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




			/*if (frame.at<unsigned char>(y, x) == 255 && sqrt(pow(y-pre_y,2) +pow(x-pre_x,2)) >= 10) {
				circle(frame, Point((int)x, (int)y), 3, 100, 3, 4);
				pre_y = y;
				pre_x = x;
				cnt++;
			}
			if(cnt == 3)	return frame;*/
		}
	}

	printf("finger count:%d\n", finger_cnt);

	return frame;

	/*int thresh = 200;
	int max_thresh = 255;

    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    Mat dst = Mat::zeros( frame.size(), CV_32FC1 );
    cornerHarris( frame, dst, blockSize, apertureSize, k );
    Mat dst_norm, dst_norm_scaled;
    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( dst_norm, dst_norm_scaled );
    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh )
            {
                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
            }
        }
    }*/
    //namedWindow( corners_window );
    //imshow( corners_window, dst_norm_scaled );


	//return dst_norm_scaled;
}

/*Mat CornerDetect(Mat frame) {
	int width = frame.cols;
	int height = frame.rows;

		sobely = np.array(((1, 2, 1),
						(0, 0, 0),
						(-1, -2, -1)), dtype=np.float32)

		sobelx = np.array(((1, 0, -1),
						(2, 0, -2),
						(1, 0, -1)), dtype=np.float32)

		# padding
		tmp = np.pad(gray, (1, 1), 'edge')

	Mat Ix = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);
	Mat Iy = Mat::zeros(Size(frame.cols, frame.rows), CV_8UC1);

	for (int y = 0; y < height; y++){
		for (int x = 0; x < width; x++){

		for y in range(H):
			for x in range(W):
				Ix[y, x] = np.mean(tmp[y : y  + 3, x : x + 3] * sobelx)
				Iy[y, x] = np.mean(tmp[y : y + 3, x : x + 3] * sobely)
			
		Ix2 = Ix ** 2
		Iy2 = Iy ** 2
		Ixy = Ix * Iy

		return Ix2, Iy2, Ixy
}*/



Mat Sikisou_tyuusyutu(Mat frame, Mat frame_out, int down1, int down2){
	int x, y;
	int d1 = down1, d2 = down2;
	int H[frame.rows][frame.cols];
	if(d1 > d2){
                int t = d1;
                d1 = d2;
                d2 = t;
        }
	for(y = 0; y < frame.rows; y++){
                for(x = 0; x < frame.cols; x++){
                        if(frame.at<cv::Vec3b>(y,x)[2] >= frame.at<cv::Vec3b>(y,x)[1] &&
                           frame.at<cv::Vec3b>(y,x)[2] >= frame.at<cv::Vec3b>(y,x)[0]){
                                if(frame.at<cv::Vec3b>(y,x)[1] > frame.at<cv::Vec3b>(y,x)[0]){
                                        H[y][x] = (int)floor(60.0*(frame.at<cv::Vec3b>(y,x)[1]-frame.at<cv::Vec3b>(y,x)[0])/
                                                (frame.at<cv::Vec3b>(y,x)[2]-frame.at<cv::Vec3b>(y,x)[0]));
                                }else{
                                        H[y][x] = (int)floor(60.0*(frame.at<cv::Vec3b>(y,x)[1]-frame.at<cv::Vec3b>(y,x)[0]*1.0)/
                                                (frame.at<cv::Vec3b>(y,x)[2]-frame.at<cv::Vec3b>(y,x)[1]));
                                }
                        }else if(frame.at<cv::Vec3b>(y,x)[1] >= frame.at<cv::Vec3b>(y,x)[2] &&
                           frame.at<cv::Vec3b>(y,x)[1] >= frame.at<cv::Vec3b>(y,x)[0]){
                                if(frame.at<cv::Vec3b>(y,x)[2] > frame.at<cv::Vec3b>(y,x)[0]){
                                        H[y][x] = (int)floor(60.0*(frame.at<cv::Vec3b>(y,x)[0]-frame.at<cv::Vec3b>(y,x)[2]*1.0)/
                                                (frame.at<cv::Vec3b>(y,x)[1]-frame.at<cv::Vec3b>(y,x)[0]));
                                }else{
                                        H[y][x] = (int)floor(60.0*(frame.at<cv::Vec3b>(y,x)[0]-frame.at<cv::Vec3b>(y,x)[2]*1.0)/
                                                (frame.at<cv::Vec3b>(y,x)[1]-frame.at<cv::Vec3b>(y,x)[2]));
                                }
                        }else if(frame.at<cv::Vec3b>(y,x)[0] >= frame.at<cv::Vec3b>(y,x)[2] &&
                           frame.at<cv::Vec3b>(y,x)[0] >= frame.at<cv::Vec3b>(y,x)[1]){
                                if(frame.at<cv::Vec3b>(y,x)[2] > frame.at<cv::Vec3b>(y,x)[1]){
                                        H[y][x] = (int)floor(60.0*(frame.at<cv::Vec3b>(y,x)[2]-frame.at<cv::Vec3b>(y,x)[1]*1.0)/
                                                (frame.at<cv::Vec3b>(y,x)[0]-frame.at<cv::Vec3b>(y,x)[1]));
                                }else{
                                        H[y][x] = (int)floor(60.0*(frame.at<cv::Vec3b>(y,x)[2]-frame.at<cv::Vec3b>(y,x)[1]*1.0)/
                                                (frame.at<cv::Vec3b>(y,x)[0]-frame.at<cv::Vec3b>(y,x)[2]));
                                }
                        }
                        if(H[y][x] < 0){
                                H[y][x] += 360;
                                if(H[y][x] < 0){
                                        H[y][x] = 360;
                                }
                        }
		}
	}
	for(y = 0; y < frame.rows; y++){
                for(x = 0; x < frame.cols; x++){
                        if(H[y][x] >= d1 && H[y][x] <= d2)	H[y][x] = 360;
                }
        }
	for(y = 0; y < frame.rows; y++){
                for(x = 0; x < frame.cols; x++){
                        if(H[y][x] == 360){
                                frame_out.at<cv::Vec3b>(y,x)[2] = 0;
				frame_out.at<cv::Vec3b>(y,x)[1] = 0;
				frame_out.at<cv::Vec3b>(y,x)[0] = 0;
			}else{
				frame_out.at<cv::Vec3b>(y,x)[2] = frame.at<cv::Vec3b>(y,x)[2];
				frame_out.at<cv::Vec3b>(y,x)[1] = frame.at<cv::Vec3b>(y,x)[1];
				frame_out.at<cv::Vec3b>(y,x)[0] = frame.at<cv::Vec3b>(y,x)[0];
			}
                }
    }

	return frame_out;
}

Mat kido_tyuusyutu(Mat frame, Mat frame_out, int down1, int down2){
	int x, y;
	int d1 = down1, d2 = down2;

	if(d1 > d2){
                int t = d1;
                d1 = d2;
                d2 = t;
    }

	for(y = 0; y < frame.rows; y++){
                for(x = 0; x < frame.cols; x++){
                        if(frame.at<unsigned char>(y,x) < d1){
                                frame_out.at<unsigned char>(y,x) = 0;
						}
                        else if(frame.at<unsigned char>(y,x) > d2){
                                frame_out.at<unsigned char>(y,x) = 0;
						}
                        else{
                                frame_out.at<unsigned char>(y,x) = frame.at<unsigned char>(y,x);
						}
				}
    }

	return frame_out;
}

Mat Nichika(Mat frame, int Shikiichi){
	int x,y;
	Mat frame_out = frame;

	for(y = 0; y < frame.rows; y++){
		for(x = 0; x < frame.cols; x++){
			if(frame.at<unsigned char>(y,x) > Shikiichi)
				frame_out.at<unsigned char>(y,x) = 255;
			else
				frame_out.at<unsigned char>(y,x) = 0;
		}
	}

	return frame_out;
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

Mat Filter_laplacian(Mat frame_gray, Mat frame_out){
	int x,y;
	int Oper1[9] = {-1,-1,-1,
			   -1, 8,-1,
			   -1,-1,-1};
	int p;

	//Mat frame_out = frame_gray;

	for(y = 0; y < frame_gray.rows; y++){
                for(x = 0; x < frame_gray.cols; x++){
			if(x > 0 && x < frame_gray.cols - 1 && y > 0 && y < frame_gray.rows - 1){
                        	p = (int)Filter(frame_gray, x, y, Oper1);
				if(p > 255){
					p = 255;
				}else if(p < 0){
					p = 0;
				}
			}else{
				p = 0;
			}
			frame_out.at<unsigned char>(y,x) = p;
                }
        }

	return frame_out;
}
