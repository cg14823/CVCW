/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <math.h>
#define PI 3.14159265

using namespace cv;
struct Sobel_return{
  Mat sobelImagex;
  Mat sobelImagey;
  Mat magnitude;
  Mat direction;
  Mat directionRads;
};

struct Circle{
  float x;
  float y;
  float r;
};

/** Function Headers */
void detectAndDisplay( Mat frame );
void houghCircleCT (Mat imageMag, Mat imageDire, int minR, int maxR,int step,std::vector<Circle>* circles);
Sobel_return sobel(Mat image);
Mat convolution (Mat input, Mat kernel);

/** Global variables */
String cascade_name = "dart/dartcascade/cascade.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
  std::vector<Circle> circles;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;
  Mat blurred;
  //GaussianBlur(frame_gray,blurred,Size(7,7),0,0);
	Sobel_return x = sobel(frame_gray);
  threshold(x.magnitude,x.magnitude,80,255,THRESH_BINARY);
	houghCircleCT(x.magnitude,x.directionRads,40,120,2,&circles);

  namedWindow("direction",CV_WINDOW_AUTOSIZE);
  imshow("direction",x.direction);
  namedWindow("magthres",CV_WINDOW_AUTOSIZE);
  imshow("magthres",x.magnitude);
  waitKey(0);

  std::vector<Circle> finalCircles = circles;

  for( int i =0; i<circles.size();i++){
    Circle u = circles[i];
    Circle uf = finalCircles[i];
    if (uf.r < 0) continue;

    for(int jj =0; jj<circles.size();jj++){
      if( jj == i) continue;
      else{
        Circle q = circles[jj];
        if((abs(q.x -u.x) < 15) && (abs(q.y -u.y) < 15) && (abs(q.r -u.r) < 30)){
          if( q.r > u.r){
            uf.y = -1;
          }
          else{
            finalCircles[jj].r =-1;
          }
        }
      }
    }
  }

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 255, 0, 0 ), 2);
	}

  int count =0;
  for ( int i = 0; i < finalCircles.size();i++){
    if(finalCircles[i].r > 0){
      circle(frame,Point(finalCircles[i].x, finalCircles[i].y),finalCircles[i].r,Scalar( 0, 255, 0 ), 2);
      count++;
    }
  }
  std::cout << circles.size() << std::endl;
  std::cout << count << std::endl;

  namedWindow("image",CV_WINDOW_AUTOSIZE);
  imshow("image",frame);
  waitKey(0);
}

void houghCircleCT (Mat imageMag, Mat imageDire, int minR, int maxR, int step, std::vector<Circle>* circles){
  int hthresh = 12;
  int zsize = (maxR - minR)/step;
  int houghSizes []= {imageMag.rows,imageMag.cols,zsize};
  Mat houghSpace = Mat(3,houghSizes,CV_64F, Scalar(0));

	for (int ii = 0; ii < imageMag.rows; ii++){
    for (int jj = 0; jj < imageMag.cols; jj++){
      for(int ri = 0; ri < zsize; ri += step){
        if(imageMag.at<uchar>(ii,jj) == 255){
          int r = ri*step + minR;
          int x0p = (int)(ii + r * sin(imageDire.at<double>(ii,jj)));
          int y0p = (int)(jj + r * cos(imageDire.at<double>(ii,jj)));
          int x0m = (int)(ii - r * sin(imageDire.at<double>(ii,jj)));
          int y0m = (int)(jj - r * cos(imageDire.at<double>(ii,jj)));
          if((x0p >= 0 && x0p < imageMag.rows) && (y0p >= 0 && y0p < imageMag.cols)){
            houghSpace.at<double>(x0p,y0p,ri) += 1;
          }
          if((x0m >= 0 && x0m < imageMag.rows) && (y0m >= 0 && y0m < imageMag.cols)){
            houghSpace.at<double>(x0m,y0m,ri) += 1;
          }
        }
      }
    }
  }

    int dims[] = {imageMag.rows,imageMag.cols};
    Mat houghImage = Mat(2,dims,CV_8U, Scalar(0));

    for (int ii = 0; ii < imageMag.rows; ii++){
      for (int jj = 0; jj < imageMag.cols; jj++){
        for(int ri = 0; ri < zsize; ri += step){
          houghImage.at<uchar>(ii,jj) += (uchar)  houghSpace.at<double>(ii,jj,ri);
        }
      }
    }

    for (int ii = 0; ii < imageMag.rows; ii++){
     for (int jj = 0; jj <imageMag.cols; jj++){
       for(int ri = 0; ri < zsize; ri++){
         if(houghSpace.at<double>(ii,jj,ri) > hthresh){
           Circle c;
           c.x = jj;
           c.y = ii;
           c.r = ri*step+minR;
           circles->push_back(c);
         }
       }
     }
   }
    //houghImage *= 50;
    namedWindow("hough",CV_WINDOW_AUTOSIZE);
    imshow("hough",houghImage);
}

Sobel_return sobel(Mat image){
  const double normalizemag =  255/sqrt(130050);
  const double normalizedire = 255/360;
  int kernelx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  int kernely[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
  Mat matrixX = Mat(3,3, CV_32S,kernelx);
  Mat matrixY = Mat(3,3, CV_32S,kernely);
  Mat directionRads = Mat(image.rows, image.cols,CV_64F);
  Mat sobelImagex = convolution(image,matrixX);
  Mat sobelImagey = convolution(image,matrixY);
  Mat magnitude = image.clone();
  Mat direction = image.clone();

  for (int i = 0; i < image.rows; i++){
    for(int j = 0; j <image.cols; j++){
      double nnormX = (sobelImagex.at<uchar>(i,j) -127.5) *8;
      double nnormY = (sobelImagey.at<uchar>(i,j) -127.5) *8;
      magnitude.at<uchar>(i,j) = (uchar)(sqrt(pow(nnormX,2) +pow(nnormY,2)) *normalizemag);
      direction.at<uchar>(i,j) = (uchar) ((atan2(nnormY,nnormX) * (180/PI)));
      directionRads.at<double>(i,j) = atan2(nnormY,nnormX);
    }
  }
  Sobel_return val;
  val.sobelImagex = sobelImagex;
  val.sobelImagey = sobelImagey;
  val.direction = direction;
  val.magnitude = magnitude;

  val.directionRads = directionRads;
  return val;

}

Mat convolution (Mat input, Mat kernel){

  Mat output = input.clone();

  int kernelRadiusX = ( kernel.size[0] - 1 ) / 2;
  int kernelRadiusY = ( kernel.size[1] - 1 ) / 2;

  Mat paddedInput;
  copyMakeBorder( input, paddedInput,
    kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
    cv::BORDER_REPLICATE );

  for ( int i = 0; i < input.rows; i++ )
  {
    for( int j = 0; j < input.cols; j++ )
    {
      double sum = 0.0;
      for( int m = -kernelRadiusX; m <= kernelRadiusX; m++ )
      {
        for( int n = -kernelRadiusY; n <= kernelRadiusY; n++ )
        {
          // find the correct indices we are using
          int imagex = i + m + kernelRadiusX;
          int imagey = j + n + kernelRadiusY;
          int kernelx = m + kernelRadiusX;
          int kernely = n + kernelRadiusY;

          // get the values from the padded image and the kernel
          int imageval = ( int ) paddedInput.at<uchar>( imagex, imagey );
          int kernalval = kernel.at<int>( kernelx, kernely );

          // do the multiplication
          sum += imageval * kernalval;
        }
      }
      // set the output value as the sum of the convolution
      output.at<uchar>(i, j) = (uchar) (sum/8 +255/2);
    }
  }

  return output;

}
