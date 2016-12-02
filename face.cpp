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
  int plot;
};

struct Bbox{
  Rect d;
  bool plot;
};

/** Function Headers */
void detectAndDisplay( Mat frame );
void houghCircleCT (Mat imageMag, Mat imageDire, int minR, int maxR,int step,std::vector<Circle>* circles);
void houghLineCT (Mat imageMag,Mat frame);
Sobel_return sobel(Mat image);
Mat convolution (Mat input, Mat kernel);
bool colourdetection(Mat image);

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
	imwrite( "detectedMERGED.jpg", frame );

	return 0;
}
bool isGray(Mat img){
  //Checks if RGB image is grayscale
  Mat dst;
  Mat bgr[3];
  split( img, bgr );
  absdiff( bgr[0], bgr[1], dst );
  double min, max;
  minMaxLoc(dst, &min, &max);
  if(max >45.0 ) return false;
  else return true;
}

bool overlap (Rect u, Rect q){
  //Checks if given rectangles overlap
  int qx2 = q.x + q.width;
  int qy2 = q.y + q.height;
  int ux2 = u.x + u.width;
  int uy2 = u.y + u.height;
  if (((q.x > u.x && q.x < ux2) || (qx2 > u.x && qx2 < ux2)) && ((q.y > u.y && q.y < uy2) || (qy2 > u.y && qy2 < uy2)) ) return true;
  else return false;
}

int concentric (Rect u, Rect q){
  //Checks if given rectangle is inside the other given rectangle.
  int qx2 = q.x + q.width;
  int qy2 = q.y + q.height;
  int ux2 = u.x + u.width;
  int uy2 = u.y + u.height;
  if ((q.x > u.x && qx2 < ux2) && (q.y > u.y && qy2 < uy2)) return 1;
  else if ((u.x > q.x && ux2 < qx2) && (u.y > q.y && uy2 < qy2)) return 2;
  else return 0;
}

bool closeCenter(Rect u, Rect q){
  //Checks if centres are "close"
  if(abs(u.x +u.width/2 -q.x -q.width/2)>15 && abs(u.y +u.height/2 -q.height -q.height/2)>15) return true;
  else return false;
}

bool sameRec(Rect u, Rect q){
  if(u.x == q.x && u.y ==q.y && u.height == q.height && u.width == q.width) return true;
  else return false;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
  std::vector<Rect> darts;
  std::vector<Rect> finaldarts;
  std::vector<Rect> finalrects;
  std::vector<Circle> circles;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	//std::cout << faces.size() << std::endl;
  Mat blurred;
  //GaussianBlur(frame_gray,blurred,Size(7,7),0,0);
	Sobel_return x = sobel(frame_gray);
  threshold(x.magnitude,x.magnitude,80,255,THRESH_BINARY);
	houghCircleCT(x.magnitude,x.directionRads,40,120,2,&circles);


  std::vector<Circle> finalCircles = circles;
  // remove circles that are clos to each other in a 30x30 box and in a radius of 30
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

  if(isGray(frame)){
    //std::cout << "GRAY" << std::endl;
    finalrects = faces;
  }
  else{
  // remove boxes that dont much the color pattern for viola-jones detector
    for(int i =0; i<faces.size();i++){
      Rect dart = faces[i];
      Mat onlyDartboard = frame(dart);
      bool valid =colourdetection(onlyDartboard);
      //std::cout << "VALID: "<<valid << std::endl;
      if (valid){
        finalrects.push_back(dart);
      }
    }
  }

  int count =0;
  //define final points if square inside circle
  for ( int i = 0; i < finalCircles.size();i++){
    if(finalCircles[i].r > 0){
      Circle u = finalCircles[i];
      count++;
      //if sqaure inside circle define as final point
      for(int j =0; j< finalrects.size();j++){
        int dx = finalrects[j].x -u.x;
        int dy = finalrects[j].y -u.y;
        float tlD = sqrt(pow(dx,2)+pow(dy,2));
        if(tlD <= u.r*1.2){
          //top left in check top right
          dx += finalrects[j].width;
          tlD = sqrt(pow(dx,2)+pow(dy,2));
          if(tlD <= u.r*1.2){
            //top right in check bottom right
            dy += finalrects[j].height;
            tlD = sqrt(pow(dx,2)+pow(dy,2));
            if(tlD <= u.r*1.2){
              //bottom right in check bottom left
              dy -= finalrects[j].width;
              tlD = sqrt(pow(dx,2)+pow(dy,2));
              if(tlD <= u.r*1.2){
                finalCircles[i].x = (int)(u.x + finalrects[j].x +(finalrects[j].width/2))/2;
                finalCircles[i].y = (int)(u.y + finalrects[j].y +(finalrects[j].height/2))/2;
                finalCircles[i].plot = 1;
              }
            }
          }
        }
      }
    }
  }

  //if circle inside circle average center and radius and defien as final point
  for(int i =0; i<finalCircles.size();i++){
    if(finalCircles[i].r>0){
        for(int j = 0;j<finalCircles.size();j++){
          if(j != i && finalCircles[j].r >0){
            Circle q = finalCircles[j];
            // distance between 2 center subtract from alrge r compare to small r
            int dx = finalCircles[i].x -q.x;
            int dy = finalCircles[i].y -q.y;
            float dcenters = sqrt(pow(dx,2)+pow(dy,2));
            if (finalCircles[i].r-dcenters >= q.r){
              finalCircles[i].x =  (int)(finalCircles[i].x +q.x)/2;
              finalCircles[i].y =  (int)(finalCircles[i].y +q.y)/2;
              finalCircles[i].r =  (finalCircles[i].r +q.r)/2;
              finalCircles[i].plot = 1;
            }
          }
        }
    }
  }
//get all circles to change to rectangles
  for(int i =0; i<finalCircles.size();i++){
    if(finalCircles[i].plot ==1){
      int x = (int)finalCircles[i].x -finalCircles[i].r;
      int y = (int)finalCircles[i].y -finalCircles[i].r;
      Rect dart = Rect(x,y, (int) finalCircles[i].r *2, (int) finalCircles[i].r*2);
      darts.push_back(dart);
      //rectangle(frame, Point(dart.x, dart.y), Point(dart.x + dart.width, dart.y + dart.height), Scalar( 0, 0, 255 ), 2);
    }
  }
  //check if final detections match color pattern
  if(isGray(frame)){
    //std::cout << "GRAY" << std::endl;
    finaldarts = darts;
  }
  else{
    for(int i =0; i<darts.size();i++){
      Rect dart = darts[i];
      Mat onlyDartboard = frame(dart);
      bool valid =colourdetection(onlyDartboard);
      if (valid){
        finaldarts.push_back(dart);
      }

    }
  }



  //std::cout << finaldarts.size() << std::endl; //bluebox
  //std::cout << finalrects.size() << std::endl; //redbox
  /*----------------------------MERGE STUFFFFFF ---------------------------------------*/
  std::vector<Bbox> finalPlot;
  int sized = finaldarts.size();
  for(int i = 0; i<finaldarts.size();i++){
    Bbox b;
    b.d =finaldarts[i];
    b.plot = true;
    finalPlot.push_back(b);
  }
  for(int i = 0; i<finalrects.size();i++){
    Bbox b;
    b.d =finalrects[i];
    b.plot = true;
    finalPlot.push_back(b);
  }

  //if there is a red box inside a bluebox do not plot redbox or if blue inside red only plot blue
  for(int i = 0; i<finaldarts.size();i++){
    Rect q = finalPlot[i].d;
    for(int j = 0; j<finalrects.size();j++){
      Rect u = finalPlot[sized+j].d;
      if(concentric(q,u) > 0) finalPlot[sized+j].plot =false;
    }
  }

  //if various redboxes overlap average to only have on
  int tot_overlaps;
  do{
    tot_overlaps =0;
    for(int j = 0; j<finalrects.size();j++){
      Rect u = finalrects[j];
      Rect av =u;
      int overlaps =1; //overlaps with itsselc
      for(int i = 0; i<finalrects.size();i++){
        if(i != j){
          Rect q =finalrects[i];
          if(sameRec(u,q) && finalPlot[sized +j].plot) finalPlot[sized +i].plot =false;
          else if(overlap(u,q)){
            av.x += q.x;
            av.y += q.y;
            av.width += q.width;
            av.height += q.height;
            overlaps++;
            tot_overlaps++;
          }
        }
      }
      av.x /= overlaps;
      av.y /= overlaps;
      av.height /= overlaps;
      av.width /= overlaps;
      finalPlot[sized+j].d =av;
    }

    for(int j = 0; j<finalrects.size();j++){
      finalrects[j] =finalPlot[sized +j].d;
    }
  }while(tot_overlaps > 0);

  // if various blueboxes overlap average them out

  do{
    tot_overlaps=0;
    for(int j = 0; j<finaldarts.size();j++){
      Rect u = finaldarts[j];
      Rect av =u;
      int overlaps =1; //overlaps with itsselc
      for(int i = 0; i<finaldarts.size();i++){
        if(i != j){
          Rect q =finaldarts[i];
          if(sameRec(u,q) && finalPlot[sized +j].plot) finalPlot[i].plot =false;
          else if(overlap(u,q)){
            av.x += q.x;
            av.y += q.y;
            av.width += q.width;
            av.height += q.height;
            overlaps++;
            tot_overlaps++;
          }
        }
      }
      av.x /= overlaps;
      av.y /= overlaps;
      av.height /= overlaps;
      av.width /= overlaps;
      finalPlot[j].d =av;
    }

    for(int j = 0; j<finaldarts.size();j++){
      finaldarts[j] =finalPlot[j].d;
    }
  }while(tot_overlaps > 0);

  // if a red box overlaps with a blue box ignor it

  for(int i= 0; i<finaldarts.size();i++){
    if(finalPlot[i].plot){
      Rect q =finaldarts[i];
      for(int j=0;j<finalrects.size();j++){
        Rect u =finalrects[j];
        if(finalPlot[sized+j].plot && overlap(q,u)) finalPlot[sized+j].plot = false;
      }
    }
  }
  //std::cout << finalPlot.size() << std::endl;

  //draw final boundries
  int countF =0;
  for(int i =0;i<finalPlot.size();i++){
    if(finalPlot[i].plot){
      countF++;
      Rect dart = finalPlot[i].d;
      Scalar c = Scalar(255,0,0);
      if(i >=sized) c = Scalar(0,0,255);
      rectangle(frame, Point(dart.x, dart.y), Point(dart.x + dart.width, dart.y + dart.height),c, 2);
    }
  }
  std::cout<<"N# Dartoards: "<<countF<<std::endl;
}

bool colourdetection(Mat image){
  int black =0;
  int white =0;
  int red =0;
  for(int ii =0; ii< image.rows; ii++){
    for(int jj=0; jj< image.cols; jj++){
      int r = (int)image.at<Vec3b>(ii,jj)[2];
      int g= (int)image.at<Vec3b>(ii,jj)[1];
      int b= (int)image.at<Vec3b>(ii,jj)[0];
      if (r < (g +20) && r > (g-20) && b < (g +20) && b > (g-20) && r < 80) black++;
      if(r > 140 && g < 70 && b < 70) red++;
      if(r > 150 && g> 140 && b > 60) white++;
    }
  }
  //std::cout << "black :"<<black << std::endl;
  //std::cout << "red :"<<red << std::endl;
  //std::cout << "white :"<<white << std::endl;

  /*namedWindow("someimage",CV_WINDOW_AUTOSIZE);
  imshow("someimage",image);
  waitKey(0);*/
  if(red == 0 || white == 0 || black ==0) return false;
  if( red <20) return false;
  if (red > black || red >white) return false;
  return true;
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
           c.plot = 0;
           circles->push_back(c);
         }
       }
     }
   }
}

void houghLineCT (Mat imageMag,Mat frame){
  std::vector<Vec4i> lines;
  HoughLinesP(imageMag,lines,1,PI/180,50,10,1);
  std::vector<Vec4f> newLines;
  Mat newFrame = frame.clone();
  int range = 4;
  for( int i = 0; i < lines.size(); i++ )
  {
    float x1 = lines[i][0], x2 = lines[i][2], y1 = lines[i][1],y2 = lines[i][3];
    if ((x1 > x2+range || x1 < x2-range) && (y1 > y2+range || y1 < y2-range)){
      newLines.push_back(lines[i]);
    }
  }
  for( int i = 0; i < lines.size(); i++ )
  {
      float x1 = lines[i][0], x2 = lines[i][2], y1 = lines[i][1],y2 = lines[i][3];
      Point pt1, pt2;
      pt1.x = (int) x1;
      pt1.y = (int) y1;
      pt2.x = (int) x2;
      pt2.y = (int) y2;
      line( frame, pt1, pt2, Scalar(0,255,0), 2, CV_AA);
  }
  for( int i = 0; i < newLines.size(); i++ )
  {
      float x1 = newLines[i][0], x2 = newLines[i][2], y1 = newLines[i][1],y2 = newLines[i][3];
      Point pt1, pt2;
      pt1.x = (int) x1;
      pt1.y = (int) y1;
      pt2.x = (int) x2;
      pt2.y = (int) y2;
      line( newFrame, pt1, pt2, Scalar(0,255,0), 2, CV_AA);
  }
  namedWindow("image2",CV_WINDOW_AUTOSIZE);
  imshow("image2",newFrame);
  waitKey(0);
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
