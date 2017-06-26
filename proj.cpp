#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iostream>
#include <string> 

// openCV
#include "opencv2/opencv.hpp"
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"

using namespace cv;

// timer
#ifndef _TIMER_H_
#define _TIMER_H_

#include <sys/time.h>

#define GET_TIME(now) { \
   struct timeval t; \
   gettimeofday(&t, NULL); \
   now = t.tv_sec + t.tv_usec/1000000.0; \
}

#endif

#define IMAGE_NUM 800

// shared viariables   
int thread_count;
int feature_num;
int image_num;
String image_name;

void* thread_work(void* rank);
void* thread_work_io(void* rank);
void compare_histogram(int s, int f);
void template_matching(int s, int f);
void feature_matching(int s, int f);
void usage(char* prog_name);

int main(int argc, char* argv[]) {
  double total, io, start, finish;
  int i;
  pthread_t* thread_handles; 

  if(argc != 4) usage(argv[0]);

  feature_num = strtol(argv[1], NULL, 10);
  thread_count = strtol(argv[2], NULL, 10);
  image_name = argv[3];
  image_num = IMAGE_NUM; 

  printf("\n**********************************\n");
  switch(feature_num) {
    case 0 : printf("***     Compare Histogram      ***\n"); break;
    case 1 : printf("***     Template Matching      ***\n"); break;
    case 2 : printf("***     Feature  Matching      ***\n"); break;
  }

// total
  thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
  GET_TIME(start);
  for(i=0 ; i<thread_count ; i++) 
    pthread_create(&thread_handles[i], NULL, thread_work, (void*)i);
  
  for(i=0 ; i<thread_count ; i++)
    pthread_join(thread_handles[i], NULL);
  GET_TIME(finish);

  free(thread_handles);
  total = finish - start;
  
// IO
  thread_handles = (pthread_t*)malloc(thread_count * sizeof(pthread_t));
  GET_TIME(start);
  for(i=0 ; i<thread_count ; i++)
    pthread_create(&thread_handles[i], NULL, thread_work_io, (void*)i);

  for(i=0 ; i<thread_count ; i++)
    pthread_join(thread_handles[i], NULL);
  GET_TIME(finish);

  free(thread_handles);
  io = finish - start;

  printf("thread num : %d\n", thread_count);
  printf("total time = %e seconds\n", total);
  printf("io time = %e seconds\n", io);  
  printf("processing time = %e seconds\n", total-io);
}

void* thread_work(void *rank) {
  long myRank;
  int sIndex, fIndex;
  
  myRank = (long)rank;
  sIndex = myRank * (image_num/thread_count);
  fIndex = sIndex + (image_num/thread_count) - 1;
  //printf("rank %d : %d ~ %d\n", myRank, sIndex, fIndex); 

  switch(feature_num) {
  case 0 : compare_histogram(sIndex, fIndex); break;
  case 1 : template_matching(sIndex, fIndex); break;
  case 2 : feature_matching(sIndex, fIndex); break;
  }

}

void* thread_work_io(void *rank) {
  int i;
  long myRank;
  int sIndex, fIndex;

  myRank = (long)rank;
  sIndex = myRank * (image_num/thread_count);
  fIndex = sIndex + (image_num/thread_count) - 1;

  for (i=sIndex ; i<=fIndex ; i++) {
    Mat src;
    std::string str = "input/" + std::to_string(i) + ".jpg";
    src = imread(str, IMREAD_COLOR );

    if (feature_num) {
      std::string str2 = "io/" + std::to_string(i)+ ".jpg";
      imwrite(str2, src);
    }
  }
}

void compare_histogram(int s, int f) {
  int j;
  
  
  Mat src_base = imread( image_name, IMREAD_COLOR );
  Mat src_hsv;
  cvtColor( src_base, src_hsv, COLOR_BGR2HSV );

  int h_bins = 50; int s_bins = 60;
  int histSize[] = { h_bins, s_bins };

  float h_ranges[] = { 0, 180 };
  float s_ranges[] = { 0, 256 };
  const float* ranges[] = { h_ranges, s_ranges };
  int channels[] = { 0, 1 };
  
  MatND src_hist;
  calcHist( &src_hsv, 1, channels, Mat(), src_hist, 2, histSize, ranges, true, false );
  normalize( src_hist, src_hist, 0, 1,  NORM_MINMAX, -1, Mat() );

  for (j=s ; j<=f ; j++) {
    std::string str = "input/" + std::to_string(j) + ".jpg";
    Mat dst_base = imread( str, IMREAD_COLOR );
    Mat dst_hsv;
    cvtColor( dst_base, dst_hsv, COLOR_BGR2HSV );
    
    MatND dst_hist;
    calcHist( &dst_hsv, 1, channels, Mat(), dst_hist, 2, histSize, ranges, true, false );
    normalize( dst_hist, dst_hist, 0, 1,  NORM_MINMAX, -1, Mat() );
    
    double result0 = compareHist( src_hist, dst_hist, 0);
    double result1 = compareHist( src_hist, dst_hist, 1);
    double result2 = compareHist( src_hist, dst_hist, 2);
    double result3 = compareHist( src_hist, dst_hist, 3);

    if(result0 > 0.9 ||
       result1 < 0.1 ||
       result2 < 1.5 ||
       result3 < 0.3)
    printf("%d.jpg\n", j);
  }
}

void template_matching(int s, int f) {
  int j;


  Mat templ = imread( image_name, IMREAD_COLOR );

  for (j=s ; j<=f ; j++) {
    std::string str = "input/" + std::to_string(j) + ".jpg";
    Mat img = imread( str, IMREAD_COLOR );
    
    Mat img_display, result;
    img.copyTo( img_display );

    int result_cols =  img.cols - templ.cols + 1;
    int result_rows = img.rows - templ.rows + 1;
    result.create( result_rows, result_cols, CV_32FC1 );
   
    int match_method = 5;
    matchTemplate( img, templ, result, match_method);
    normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );
    
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point matchLoc;
    minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    matchLoc = maxLoc;
    rectangle( img_display, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    rectangle( result, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(0), 2, 8, 0 );
    
    std::string str2 = "output/template_matching/" + std::to_string(j) + ".jpg";
    imwrite(str2, img_display);
  }
}

void feature_matching(int s, int f) {
  int j;

  Mat img_1 = imread( image_name, CV_LOAD_IMAGE_GRAYSCALE );
  int minHessian = 400;
  SurfFeatureDetector detector( minHessian );
  SurfDescriptorExtractor extractor;
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;

  detector.detect( img_1, keypoints_1 );
  extractor.compute( img_1, keypoints_1, descriptors_1 );

  for (j=s ; j<=f ; j++) {
    std::string str = "input/" + std::to_string(j) + ".jpg";
    Mat img_2 = imread( str, IMREAD_COLOR );

    detector.detect( img_2, keypoints_2 );
    extractor.compute( img_2, keypoints_2, descriptors_2 );
    
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;
    
     for( int i = 0; i < descriptors_1.rows; i++ )
     { double dist = matches[i].distance;
       if( dist < min_dist ) min_dist = dist;
       if( dist > max_dist ) max_dist = dist;
     }    

     std::vector< DMatch > good_matches;

     for( int i = 0; i < descriptors_1.rows; i++ )
     { if( matches[i].distance <= max(2*min_dist, 0.02) )
       { good_matches.push_back( matches[i]); }
     }

     Mat img_matches;
     drawMatches( img_1, keypoints_1, img_2, keypoints_2,
                  good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                  vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    std::string str2 = "output/feature_matching/" + std::to_string(j) + ".jpg";
    imwrite(str2, img_matches);
  }
}

void usage(char* prog_name) {
  fprintf(stderr, "usage : %s <feature number> <thread_count> <file> \n", prog_name);
  exit(0); 
}
