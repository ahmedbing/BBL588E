//
//  abingol_homework_1.cpp
//  Opencv1
//
//  Created by Ahmed Bingol on 4/24/20.
//  Copyright © 2020 Ahmed Bingol. All rights reserved.
//

#include "abingol_homework_1.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void histDisplay(int histogram[], const char* name);
void problem4and5(const cv::Mat &I, uint8_t threshold );
uint8_t ThresholdDeterminer( const int hist[]);
void NoiseImgCreator(cv::Mat &Input, cv::Mat &Output, double threshold);
void NoiseImgDisplay(cv::Mat BGR[], cv::Mat &Gray_I_1, cv::Mat &Gray_I_5, cv::Mat &Gray_I_10, cv::Mat &Gray_I_20);
void LPF(cv::Mat Img ,const::std::string name);
void HPF(cv::Mat Img,const::std::string name );
void problem10meanfilter(const cv::Mat input,cv::Mat &output);

int main(){
    cv::Mat original= cv::imread("/Users/ahmedbingol/Documents/Xcode_projects/Opencv1/Opencv1/Homeworks/SunnyLake.bmp", cv::IMREAD_COLOR);
    cv::Mat problem10= cv::imread("/Users/ahmedbingol/Documents/Xcode_projects/Opencv1/Opencv1/Homeworks/Figure_1.png",cv::IMREAD_COLOR);
    
    cv::Mat BGR[3];
    
    if(original.empty()){
        std::cout<<"Error in Img Path or Img name. \n";
        return 0;
    }
    std::cout<<"Img Uploaded correctly\n";
    
    cv::Mat I(original.rows,original.cols,CV_8UC1); //GrayScale Output
    int dummy;
    cv::split(original,BGR);
    
    int histogram[256] = {0};
    
    for(int x=0; x<original.rows; x++){
        for(int y=0; y<original.cols; y++){
            dummy=0;
            dummy= BGR[0].at<uint8_t>(x, y)+BGR[1].at<uint8_t>(x,y)+BGR[2].at<uint8_t>(x,y);
            I.at<uchar>(x,y) = (dummy/3);
            histogram[(int)I.at<uint8_t>(x,y)]++;
        }
    }
    
    //Problem 1-2 Input and avarage taken Output
    cv::namedWindow("Input");
    cv::namedWindow("Output");
    
    cv::imshow("Input", original);
    cv::imshow("Output",I);
    cv::waitKey(0);
    
    char name[]= "Original Histogram";
    histDisplay(histogram, name);
    
    uint8_t threshold= ThresholdDeterminer(histogram);
    std::cout<<"Threshold is : " << int(threshold) << "\n";
    problem4and5(I, threshold);
    
    //Answer of Problem 6 and 7
    cv::Mat I_1 ,I_5,I_10,I_20;
    NoiseImgDisplay(BGR,I_1,I_5,I_10,I_20);
    
    /*Answer of Problem 8.
    Uncomment to Noised image you want to observe and comment others.
    */
    
    //LPF(I_1, "GrayScale Img Std=1  ");
    //LPF(I_5, "GrayScale Img Std=5  ");
    //LPF(I_10, "GrayScale Img Std=10  ");
    LPF(I_20, "GrayScale Img Std=20  ");
    //cv::waitKey(0);
    /*Answer of Problem 9:
    Uncomment to Noised image you want to observe and comment others.
    */
    //HPF(I_1, "GrayScale Img Std=1  ");
    //HPF(I_5, "GrayScale Img Std=5  ");
    //HPF(I_10, "GrayScale Img Std=10  ");
    HPF(I_20,"GrayScale Img Std=20  ");
    
    cv::namedWindow("Pepper-Salt Noise");
    cv::imshow("Pepper Salt Noise", problem10);
    
    cv::Mat problem10_output=problem10.clone();
    
    std::vector<cv::Mat> in_BGR(3);
    std::vector<cv::Mat> out_BGR(3);
    cv::split(problem10,in_BGR);
    cv::split(problem10_output,out_BGR);
    
    
    problem10meanfilter(in_BGR[0],out_BGR[0]);
    problem10meanfilter(in_BGR[1],out_BGR[1]);
    problem10meanfilter(in_BGR[2],out_BGR[2]);
    
    cv::merge(out_BGR, problem10_output);
    cv::namedWindow("Pepper-Salt de-Noise");
    cv::imshow("Pepper Salt de-Noise", problem10_output);
    
    cv::waitKey(0);
    
    return 0;
}

uint8_t ThresholdDeterminer( const int hist[]){
    int max[20];
    std::fill_n(max,20,100);
    int flag=0;
    for(int i=0; i<256; i++){
        flag=0;
        for(int y=0; y<20; y++){
            if( flag != 1 & hist[i]>hist[max[y]] ){
                max[y]=i;
                flag=1;
                break;
            }
            if(flag==1){
                break;
            }
        }
    }
    int dummy=0;
    for(int i=0; i<20; ++i){
        dummy +=max[i];
        std::cout<< i<< " th most freq Num: " << max[i] << " and value is  "<< hist[max[i]]<<  " \n";
    }
    return uint8_t(dummy/20);
}

void problem4and5(const cv::Mat &I, uint8_t threshold ){
    cv::Mat BinaryOutput(I.rows,I.cols, CV_8UC1, cv::Scalar(255));
    for(int x=0 ; x<I.rows; x++){
        for (int y=0 ; y<I.cols; y++) {
            if( I.at<uint8_t>(x,y)< threshold )
                BinaryOutput.at<uint8_t>(x,y)= 0;
        }
    }
    cv::namedWindow("Thresholded Binary Image ");
    cv::imshow("Thresholded Binary Image ", BinaryOutput);
    cv::waitKey(0);
}


void histDisplay(int histogram[], const char* name)
{
    int hist[256];
    for(int i = 0; i < 256; i++)
    {
        hist[i]=histogram[i];
    }
    // draw the histograms
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w/256);
  
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(255, 255, 255));
  
    // find the maximum intensity element from histogram
    int max = hist[0];
    for(int i = 1; i < 256; i++){
        if(max < hist[i]){
            max = hist[i];
        }
    }
 
    // normalize the histogram between 0 and histImage.rows
    for(int i = 0; i < 256; i++)
 {
        hist[i] = ((double)hist[i]/max)*histImage.rows;
    }
  
  
    // draw the intensity line for histogram
    for(int i = 0; i < 256; i++)
    {
        line(histImage, cv::Point(bin_w*(i), hist_h), cv::Point(bin_w*(i), hist_h - hist[i]),cv::Scalar(0,0,0), 1, 8, 0);
    }
  
    // display histogram
    cv::namedWindow(name, CV_WINDOW_AUTOSIZE);
    cv::imshow(name, histImage);
    cv::waitKey(0);
}

void NoiseImgCreator(cv::Mat &Input, cv::Mat &Output, double threshold ){
    
    cv::Mat noise = cv::Mat(Input.size(),Input.type());
    cv::Mat result=Input;
    cv::randn(noise, 0, threshold);
    result = result + noise;
    Output=result;
}

void NoiseImgDisplay(cv::Mat BGR[], cv::Mat &Gray_I_1, cv::Mat &Gray_I_5, cv::Mat &Gray_I_10, cv::Mat &Gray_I_20){
    cv::Mat I_1, I_5, I_10, I_20;
    std::vector<cv::Mat> noise_BGR(3);
       
    NoiseImgCreator(BGR[0], noise_BGR[0], 1);
    NoiseImgCreator(BGR[1], noise_BGR[1], 1);
    NoiseImgCreator(BGR[2], noise_BGR[2], 1);
    cv::merge(noise_BGR, I_1);
    
    cv::namedWindow("Noise Image with Std=1");
    cv::imshow("Noise Image with Std=1", I_1);
    
    NoiseImgCreator(BGR[0], noise_BGR[0], 5);
    NoiseImgCreator(BGR[1], noise_BGR[1], 5);
    NoiseImgCreator(BGR[2], noise_BGR[2], 5);
    cv::merge(noise_BGR, I_5);
   
    cv::namedWindow("Noise Image with Std=5");
    cv::imshow("Noise Image with Std=5", I_5);
    
    NoiseImgCreator(BGR[0], noise_BGR[0], 10);
    NoiseImgCreator(BGR[1], noise_BGR[1], 10);
    NoiseImgCreator(BGR[2], noise_BGR[2], 10);
    cv::merge(noise_BGR, I_10);
   
    cv::namedWindow("Noise Image with Std=10");
    cv::imshow("Noise Image with Std=10", I_10);
    
    NoiseImgCreator(BGR[0], noise_BGR[0], 20);
    NoiseImgCreator(BGR[1], noise_BGR[1], 20);
    NoiseImgCreator(BGR[2], noise_BGR[2], 20);
    cv::merge(noise_BGR, I_20);
   
    cv::namedWindow("Noise Image with Std=20");
    cv::imshow("Noise Image with Std=20", I_20);
    cv::waitKey(0);
    
    //Problem 7 Answer
    cv::cvtColor(I_1, Gray_I_1, CV_BGR2GRAY);
    cv::cvtColor(I_5, Gray_I_5, CV_BGR2GRAY);
    cv::cvtColor(I_10, Gray_I_10, CV_BGR2GRAY);
    cv::cvtColor(I_20, Gray_I_20, CV_BGR2GRAY);
    
    cv::namedWindow("GrayScale Noised Image Std=1");
    cv::namedWindow("GrayScale Noised Image Std=5");
    cv::namedWindow("GrayScale Noised Image Std=10");
    cv::namedWindow("GrayScale Noised Image Std=20");
    
    cv::imshow("GrayScale Noised Image Std=1",Gray_I_1);
    cv::imshow("GrayScale Noised Image Std=5",Gray_I_5);
    cv::imshow("GrayScale Noised Image Std=10",Gray_I_10);
    cv::imshow("GrayScale Noised Image Std=20",Gray_I_20);
    cv::waitKey(0);
}

//Low Pass filter for Problem 8

void LPF(cv::Mat Img ,const::std::string name){
    
    // Normalized Box filters 3x3 and 5x5 kernel masks
    cv::Mat kernel1 = cv::Mat::ones( 3, 3, CV_32F )/ (float)(9);
    cv::Mat kernel2 = cv::Mat::ones( 5, 5, CV_32F )/ (float)(25);
    
    cv::Mat output1,output2,output3;
    cv::filter2D(Img, output1, Img.depth(), kernel1);
    cv::filter2D(Img, output2, Img.depth(), kernel2);
    cv::GaussianBlur(Img, output3, cv::Size(3,3),0,0 );
    
    cv::namedWindow(name+" 3x3 Low-Pass Filter");
    cv::namedWindow(name+" 5x5 Low-Pass Filter");
    cv::namedWindow(name+" Gaussian Filter");
    
    cv::imshow(name+" 3x3 Low-Pass Filter", output1);
    cv::imshow(name+" 5x5 Low-Pass Filter", output2);
    cv::imshow(name+" Gaussian Filter",output3);
    cv::waitKey(0);
}

//High Pass Filters for Problem 9
void HPF(cv::Mat Img,const::std::string name ){
    cv::Mat output1,output2,output3;
   
    cv::Mat kernel1 =(cv::Mat_<char>(3,3)<< -1,-1,-1,
                                            -1,8,-1,
                                            -1,-1,-1);
    cv::Mat kernel2 =(cv::Mat_<float>(3,3)<< 0.17,0.67,0.17,
                                            0.67,-3.33,0.67,
                                            0.17,0.67,0.17);
    
    float HIGH_BOOST_MULTIPLIER = 1.2;
    cv::Mat kernel3 =(cv::Mat_<float>(3,3)<< -1,-1,-1,
                                            -1,HIGH_BOOST_MULTIPLIER*9,-1,
                                            -1,-1,-1);
    
    cv::filter2D(Img, output1, Img.depth(), kernel1);
    cv::filter2D(Img, output2, Img.depth(), kernel2);
    cv::filter2D(Img, output3, Img.depth(), kernel3);
    
    cv::namedWindow(name+" 1st High-Pass Filter");
    cv::namedWindow(name+" 2nd High-Pass Filter(Laplace)");
    cv::namedWindow(name+" High-Boost Filter");
    
    cv::imshow(name+" 1st High-Pass Filter", output1);
    cv::imshow(name+" 2nd High-Pass Filter(Laplace)", output2);
    cv::imshow(name+" High-Boost Filter", output3);
    cv::waitKey(0);
}

void problem10meanfilter(const cv::Mat input,cv::Mat &output){
    for(int i=1; i<input.rows-1; ++i)
        for(int j=1; j<input.cols-1; ++j)
        {
            if(input.at<uint8_t>(i,j)==0 | input.at<uint8_t>(i,j)==255 ){
                output.at<uint8_t>(i,j)= uint8_t((input.at<uint8_t>(i-1,j-1)+ input.at<uint8_t>(i,j-1)+ input.at<uint8_t>(i-1,j)+ input.at<uint8_t>(i+1,j-1)+input.at<uint8_t>(i-1,j+1)+ input.at<uint8_t>(i,j+1)+ input.at<uint8_t>(i+1,j)+ input.at<uint8_t>(i+1,j+1))/8);
            }
        }
}
