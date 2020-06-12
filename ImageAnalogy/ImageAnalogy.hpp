//
//  ImageAnalogy.hpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/22.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#ifndef ImageAnalogy_hpp
#define ImageAnalogy_hpp

#include <opencv2/opencv.hpp>
#include <flann/flann.hpp>

typedef flann::Matrix<float> FeatureMatrix;

using namespace cv;

const float PI = 4.0 * atan(1.0);
const float eps = 1e-5;

class ImageAnalogy {
public:
    ImageAnalogy();
    ~ImageAnalogy();
    void process(const Mat& src, const Mat& srcFiltered, const Mat& dst, Mat& dstFiltered);
private:
    // 金字塔层数
    static const int levels = 3;
    // 窗口大小
    static const int smallWindow = 3;
    static const int largeWindow = 5;
    static const int smallWindowSize = smallWindow * smallWindow;
    static const int largeWindowSize = largeWindow * largeWindow;
    // 高斯金字塔
    Mat srcPyramid[levels], srcFilteredPyramid[levels], dstPyramid[levels], dstFilteredPyramid[levels];
    // 高斯卷积核
    float *kernel;
    // 特征向量维度
    static const int dimension = smallWindowSize * 2 + largeWindowSize + largeWindowSize / 2 + 1;
    // 特征向量
    FeatureMatrix *srcFeatures[levels];
    
    void extractLuminance(const Mat& src, Mat &dst);
    void buildPyramids(const Mat& src, const Mat& srcFiltered, const Mat& dst, const Mat& dstFiltered);
    void fillKernel(float *kernel, int size, float sigma);
    void createKernel();
    float* calculateFeatures(const Mat& origin, const Mat& filtered);
    float* calculateFeatures(const Mat& lowerOrigin, const Mat& lowerFiltered, const Mat& origin, const Mat& filtered);
    void calculateSrcFeatures();
};

#endif /* ImageAnalogy_hpp */
