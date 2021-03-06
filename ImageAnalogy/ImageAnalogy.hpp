//
//  ImageAnalogy.hpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/22.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#ifndef ImageAnalogy_hpp
#define ImageAnalogy_hpp

#include <iostream>
#include <vector>
#include <flann/flann.hpp>
#include <opencv2/opencv.hpp>

#include "CoherenceMatch.hpp"

typedef flann::Matrix<float> FloatMatrix;
typedef flann::Matrix<int> IntMatrix;
typedef flann::Index<flann::L2<float>> FlannIndex;
typedef flann::KDTreeIndexParams FlannKDTreeIndexParams;
typedef flann::SearchParams FlannSearchParams;

using namespace std;
using namespace cv;

const float PI = 4.0 * atan(1.0);
const float EPS = 1e-5;

class ImageAnalogy {
public:
    ImageAnalogy();
    ~ImageAnalogy();
    // 进行图像类比
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
    static const int channels = 3;
    static const int dimension = (smallWindowSize * 2 + largeWindowSize + largeWindowSize / 2) * channels;
    // 特征向量
    FloatMatrix *srcFeatures[levels];
    // 一致性参数
    constexpr static const float kappa = 0.8;
    
    // 提取亮度值
    void extractLuminance(const Mat& src, Mat &dst);
    // 生成高斯金字塔
    void buildPyramids(const Mat& src, const Mat& srcFiltered, const Mat& dst, const Mat& dstFiltered);
    // 填充高斯卷积核
    void fillKernel(float *kernel, int size, float sigma);
    // 创建卷积核
    void createKernel();
    // 计算单个像素特征向量（分为最底层和非最底层两个版本）
    void calculateFeature(float *result, int x, int y, const Mat& origin, const Mat& filtered);
    void calculateFeature(float *result, int x, int y, const Mat& lowerOrigin, const Mat& lowerFiltered, const Mat& origin, const Mat& filtered);
    // 计算整层所有像素特征向量（分为最底层和非最底层两个版本）
    float* calculateFeatures(const Mat& origin, const Mat& filtered);
    float* calculateFeatures(const Mat& lowerOrigin, const Mat& lowerFiltered, const Mat& origin, const Mat& filtered);
    // 计算source图片所有层的特征向量
    void calculateSrcFeatures();
    // 计算两个特征向量之间的距离
    float featureDistance(float *a, float *b);
};

#endif /* ImageAnalogy_hpp */
