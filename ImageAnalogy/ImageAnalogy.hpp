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

using namespace cv;

class ImageAnalogy {
public:
    ImageAnalogy();
    ~ImageAnalogy();
    void process(const Mat& src, const Mat& srcFiltered, const Mat& dst, Mat& dstFiltered);
private:
    // 金字塔层数
    static const int levels = 3;
    // 窗口大小
    static const int smallWindowSize = 3;
    static const int largeWindowSize = 5;
    // 高斯金字塔
    Mat srcPyramid[levels], srcFilteredPyramid[levels], dstPyramid[levels], dstFilteredPyramid[levels];
    
    void buildPyramids(const Mat& src, const Mat& srcFiltered, const Mat& dst, const Mat& dstFiltered);
};

#endif /* ImageAnalogy_hpp */
