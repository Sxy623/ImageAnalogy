//
//  ImageAnalogy.cpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/22.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#include "ImageAnalogy.hpp"

ImageAnalogy::ImageAnalogy() {}

ImageAnalogy::~ImageAnalogy() {}

void ImageAnalogy::process(const Mat& src, const Mat& srcFiltered, const Mat& dst, Mat& dstFiltered) {
    dstFiltered.create(dst.size(), CV_8UC3);
    buildPyramids(src, srcFiltered, dst, dstFiltered);
}

void ImageAnalogy::buildPyramids(const Mat& src, const Mat& srcFiltered, const Mat& dst, const Mat& dstFiltered) {
    
    // 创建金字塔最高层
    src.convertTo(srcPyramid[levels - 1], CV_32FC3);
    srcFiltered.convertTo(srcFilteredPyramid[levels - 1], CV_32FC3);
    dst.convertTo(dstPyramid[levels - 1], CV_32FC3);
    dstFiltered.convertTo(dstFilteredPyramid[levels - 1], CV_32FC3);
    
    // 降采样
    for (int i = levels - 2; i >= 0; i--) {
        pyrDown(srcPyramid[i + 1], srcPyramid[i]);
        pyrDown(srcFilteredPyramid[i + 1], srcFilteredPyramid[i]);
        pyrDown(dstPyramid[i + 1], dstPyramid[i]);
        pyrDown(dstFilteredPyramid[i + 1], dstFilteredPyramid[i]);
    }
}
