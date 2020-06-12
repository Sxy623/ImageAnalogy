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
    createKernel();
    calculateSrcFeatures();
    // 从金字塔底层向上重建
    for (int i = 0; i < levels; i++) {
        float weight = 1 + pow(2, i + 1 - levels) * kappa;
//        int na = 0, nc = 0;
        // 数据结构初始化
        FlannIndex ann(*srcFeatures[i], FlannKDTreeIndexParams(4));
        ann.buildIndex();
        // 用于保存结果
        int numQuery = 1;
        IntMatrix indices(new int[numQuery * dimension], numQuery, dimension);
        FloatMatrix dists(new float[numQuery * dimension], numQuery, dimension);
        float *queryData = new float[dimension];
        // 一致性搜索
        CoherenceMatch cm(srcPyramid[i].rows, srcPyramid[i].cols, dstPyramid[i].rows, dstPyramid[i].cols);
        int size = dstPyramid[i].rows * dstPyramid[i].cols;
        int *s = new int[size];
        memset(s, 0, size * sizeof(int));
        // 遍历像素
        for (int y = 0; y < dstPyramid[i].rows; y++) {
            for (int x = 0; x < dstPyramid[i].cols; x++) {
                int q = y * dstPyramid[i].cols + x;
                int p;
                // 计算当前像素的特征向量
                if (i == 0) {
                    calculateFeature(queryData, x, y, dstPyramid[i], dstFilteredPyramid[i]);
                }
                else {
                    calculateFeature(queryData, x, y, dstPyramid[i - 1], dstFilteredPyramid[i - 1], dstPyramid[i], dstFilteredPyramid[i]);
                }
                // 寻找近似最近邻
                FloatMatrix query(queryData, numQuery, dimension);
                int pA = indices.ptr()[0];
                ann.knnSearch(query, indices, dists, numQuery, FlannSearchParams(128));
                // 一致性搜索
                int pC = cm.match(*srcFeatures[i], queryData, dimension, x, y, s);
                if (pC == -1) {
                    p = pA;
                }
                else {
                    float distA = featureDistance(srcFeatures[i]->ptr() + pA * dimension, queryData);
                    float distC = featureDistance(srcFeatures[i]->ptr() + pC * dimension, queryData);
//                    if (weight * distA < distC) {
//                        p = pA;
//                        na++;
//                    } else {
//                        p = pC;
//                        nc++;
//                    }
                    p = weight * distA < distC ? pA : pC;
                }
                // 填充像素
                s[q] = p;
                int px = p % srcPyramid[i].cols, py = p / srcPyramid[i].cols;
                dstFilteredPyramid[i].at<float>(y, x) = srcFilteredPyramid[i].at<float>(py, px);
            }
            cout << "Finish level " << i << " row " << y << endl;
//            cout << na << ":" << nc << endl;
        }
        Mat downSampled;
        cvtColor(dst, downSampled, COLOR_BGR2YUV);
        // 降采样到同一尺度
        for (int level = levels - 1; level > i; level--) {
            pyrDown(downSampled, downSampled);
        }
        // 用生成的结果替换原图Y通道
        vector<Mat> YUV(3);
        split(downSampled, YUV);
        dstFilteredPyramid[i].convertTo(YUV[0], CV_8UC1);
        Mat result;
        merge(YUV, result);
        cvtColor(result, result, COLOR_YUV2BGR);
        // 显示当前层图片
        imshow("image", result);
        waitKey();
    }
}

void ImageAnalogy::extractLuminance(const Mat& src, Mat &dst) {
    Mat srcYUV, YUV[3];
    cvtColor(src, srcYUV, COLOR_BGR2YUV);
    split(srcYUV, YUV);
    YUV[0].convertTo(dst, CV_32FC1);
}

void ImageAnalogy::buildPyramids(const Mat& src, const Mat& srcFiltered, const Mat& dst, const Mat& dstFiltered) {
    
    // 创建金字塔最高层
    extractLuminance(src, srcPyramid[levels - 1]);
    extractLuminance(srcFiltered, srcFilteredPyramid[levels - 1]);
    extractLuminance(dst, dstPyramid[levels - 1]);
    extractLuminance(dstFiltered, dstFilteredPyramid[levels - 1]);
    
    // 降采样
    for (int i = levels - 2; i >= 0; i--) {
        pyrDown(srcPyramid[i + 1], srcPyramid[i]);
        pyrDown(srcFilteredPyramid[i + 1], srcFilteredPyramid[i]);
        pyrDown(dstPyramid[i + 1], dstPyramid[i]);
        pyrDown(dstFilteredPyramid[i + 1], dstFilteredPyramid[i]);
    }
}

void ImageAnalogy::fillKernel(float *kernel, int size, float sigma) {
    int center = size / 2;
    int p = 0;
    float sum = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[p] = 1 / (2 * PI * sigma * sigma) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
            sum += kernel[p];
            p++;
        }
    }
    // 归一化
    p = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[p++] /= sum;
        }
    }
}

void ImageAnalogy::createKernel() {
    kernel = new float[2 * (smallWindowSize + largeWindowSize)];
    float sigma = 0.8;
    fillKernel(kernel, smallWindow, sigma);
    fillKernel(kernel + smallWindowSize, smallWindow, sigma);
    fillKernel(kernel + 2 * smallWindowSize, largeWindow, sigma);
    fillKernel(kernel + 2 * smallWindowSize + largeWindowSize, largeWindow, sigma);
}

void ImageAnalogy::calculateFeature(float *result, int x, int y, const Mat& origin, const Mat& filtered) {
    int offset = 0;
    float sum = 0;
    
    for (int k = 0; k < 2 * smallWindowSize; k++)
        result[offset++] = 0;
    
    int count = 0;
    int filteredFeatureDimension = largeWindowSize / 2 + 1;
    for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
        for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
            int yy = y + dy, xx = x + dx;
            count++;
            if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                result[offset] = 0;
                if (count <= filteredFeatureDimension) {
                    result[offset + largeWindowSize] = 0;
                }
                offset++;
                continue;
            }
            result[offset] = origin.at<float>(yy, xx) * kernel[offset];
            sum += result[offset] * result[offset];
            if (count <= filteredFeatureDimension) {
                result[offset + largeWindowSize] = filtered.at<float>(yy, xx) * kernel[offset + largeWindowSize];
                sum += result[offset + largeWindowSize] * result[offset + largeWindowSize];
            }
            offset++;
        }
    }
    
    // 归一化
    if (sum < EPS) { return;; }
    for (int offset = 0; offset < dimension; offset++) {
        result[offset] /= sum;
    }
}

void ImageAnalogy::calculateFeature(float *result, int x, int y, const Mat& lowerOrigin, const Mat& lowerFiltered, const Mat& origin, const Mat& filtered) {
    int offset = 0;
    float sum = 0;
    
    int halfY = y / 2, halfX = x / 2;
    for (int dy = -smallWindow / 2; dy <= smallWindow / 2; dy++) {
        for (int dx = -smallWindow / 2; dx <= smallWindow / 2; dx++) {
            int yy = halfY + dy, xx = halfX + dx;
            if (yy < 0 || xx < 0 || yy >= lowerOrigin.rows || xx >= lowerOrigin.cols) {
                result[offset] = 0;
                result[offset + smallWindowSize] = 0;
                offset++;
                continue;
            }
            result[offset] = lowerOrigin.at<float>(yy, xx) * kernel[offset];
            sum += result[offset] * result[offset];
            result[offset + smallWindowSize] = lowerFiltered.at<float>(yy, xx) * kernel[offset + smallWindowSize];
            sum += result[offset + smallWindowSize] * result[offset + smallWindowSize];
            offset++;
        }
    }
    
    int count = 0;
    int filteredFeatureDimension = largeWindowSize / 2 + 1;
    for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
        for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
            int yy = y + dy, xx = x + dx;
            count++;
            if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                result[offset] = 0;
                if (count <= filteredFeatureDimension) {
                    result[offset + largeWindowSize] = 0;
                }
                offset++;
                continue;
            }
            result[offset] = origin.at<float>(yy, xx) * kernel[offset];
            sum += result[offset] * result[offset];
            if (count <= filteredFeatureDimension) {
                result[offset + largeWindowSize] = filtered.at<float>(yy, xx) * kernel[offset + largeWindowSize];
                sum += result[offset + largeWindowSize] * result[offset + largeWindowSize];
            }
            offset++;
        }
    }
    
    // 归一化
    if (sum < EPS) { return; }
    for (int offset = 0; offset < dimension; offset++) {
        result[offset] /= sum;
    }
}

float* ImageAnalogy::calculateFeatures(const Mat& origin, const Mat& filtered) {
    int size = origin.rows * origin.cols;
    float *result = new float[size * dimension];
    for (int y = 0; y < origin.rows; y++) {
        for (int x = 0; x < origin.cols; x++) {
            int pixelIndex = y * origin.cols + x;
            int start = pixelIndex * dimension;
            int offset = 0;
            float sum = 0;
            
            for (int k = 0; k < 2 * smallWindowSize; k++)
                result[start + offset++] = 0;
            
            int count = 0;
            int filteredFeatureDimension = largeWindowSize / 2 + 1;
            for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
                for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
                    int yy = y + dy, xx = x + dx;
                    count++;
                    if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                        result[start + offset] = 0;
                        if (count <= filteredFeatureDimension) {
                            result[start + offset + largeWindowSize] = 0;
                        }
                        offset++;
                        continue;
                    }
                    result[start + offset] = origin.at<float>(yy, xx) * kernel[offset];
                    sum += result[start + offset] * result[start + offset];
                    if (count <= filteredFeatureDimension) {
                        result[start + offset + largeWindowSize] = filtered.at<float>(yy, xx) * kernel[offset + largeWindowSize];
                        sum += result[start + offset + largeWindowSize] * result[start + offset + largeWindowSize];
                    }
                    offset++;
                }
            }
            
            // 归一化
            if (sum < EPS) { continue; }
            for (int offset = 0; offset < dimension; offset++) {
                result[start + offset] /= sum;
            }
        }
    }
    return result;
}

float* ImageAnalogy::calculateFeatures(const Mat& lowerOrigin, const Mat& lowerFiltered, const Mat& origin, const Mat& filtered) {
    int size = origin.rows * origin.cols;
    float *result = new float[size * dimension];
    for (int y = 0; y < origin.rows; y++) {
        for (int x = 0; x < origin.cols; x++) {
            int pixelIndex = y * origin.cols + x;
            int start = pixelIndex * dimension;
            int offset = 0;
            float sum = 0;
            
            int halfY = y / 2, halfX = x / 2;
            for (int dy = -smallWindow / 2; dy <= smallWindow / 2; dy++) {
                for (int dx = -smallWindow / 2; dx <= smallWindow / 2; dx++) {
                    int yy = halfY + dy, xx = halfX + dx;
                    if (yy < 0 || xx < 0 || yy >= lowerOrigin.rows || xx >= lowerOrigin.cols) {
                        result[start + offset] = 0;
                        result[start + offset + smallWindowSize] = 0;
                        offset++;
                        continue;
                    }
                    result[start + offset] = lowerOrigin.at<float>(yy, xx) * kernel[offset];
                    sum += result[start + offset] * result[start + offset];
                    result[start + offset + smallWindowSize] = lowerFiltered.at<float>(yy, xx) * kernel[offset + smallWindowSize];
                    sum += result[start + offset + smallWindowSize] * result[start + offset + smallWindowSize];
                    offset++;
                }
            }
            
            int count = 0;
            int filteredFeatureDimension = largeWindowSize / 2 + 1;
            for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
                for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
                    int yy = y + dy, xx = x + dx;
                    count++;
                    if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                        result[start + offset] = 0;
                        if (count <= filteredFeatureDimension) {
                            result[start + offset + largeWindowSize] = 0;
                        }
                        offset++;
                        continue;
                    }
                    result[start + offset] = origin.at<float>(yy, xx) * kernel[offset];
                    sum += result[start + offset] * result[start + offset];
                    if (count <= filteredFeatureDimension) {
                        result[start + offset + largeWindowSize] = filtered.at<float>(yy, xx) * kernel[offset + largeWindowSize];
                        sum += result[start + offset + largeWindowSize] * result[start + offset + largeWindowSize];
                    }
                    offset++;
                }
            }
            
            // 归一化
            if (sum < EPS) { continue; }
            for (int offset = 0; offset < dimension; offset++) {
                result[start + offset] /= sum;
            }
        }
    }
    return result;
}

void ImageAnalogy::calculateSrcFeatures() {
    for (int i = 0; i < levels; i++) {
        float *features = nullptr;
        int count = srcPyramid[i].rows * srcPyramid[i].cols;
        if (i == 0) {
            features = calculateFeatures(srcPyramid[i], srcFilteredPyramid[i]);
        }
        else {
            features = calculateFeatures(srcPyramid[i - 1], srcFilteredPyramid[i - 1], srcPyramid[i], srcFilteredPyramid[i]);
        }
        srcFeatures[i] = new FloatMatrix(features, count, dimension);
//        for (int y = 0; y < srcFeatures[i]->rows; y++) {
//            for (int x = 0; x < srcFeatures[i]->cols; x++)
//                std::cout << srcFeatures[i]->ptr()[y * srcFeatures[i]->cols + x] << " ";
//            std::cout << std::endl;
//        }
    }
}

float ImageAnalogy::featureDistance(float *a, float *b) {
    float result = 0;
    for (int i = 0; i < dimension; i++) {
        float difference = a[i] - b[i];
        result += difference * difference;
    }
    return result;
}
