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

// 进行图像类比
void ImageAnalogy::process(const Mat& src, const Mat& srcFiltered, const Mat& dst, Mat& dstFiltered) {
    dstFiltered.create(dst.size(), CV_8UC3);
    buildPyramids(src, srcFiltered, dst, dstFiltered);
    createKernel();
    calculateSrcFeatures();
    // 从金字塔底层向上重建
    for (int i = 0; i < levels; i++) {
        float weight = 1 + pow(2, i + 1 - levels) * kappa;
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
                ann.knnSearch(query, indices, dists, numQuery, FlannSearchParams(128));
                int pA = indices.ptr()[0];
                // 一致性搜索
                int pC = cm.match(*srcFeatures[i], queryData, dimension, x, y, s);
                // 搜索失败的情况
                if (pC == -1) {
                    p = pA;
                }
                else {
                    float distA = featureDistance(srcFeatures[i]->ptr() + pA * dimension, queryData);
                    float distC = featureDistance(srcFeatures[i]->ptr() + pC * dimension, queryData);
                    p = weight * distA < distC ? pA : pC;
                }
                // 处理边界
                if (x < 1 || y < 1 || x > dstPyramid[i].cols - 2 || y > dstPyramid[i].rows - 2) p = pA;
                // 填充像素
                s[q] = p;
                int px = p % srcPyramid[i].cols, py = p / srcPyramid[i].cols;
                dstFilteredPyramid[i].at<Vec3b>(y, x) = srcFilteredPyramid[i].at<Vec3b>(py, px);
//                if (i == levels - 1) {
//                    dstFiltered.at<Vec3b>(y, x) = srcFiltered.at<Vec3b>(py, px);
//                }
            }
            cout << "Finish level " << i << " row " << y << endl;
        }
//        Mat downSampled;
//        cvtColor(dst, downSampled, COLOR_BGR2YUV);
//        // 降采样到同一尺度
//        for (int level = levels - 1; level > i; level--) {
//            pyrDown(downSampled, downSampled);
//        }
//        // 用生成的结果替换原图Y通道
//        vector<Mat> YUV(3);
//        split(downSampled, YUV);
//        dstFilteredPyramid[i].convertTo(YUV[0], CV_8UC1);
//        Mat result;
//        merge(YUV, result);
//        cvtColor(result, result, COLOR_YUV2BGR);
        if (i == levels - 1)
            dstFilteredPyramid[i].copyTo(dstFiltered);
//        // 显示当前层图片
//        imshow("image", dstFilteredPyramid[i]);
//        waitKey();
    }
}

// 提取亮度值
void ImageAnalogy::extractLuminance(const Mat& src, Mat &dst) {
    Mat srcYUV, YUV[3];
    cvtColor(src, srcYUV, COLOR_BGR2YUV);
    split(srcYUV, YUV);
    YUV[0].convertTo(dst, CV_32FC1);
}

// 生成高斯金字塔
void ImageAnalogy::buildPyramids(const Mat& src, const Mat& srcFiltered, const Mat& dst, const Mat& dstFiltered) {
    
    // 创建金字塔最高层
//    extractLuminance(src, srcPyramid[levels - 1]);
//    extractLuminance(srcFiltered, srcFilteredPyramid[levels - 1]);
//    extractLuminance(dst, dstPyramid[levels - 1]);
    src.copyTo(srcPyramid[levels - 1]);
    srcFiltered.copyTo(srcFilteredPyramid[levels - 1]);
    dst.copyTo(dstPyramid[levels - 1]);
    srcFiltered.copyTo(dstFilteredPyramid[levels - 1]);
    dstFilteredPyramid[levels - 1] = dstFilteredPyramid[levels - 1](Rect(0, 0, dst.cols, dst.rows));
    
    // 降采样
    for (int i = levels - 2; i >= 0; i--) {
        pyrDown(srcPyramid[i + 1], srcPyramid[i]);
        pyrDown(srcFilteredPyramid[i + 1], srcFilteredPyramid[i]);
        pyrDown(dstPyramid[i + 1], dstPyramid[i]);
        dstPyramid[i].copyTo(dstFilteredPyramid[i]);
    }
}

// 填充高斯卷积核
void ImageAnalogy::fillKernel(float *kernel, int size, float sigma) {
    int center = size / 2;
    int p = 0;
    float sum = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel[p] = 1 / (2 * PI * sigma * sigma) * exp(-((i - center) * (i - center) + (j - center) * (j - center)) / (2 * sigma * sigma));
            p++;
            for (int k = 1; k < channels; k++) {
                kernel[p] = kernel[p - 1];
                p++;
            }
            sum += kernel[p - 1] * channels;
        }
    }
    // 归一化
    p = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < channels; k++) {
                kernel[p++] /= sum;
            }
        }
    }
}

// 创建卷积核
void ImageAnalogy::createKernel() {
    kernel = new float[2 * (smallWindowSize + largeWindowSize) * channels];
    float sigma = 0.8;
    fillKernel(kernel, smallWindow, sigma);
    fillKernel(kernel + channels * smallWindowSize, smallWindow, sigma);
    fillKernel(kernel + channels * (2 * smallWindowSize), largeWindow, sigma);
    fillKernel(kernel + channels * (2 * smallWindowSize + largeWindowSize), largeWindow, sigma);
}

// 计算单个像素特征向量（最底层）
void ImageAnalogy::calculateFeature(float *result, int x, int y, const Mat& origin, const Mat& filtered) {
    int offset = 0;
    float sum = 0;
    
    for (int k = 0; k < 2 * smallWindowSize * channels; k++)
        result[offset++] = 0;
    
    int count = 0;
    int filteredFeatureDimension = largeWindowSize / 2 + 1;
    for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
        for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
            int yy = y + dy, xx = x + dx;
            count++;
            if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                for (int k = 0; k < channels; k++) {
                    result[offset] = 0;
                    if (count < filteredFeatureDimension) {
                        result[offset + largeWindowSize * channels] = 0;
                    }
                    offset++;
                }
                continue;
            }
            for (int k = 0; k < channels; k++) {
                result[offset] = origin.at<Vec3b>(yy, xx)[k] * kernel[offset];
                sum += result[offset] * result[offset];
                if (count < filteredFeatureDimension) {
                    result[offset + largeWindowSize * channels] = filtered.at<Vec3b>(yy, xx)[k] * kernel[offset + largeWindowSize * channels];
                    sum += result[offset + largeWindowSize * channels] * result[offset + largeWindowSize * channels];
                }
                offset++;
            }
        }
    }
    
    // 归一化
    if (sum < EPS) { return; }
    for (int offset = 0; offset < dimension; offset++) {
        result[offset] /= sum;
    }
}

// 计算单个像素特征向量（非最底层）
void ImageAnalogy::calculateFeature(float *result, int x, int y, const Mat& lowerOrigin, const Mat& lowerFiltered, const Mat& origin, const Mat& filtered) {
    int offset = 0;
    float sum = 0;
    
    int halfY = y / 2, halfX = x / 2;
    for (int dy = -smallWindow / 2; dy <= smallWindow / 2; dy++) {
        for (int dx = -smallWindow / 2; dx <= smallWindow / 2; dx++) {
            int yy = halfY + dy, xx = halfX + dx;
            if (yy < 0 || xx < 0 || yy >= lowerOrigin.rows || xx >= lowerOrigin.cols) {
                for (int k = 0; k < channels; k++) {
                    result[offset] = 0;
                    result[offset + smallWindowSize * channels] = 0;
                    offset++;
                }
                continue;
            }
            for (int k = 0; k < channels; k++) {
                result[offset] = lowerOrigin.at<Vec3b>(yy, xx)[k] * kernel[offset];
                sum += result[offset] * result[offset];
                result[offset + smallWindowSize * channels] = lowerFiltered.at<Vec3b>(yy, xx)[k] * kernel[offset + smallWindowSize * channels];
                sum += result[offset + smallWindowSize * channels] * result[offset + smallWindowSize * channels];
                offset++;
            }
        }
    }
    
    offset = 2 * smallWindowSize * channels;
    int count = 0;
    int filteredFeatureDimension = largeWindowSize / 2 + 1;
    for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
        for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
            int yy = y + dy, xx = x + dx;
            count++;
            if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                for (int k = 0; k < channels; k++) {
                    result[offset] = 0;
                    if (count < filteredFeatureDimension) {
                        result[offset + largeWindowSize * channels] = 0;
                    }
                    offset++;
                }
                continue;
            }
            for (int k = 0; k < channels; k++) {
                result[offset] = origin.at<Vec3b>(yy, xx)[k] * kernel[offset];
                sum += result[offset] * result[offset];
                if (count < filteredFeatureDimension) {
                    result[offset + largeWindowSize * channels] = filtered.at<Vec3b>(yy, xx)[k] * kernel[offset + largeWindowSize * channels];
                    sum += result[offset + largeWindowSize * channels] * result[offset + largeWindowSize * channels];
                }
                offset++;
            }
        }
    }
    
    // 归一化
    if (sum < EPS) { return; }
    for (int offset = 0; offset < dimension; offset++) {
        result[offset] /= sum;
    }
}

// 计算整层所有像素特征向量（最底层）
float* ImageAnalogy::calculateFeatures(const Mat& origin, const Mat& filtered) {
    int size = origin.rows * origin.cols;
    float *result = new float[size * dimension];
    for (int y = 0; y < origin.rows; y++) {
        for (int x = 0; x < origin.cols; x++) {
            int pixelIndex = y * origin.cols + x;
            int start = pixelIndex * dimension;
            int offset = 0;
            float sum = 0;
            
            for (int k = 0; k < 2 * smallWindowSize * channels; k++)
                result[start + offset++] = 0;
            
            int count = 0;
            int filteredFeatureDimension = largeWindowSize / 2 + 1;
            for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
                for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
                    int yy = y + dy, xx = x + dx;
                    count++;
                    if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                        for (int k = 0; k < channels; k++) {
                            result[start + offset] = 0;
                            if (count < filteredFeatureDimension) {
                                result[start + offset + largeWindowSize * channels] = 0;
                            }
                            offset++;
                        }
                        continue;
                    }
                    for (int k = 0; k < channels; k++) {
                        result[start + offset] = origin.at<Vec3b>(yy, xx)[k] * kernel[offset];
                        sum += result[start + offset] * result[start + offset];
                        if (count < filteredFeatureDimension) {
                            result[start + offset + largeWindowSize * channels] = filtered.at<Vec3b>(yy, xx)[k] * kernel[offset + largeWindowSize * channels];
                            sum += result[start + offset + largeWindowSize * channels] * result[start + offset + largeWindowSize * channels];
                        }
                        offset++;
                    }
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

// 计算整层所有像素特征向量（非最底层）
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
                        for (int k = 0; k < channels; k++) {
                            result[start + offset] = 0;
                            result[start + offset + smallWindowSize * channels] = 0;
                            offset++;
                        }
                        continue;
                    }
                    for (int k = 0; k < channels; k++) {
                        result[start + offset] = lowerOrigin.at<Vec3b>(yy, xx)[k] * kernel[offset];
                        sum += result[start + offset] * result[start + offset];
                        result[start + offset + smallWindowSize * channels] = lowerFiltered.at<Vec3b>(yy, xx)[k] * kernel[offset + smallWindowSize * channels];
                        sum += result[start + offset + smallWindowSize * channels] * result[start + offset + smallWindowSize * channels];
                        offset++;
                    }
                }
            }
            
            offset = 2 * smallWindowSize * channels;
            int count = 0;
            int filteredFeatureDimension = largeWindowSize / 2 + 1;
            for (int dy = -largeWindow / 2; dy <= largeWindow / 2; dy++) {
                for (int dx = -largeWindow / 2; dx <= largeWindow / 2; dx++) {
                    int yy = y + dy, xx = x + dx;
                    count++;
                    if (yy < 0 || xx < 0 || yy >= origin.rows || xx >= origin.cols) {
                        for (int k = 0; k < channels; k++) {
                            result[start + offset] = 0;
                            if (count < filteredFeatureDimension) {
                                result[start + offset + largeWindowSize * channels] = 0;
                            }
                            offset++;
                        }
                        continue;
                    }
                    for (int k = 0; k < channels; k++) {
                        result[start + offset] = origin.at<Vec3b>(yy, xx)[k] * kernel[offset];
                        sum += result[start + offset] * result[start + offset];
                        if (count < filteredFeatureDimension) {
                            result[start + offset + largeWindowSize * channels] = filtered.at<Vec3b>(yy, xx)[k] * kernel[offset + largeWindowSize * channels];
                            sum += result[start + offset + largeWindowSize * channels] * result[start + offset + largeWindowSize * channels];
                        }
                        offset++;
                    }
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

// 计算source图片所有层的特征向量
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
    }
}

// 计算两个特征向量之间的距离
float ImageAnalogy::featureDistance(float *a, float *b) {
    float result = 0;
    for (int i = 0; i < dimension; i++) {
        float difference = a[i] - b[i];
        result += difference * difference;
    }
    return result;
}
