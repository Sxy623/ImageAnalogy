//
//  main.cpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/20.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#include <iostream>
#include <string>
#include <ctime>
#include <opencv2/opencv.hpp>

#include "ImageAnalogy.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    
    // 检查参数个数
    if (argc < 2) {
        cout << "Parameter not enough!" << endl;
        cout << "Format: ./ImageAnalagy <data-set_path>" << endl;
        return 0;
    }
    
    // 图片路径
    string imageSetPath = argv[1];
    string srcPath = imageSetPath + "/src.jpg";
    string srcFilteredPath = imageSetPath + "/srcFiltered.jpg";
    string dstPath = imageSetPath + "/dst.jpg";
    string dstFilteredPath = imageSetPath + "/dstFiltered.jpg";
    
    // 加载图片
    Mat src, srcFiltered, dst, dstFiltered;
    src = imread(srcPath, IMREAD_COLOR);
    if (src.empty()) {
        cout << "Can't open src image!" << endl;
        return 0;
    }
    srcFiltered = imread(srcFilteredPath, IMREAD_COLOR);
    if (srcFiltered.empty()) {
        cout << "Can't open srcFiltered image!" << endl;
        return 0;
    }
    dst = imread(dstPath, IMREAD_COLOR);
    if (dst.empty()) {
        cout << "Can't open dst image!" << endl;
        return 0;
    }
    
    // 结果生成
    clock_t start = clock();
    ImageAnalogy analogy;
    analogy.process(src, srcFiltered, dst, dstFiltered);
    imwrite(dstFilteredPath, dstFiltered);
    clock_t end = clock();
    cout << "Run time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;
    
    // 显示图片
    imshow("src", src);
    imshow("srcFiltered", srcFiltered);
    imshow("dst", dst);
    imshow("dstFiltered", dstFiltered);
    waitKey();
    return 0;
}
