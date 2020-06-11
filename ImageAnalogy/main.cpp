//
//  main.cpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/20.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 图片路径
string srcPath = "/Users/sxy/Downloads/images/src.jpg";
string srcFilteredPath = "/Users/sxy/Downloads/images/srcFiltered.jpg";
string dstPath = "/Users/sxy/Downloads/images/dst.jpg";
string dstFilteredPath = "/Users/sxy/Downloads/images/dstFiltered.jpg";

int main() {
    
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
    
    // 显示图片
    imshow("src", src);
    imshow("srcFiltered", srcFiltered);
    imshow("dst", dst);
    waitKey();
    return 0;
}
