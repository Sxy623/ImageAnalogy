//
//  CoherenceMatch.cpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/25.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#include "CoherenceMatch.hpp"

CoherenceMatch::CoherenceMatch(int srcRows, int srcCols, int dstRows, int dstCols) {
    this->srcRows = srcRows;
    this->srcCols = srcCols;
    this->dstRows = dstRows;
    this->dstCols = dstCols;
}

CoherenceMatch::~CoherenceMatch() {}

int CoherenceMatch::match(const Matrix<float>& features, float *query, int dimension, int x, int y, int *s) {
    int p = -1;
    int minDistance = INF;
    
    // 搜索之前行的像素
    int minY = (y - radius >= 0 ? y - radius : 0);
    int minX = (x - radius >= 0 ? x - radius : 0);
    int maxX = (x + radius < dstCols ? x + radius : dstCols - 1);
    for (int yy = minY; yy < y; yy++) {
        for (int xx = minX; xx < maxX; xx++) {
            int sr = s[yy * dstCols + xx];
            int sx = sr % srcCols, sy = sr / srcCols;
            int px = sx + x - xx, py = sy + y - yy;
            // 判断边界
            if (px < 0 || py < 0 || px >= srcCols || py >= srcRows)
                continue;
            int index = py * srcCols + px;
            float dist = distance(query, features.ptr() + dimension * index, dimension);
            if (dist < minDistance) {
                minDistance = dist;
                p = index;
            }
        }
    }
    
    // 搜索该行左侧的像素
    for (int xx = minX; xx < x; xx++) {
        int sr = s[y * dstCols + xx];
        int sx = sr % srcCols, sy = sr / srcCols;
        int px = sx + x - xx, py = sy;
        // 判断边界
        if (px < 0 || py < 0 || px >= srcCols || py >= srcRows)
            continue;
        int index = py * srcCols + px;
        float dist = distance(query, features.ptr() + dimension * index, dimension);
        if (dist < minDistance) {
            minDistance = dist;
            p = index;
        }
    }
    
    return p;
}

float CoherenceMatch::distance(float *a, float *b, int dimension) {
    float result = 0;
    for (int i = 0; i < dimension; i++) {
        float difference = a[i] - b[i];
        result += difference * difference;
    }
    return result;
}
