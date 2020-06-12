//
//  CoherenceMatch.hpp
//  ImageAnalogy
//
//  Created by 沈心逸 on 2020/5/25.
//  Copyright © 2020 Xinyi Shen. All rights reserved.
//

#ifndef CoherenceMatch_hpp
#define CoherenceMatch_hpp

#include <flann/flann.hpp>

using namespace flann;

const float INF = 1e10;

class CoherenceMatch {
public:
    CoherenceMatch(int srcRows, int srcCols, int dstRows, int dstCols);
    ~CoherenceMatch();
    int match(const Matrix<float>& features, float *query, int dimension, int x, int y, int *s);
private:
    static const int radius = 2;
    int srcRows, srcCols;
    int dstRows, dstCols;
    
    float distance(float *a, float *b, int dimension);
};

#endif /* CoherenceMatch_hpp */
