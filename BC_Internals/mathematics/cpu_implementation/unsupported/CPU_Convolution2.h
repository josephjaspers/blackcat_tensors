/*
 * CPU_Convolution.h
 *
 *  Created on: Jul 5, 2018
 *      Author: joseph
 */

#ifndef CPU_CONVOLUTION_H_
#define CPU_CONVOLUTION_H_

#include "CPU_BLAS.h"

namespace BC {


/*
 * @param m -> kernel_length
 * @parma n -> img_length : n will be updated to the floor of n / m
 * @inc_y, inc_x -> must be '1' else error will be thrown
 */
void conv1(int m, int n,
            float* alpha, float* A, int inc_a,
                          float* X, int inc_x,
            float* beta,  float* Y, int inc_y,
            int padding=0, int stride=1) {

    if (inc_y != 1 || inc_x != 1 || inc_a != 1)
        throw std::invalid_argument("BC_CONV1 implementation is only supported by vectors");

    int lda = m;

    for (int i = 0; i < m; ++i) {
        //(int)((m-i)/n) --> the number of 'row positions' (we are converting our 1d vector img to a matrix)
        cblas_sgemv(CblasColMajor, CblasTrans, m, (int)((m-i)/n), *alpha, A, lda, X, lda, *beta, Y, m);

        //increment the stats

        Y += stride;
        A += stride;
    }
}


}






#endif /* CPU_CONVOLUTION_H_ */
