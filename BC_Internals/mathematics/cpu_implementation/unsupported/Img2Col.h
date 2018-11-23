/*
 * Img2Col.h
 *
 *  Created on: Jul 12, 2018
 *      Author: joseph
 */

#ifndef IMG2COL_H_
#define IMG2COL_H_

namespace BC {

namespace i2c {



template<class scalar_t>
void img2col(int dims, scalat_t* A, scalar_t alpha,int* dimsA, int* lda,
        scalat_t* B, int* dimsB, int* ldb, scalar_t* C) {
    img2col_packed(dims, A, alpha, dim_
}



}
}




#endif /* IMG2COL_H_ */
