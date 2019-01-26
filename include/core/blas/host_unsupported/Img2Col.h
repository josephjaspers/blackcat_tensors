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



template<class value_type>
void img2col(int dims, scalat_t* A, value_type alpha,int* dimsA, int* lda,
        scalat_t* B, int* dimsB, int* ldb, value_type* C) {
    img2col_packed(dims, A, alpha, dim_
}



}
}




#endif /* IMG2COL_H_ */
