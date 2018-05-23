/*
 * CorrToeplitz.h
 *
 *  Created on: May 23, 2018
 *      Author: joseph
 */

#ifndef CORRTOEPLITZ_H_
#define CORRTOEPLITZ_H_


class CPU_SPECIALIZED_BLAS {

	template<class T, class img, class krnl, int dim>
	void mat_conv_toeplitz_inner(T tensor, img I, krnl K) {
		int positions = I.dimension(dim - 1) = K.dimension(dim - 1) + 1;

		for (int i = 0; i < positions; ++i) {
			mat_conv_toeplitz_inner()
		}

	}


};


#endif /* CORRTOEPLITZ_H_ */
