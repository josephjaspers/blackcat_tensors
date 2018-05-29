/*
 * toeplitz_kernel_matrix.h
 *
 *  Created on: May 26, 2018
 *      Author: joseph
 */

#ifndef TOEPLITZ_KERNEL_MATRIX_H_
#define TOEPLITZ_KERNEL_MATRIX_H_

namespace BC {

template<class > struct Core;
//template<int> struct Shape;
//internal tensor type

static constexpr int max(int a, int b) {
	return a > b ? a : b;
}

template<class core_lib>
struct CPU_Signal_Processing {

	//vector,vector convolution matrix
	template<class T, class I, class K>
	static void corr_toeplitz_1d(Core<T> tensor, Core<I> img, Core<K> krnl) {
		for (int n = 0; n < tensor.cols(); ++n)
			for (int m = 0; m < tensor.rows(); ++m)
				tensor(m, n) = img(m + n);
	}



	//naive impl just a test version --
	template<class Tensor, class Img, class Krnl>
	static void corr_toeplitz_img(Tensor tensor, Img img, Krnl krnl) {
		int tensor_pos = 0;
		for (int  c = 0; c < img.cols() - krnl.cols() + 1; ++c) {
			for (int r = 0; r < img.rows() - krnl.rows() + 1; ++r) {
					for (int y = 0; y < krnl.rows(); ++y) {
						for (int x = 0; x < krnl.cols(); ++x) {
						tensor[tensor_pos] = img(r + y, c + x);
						tensor_pos ++;

					}
				}
			}
		}
	}
};

}

#endif /* TOEPLITZ_KERNEL_MATRIX_H_ */
