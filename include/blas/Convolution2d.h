/*
 * Convolution2d.h
 *
 *  Created on: Aug 20, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_BLAS_CONVOLUTION2D_H_
#define BLACKCAT_BLAS_CONVOLUTION2D_H_

namespace BC {
namespace blas {

template<class OutputPtr, class ImgPtr, class KrnlPtr>
void conv2d(OutputPtr output, size_t output_ld,
			ImgPtr img, size_t rows, size_t cols, size_t img_ld,
			KrnlPtr krnl, size_t k_rows, size_t k_cols, size_t krnl_ld, size_t stride=1, size_t padding=0) {

	for (int c = -padding; c < cols + padding - k_cols + 1; c += stride) {
		for (int r = -padding; r < rows + padding - k_rows + 1; r += stride) {
			int output_index = (r+padding) + (c+padding) * output_ld;
			output[output_index] = 0;
			for (int kc = 0; kc < k_cols; ++kc) {
				for (int kr = 0; kr < k_rows; ++kr) {

					int krnl_index = kr + kc * krnl_ld;
					int img_index = (r + kr) + (c + kc) * img_ld;

					if (c+kc >= 0 && c+kc < cols &&
						r+kr >= 0 && r+kr < rows) {
						output[output_index] += krnl[krnl_index] * img[img_index];
					}
				}
			}
		}
	}
}

}
}




#endif /* CONVOLUTION2D_H_ */
