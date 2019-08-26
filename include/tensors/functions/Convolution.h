/*
 * Convolution.h
 *
 *  Created on: Aug 24, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_TENSORS_FUNCIONS_CONVOLUTION_H_
#define BLACKCATTENSORS_TENSORS_FUNCIONS_CONVOLUTION_H_

namespace BC {
namespace tensors {
namespace not_implemented {

using size_t = BC::size_t;

/**
 * 2d Convolution of a 3d tensor with multiple 3d kernels.
 * Assume packed format.
 *
 */

template<class Output, class Image, class Kernel>
void conv2d_3dtensor_multichannel(
			Output output,
			Kernel krnl,
			Image img,
			size_t stride=1, size_t padding=0) {

	static_assert(Image::tensor_dimension == Kernel::tensor_dimension-1,
			"img tensor_dimension must equal krnl tensor_dimension - 1");
	static_assert(Output::tensor_dimension == Kernel::tensor_dimension-1,
			"output tensor_dimension must equal krnl tensor_dimension - 1");

	BC_ASSERT(output.cols() == img.cols() + padding - krnl.cols() + 1,
			"Invalid output column dimension");

	BC_ASSERT(output.rows() == img.rows() + padding - krnl.rows() + 1,
				"Invalid output column dimension");

	BC_ASSERT(output.dimension(2) == krnl.dimesion(3),
				"Invalid output column dimension");

	BC::size_t numb_krnls = krnl.dimension(3);
	BC::size_t depth = krnl.dimension(2);

	for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
		for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
			for (int k = 0; k < numb_krnls; ++k) {
				float sum = 0;
				for (int kc = 0; kc < krnl.cols(); ++kc) {
					for (int kr = 0; kr < krnl.rows(); ++kr) {
						for (int d = 0; d < depth; ++d) {
							if (c+kc >= 0 && c+kc < img.cols() &&
								r+kr >= 0 && r+kr < img.rows()) {
								auto x = img(r+kr, c+kc, d);
								auto w = krnl(kr, kc, d, k);
								sum += w * x;
							}
						}
					}
				}
				output(r, c, k) = sum;
			}
		}
	}
}

}
}
}



#endif /* CONVOLUTION_H_ */
