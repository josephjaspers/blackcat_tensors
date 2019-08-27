/*
 * Convolution_Host.h
 *
 *  Created on: Aug 26, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_TENSORS_FUNCTIONS_CONVOLUTION_HOST_H_
#define BLACKCATTENSORS_TENSORS_FUNCTIONS_CONVOLUTION_HOST_H_

namespace BC {
namespace tensors {
namespace functions {
namespace convolutions {

template<class SystemTag>
struct Convolution_Implementation;

template<>
struct Convolution_Implementation<BC::host_tag> {

	using size_t = BC::size_t;

	/**
	 * Written for 3d tensors with multiple filters,
	 * this algorithm generalizes to 2d tensors with single filters as well.
	 */
	template<class Stream, class Output, class Kernel, class Image>
	static void conv2d(
				Stream stream,
				Output output,
				Kernel krnl,
				Image img,
				size_t stride=1, size_t padding=0) {

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

	//assumes inputs have been zeroed
	template<class Delta, class Image, class Kernel>
	static void conv2d_data_backwards(
				Delta delta,
				Kernel krnl,
				Image img,
				size_t stride=1, size_t padding=0) {

		BC::size_t numb_krnls = krnl.dimension(3);
		BC::size_t depth = krnl.dimension(2);

		for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
			for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
				for (int k = 0; k < numb_krnls; ++k) {
					for (int kc = 0; kc < krnl.cols(); ++kc) {
						for (int kr = 0; kr < krnl.rows(); ++kr) {
							for (int d = 0; d < depth; ++d) {
								if (c+kc >= 0 && c+kc < img.cols() &&
									r+kr >= 0 && r+kr < img.rows()) {
									img(r+kr, c+kc, d) += krnl(kr, kc, d, k) * delta(r, c, k);
								}
							}
						}
					}
				}
			}
		}
	}

	//assumes kerenl has been zeroed
	template<class Output, class Image, class Kernel>
	static void conv2d_kernel_backwards(
				Output output,
				Kernel krnl,
				Image img,
				size_t stride=1, size_t padding=0) {

		BC::size_t numb_krnls = krnl.dimension(3);
		BC::size_t depth = krnl.dimension(2);

		for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
			for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
				for (int k = 0; k < numb_krnls; ++k) {
					for (int kc = 0; kc < krnl.cols(); ++kc) {
						for (int kr = 0; kr < krnl.rows(); ++kr) {
							for (int d = 0; d < depth; ++d) {
								if (c+kc >= 0 && c+kc < img.cols() &&
									r+kr >= 0 && r+kr < img.rows()) {
									krnl(kr, kc, d, k) += img(r+kr, c+kc, d) * output(r, c, k);
								}
							}
						}
					}
				}
			}
		}
	}

};

}
}
}
}


#endif /* CONVOLUTION_HOST_H_ */
