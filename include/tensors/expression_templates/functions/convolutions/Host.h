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
namespace exprs {
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
	template<class Output, class Kernel, class Image>
	static void conv2d(
				Output output,
				Kernel krnl,
				Image img,
				size_t padding=0,
				size_t stride=1,
				typename Output::value_type alpha=1,
				typename Output::value_type beta=0) {

		BC::size_t numb_krnls = krnl.dimension(3);
		BC::size_t depth = krnl.dimension(2);

		using value_type = typename Output::value_type;

		BC_omp_parallel__
		for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
			for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
				for (int k = 0; k < numb_krnls; ++k) {
					value_type sum = 0;
					for (int d = 0; d < depth; ++d) {
						for (int kc = 0; kc < krnl.cols(); ++kc) {
							for (int kr = 0; kr < krnl.rows(); ++kr) {
								if (c+kc >= 0 && c+kc < img.cols() &&
									r+kr >= 0 && r+kr < img.rows()) {
									auto x = img(d, c+kc, r+kr);
									auto w = krnl(k, d, kc, kr);
									sum += w * x;
								}
							}
						}
					}
					output(k, c, r) = output(k, c, r) * beta + sum * alpha;
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
				size_t stride=1,
				size_t padding=0,
				typename Image::value_type alpha=1,
				typename Image::value_type beta=0) {

		BC::size_t numb_krnls = krnl.dimension(3);
		BC::size_t depth = krnl.dimension(2);

		if (beta != 1) {
			BC_omp_parallel__
			for (BC::size_t i = 0; i < img.size(); ++i) {
				img[i] *= beta;
			}
		}

		BC_omp_parallel__
		for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
			for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
				for (int k = 0; k < numb_krnls; ++k) {
					for (int d = 0; d < depth; ++d) {
						for (int kc = 0; kc < krnl.cols(); ++kc) {
							for (int kr = 0; kr < krnl.rows(); ++kr) {
								if (c+kc >= 0 && c+kc < img.cols() &&
									r+kr >= 0 && r+kr < img.rows()) {
									img(d, c+kc, r+kr) += krnl(k, d, kc, kr) * delta(k, c, r) * alpha;
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
				size_t stride=1,
				size_t padding=0,
				typename Image::value_type alpha=1,
				typename Image::value_type beta=0) {

		if (beta != 1) {
			BC_omp_parallel__
			for (BC::size_t i = 0; i < krnl.size(); ++i) {
				krnl[i] *= beta;
			}
		}

		BC::size_t numb_krnls = krnl.dimension(3);
		BC::size_t depth = krnl.dimension(2);

		BC_omp_parallel__
		for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
			for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
				for (int k = 0; k < numb_krnls; ++k) {
					for (int d = 0; d < depth; ++d) {
						for (int kc = 0; kc < krnl.cols(); ++kc) {
							for (int kr = 0; kr < krnl.rows(); ++kr) {
								if (c+kc >= 0 && c+kc < img.cols() &&
									r+kr >= 0 && r+kr < img.rows()) {
									krnl(k, d, kc, kr) += img(d, c+kc, r+kr) * output(k, c, r) * alpha;
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
}


#endif /* CONVOLUTION_HOST_H_ */
