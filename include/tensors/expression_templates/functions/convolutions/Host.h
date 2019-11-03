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

/**
 * A class of various (host) Convolution algorithms
 * Expects col-major 3d or 4d Tensors in NCWH format.
 *
 * (BlackCat_Tensors are constructed in HWCN format but indexed via NCWH)
 *
 */
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
		BC::size_t numb_imgs = img.dimension(3);

		if (beta != 1) {
			BC_omp_parallel__
			for (BC::size_t i = 0; i < img.size(); ++i) {
				output[i] *= beta;
			}
		}

		BC_omp_parallel__
		for (int i = 0; i < numb_imgs; ++i) {
			BC_omp_parallel__
			for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
				for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
					for (int k = 0; k < numb_krnls; ++k) {
						for (int d = 0; d < depth; ++d) {
							for (int kc = 0; kc < krnl.cols(); ++kc) {
								for (int kr = 0; kr < krnl.rows(); ++kr) {
									if (c+kc >= 0 && c+kc < img.cols() &&
										r+kr >= 0 && r+kr < img.rows()) {
										auto x = img(i, d, c+kc, r+kr);
										auto w = krnl(k, d, kc, kr);
										output(i, k, c, r) += w * x;
									}
								}
							}
						}
					}
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
				size_t padding=0,
				size_t stride=1,
				typename Image::value_type alpha=1,
				typename Image::value_type beta=0) {

		BC::size_t numb_krnls = krnl.dimension(3);
		BC::size_t depth = krnl.dimension(2);
		BC::size_t numb_imgs = img.dimension(3);

		if (beta != 1) {
			BC_omp_parallel__
			for (BC::size_t i = 0; i < img.size(); ++i) {
				img[i] *= beta;
			}
		}

		BC_omp_parallel__
		for (int i = 0; i < numb_imgs; ++i) {
			BC_omp_parallel__
			for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
				for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
					for (int k = 0; k < numb_krnls; ++k) {
						for (int d = 0; d < depth; ++d) {
							for (int kc = 0; kc < krnl.cols(); ++kc) {
								for (int kr = 0; kr < krnl.rows(); ++kr) {
									if (c+kc >= 0 && c+kc < img.cols() &&
										r+kr >= 0 && r+kr < img.rows()) {
										img(i, d, c+kc, r+kr) += krnl(k, d, kc, kr) * delta(i,k,c,r);
									}
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
				size_t padding=0,
				size_t stride=1,
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
		BC::size_t numb_imgs = img.dimension(3);

		BC_omp_parallel__
		for (int i = 0; i < numb_imgs; ++i) {
			BC_omp_parallel__
			for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
				for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
					for (int k = 0; k < numb_krnls; ++k) {
						for (int d = 0; d < depth; ++d) {
							for (int kc = 0; kc < krnl.cols(); ++kc) {
								for (int kr = 0; kr < krnl.rows(); ++kr) {
									if (c+kc >= 0 && c+kc < img.cols() &&
										r+kr >= 0 && r+kr < img.rows()) {
										krnl(k, d, kc, kr) += img(i, d, c+kc, r+kr) * output(i, k, c, r);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	template<class Output, int KernelDimension, class Image> BCHOT
	static void img2col(
				Output output,
				BC::Shape<KernelDimension> krnl,
				Image img,
				size_t padding=0,
				size_t stride=1) {

		BC::size_t depth = krnl.dimension(2);
		BC::size_t numb_imgs = img.dimension(3);
		BC::Shape<6> dimension_set(
				krnl.rows(),
				krnl.cols(),
				depth,
				img.rows() - krnl.rows() + 1 + padding * 2,
				img.cols() - krnl.cols() + 1 + padding * 2,
				numb_imgs);

		BC_omp_parallel__
		for (int i = 0; i < numb_imgs; ++i) {
			BC_omp_parallel__
			for (int d = 0; d < depth; ++d) {
				for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
					for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
						for (int kc = 0; kc < krnl.cols(); ++kc) {
							for (int kr = 0; kr < krnl.rows(); ++kr) {
								auto output_index = dimension_set.dims_to_index(i, c, r, d, kc, kr);
								if (c+kc >= 0 && c+kc < img.cols() &&
									r+kr >= 0 && r+kr < img.rows()) {
									output.data()[output_index] = img(i, d, c+kc, r+kr);
								} else {
									output.data()[output_index] = 0;
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
