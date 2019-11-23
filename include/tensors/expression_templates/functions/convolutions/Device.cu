
#ifdef __CUDACC__
#ifndef BLACKCATTENSORS_TENSORS_FUNCTIONS_CONVOLUTION_DEVICE_H_
#define BLACKCATTENSORS_TENSORS_FUNCTIONS_CONVOLUTION_DEVICE_H_

#include "cuda.h"
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

template<class Tensor> __global__
static void scalar_mul(Tensor tensor, typename Tensor::value_type scalar) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (; i < tensor.size(); i += blockDim.x * gridDim.x) {
		tensor[i] *= scalar;
	}
}

template<class Delta, class Image, class Kernel> __global__
static void conv2d_data_backwards_gpu_kernel(
			Delta delta,
			Kernel krnl,
			Image img,
			BC::size_t padding=0,
			BC::size_t stride=1) {

	BC::size_t numb_krnls = krnl.dimension(3);
	BC::size_t depth = krnl.dimension(2);
	BC::size_t numb_imgs = img.dimension(3);

//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	for (; i < numb_imgs; i += blockDim.x * gridDim.x) {
//		for (int c = -padding; c < img.cols() + padding - krnl.cols() + 1; c += stride) {
//			for (int r = -padding; r < img.rows() + padding - krnl.rows() + 1; r += stride) {
//				for (int k = 0; k < numb_krnls; ++k) {
//					for (int d = 0; d < depth; ++d) {
//						for (int kc = 0; kc < krnl.cols(); ++kc) {
//							for (int kr = 0; kr < krnl.rows(); ++kr) {
//								if (c+kc >= 0 && c+kc < img.cols() &&
//									r+kr >= 0 && r+kr < img.rows()) {
//									auto x = img(i, d, c+kc, r+kr);
//									auto w = krnl(k, d, kc, kr);
//									delta(i, k, c, r) += w * x;
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//	BC::size_t numb_krnls = krnl.dimension(3);
//	BC::size_t depth = krnl.dimension(2);
//	BC::size_t numb_imgs = img.dimension(3);
//
//	int index = blockIdx.x * blockDim.x + threadIdx.x;
//	int width = delta.dimension(0);
//	int height = delta.dimension(1);
//	int channels = delta.dimension(2);

//	for (; index < img.size(); index += blockDim.x * gridDim.x) {
//		const int r = index % width;
//		const int c = (index / width) % height;
//		const int d = (index / width / height) % channels;
//		const int n = index / width / height / channels;
//		for (int k = 0; k < numb_krnls; ++k) {
//			for (int kc = 0; kc < krnl.cols(); ++kc) {
//				for (int kr = 0; kr < krnl.rows(); ++kr) {
//					if (c+kc >= 0 && c+kc < img.cols() &&
//						r+kr >= 0 && r+kr < img.rows()) {
//						auto x = img(n, d, c+kc, r+kr);
//						auto w = krnl(k, d, kc, kr);
//						delta(n, k, c, r) += w * x;
//					}
//				}
//			}
//		}
//	}
}

template<class SystemTag>
struct Convolution_Implementation;


template<>
struct Convolution_Implementation<BC::device_tag> {

	using size_t = BC::size_t;

	/**
	 * Written for 3d tensors with multiple filters,
	 * this algorithm generalizes to 2d tensors with single filters as well.
	 */

	//assumes inputs have been zeroed
	template<class Stream, class Delta, class Image, class Kernel>
	static void conv2d_data_backwards(
				Stream stream,
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

		stream.enqueue([=]() {
			if (beta != 1) {
				scalar_mul<<<calculate_block_dim(delta.size()), calculate_threads(), 0, stream>>>(delta, beta);
			}

			conv2d_data_backwards_gpu_kernel<<<calculate_block_dim(delta.size()), calculate_threads(), 0, stream>>>(
						delta,
						krnl,
						img,
						padding,
						stride);
		});




	}
};

}
}
}
}
}


#endif
#endif
