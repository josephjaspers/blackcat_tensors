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
namespace impl {

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


/**
 * 2d Convolution of a 3d tensor with multiple 3d kernels.
 * Assume packed format.
 *
 */
template<class OutputPtr, class ImgPtr, class KrnlPtr>
void conv2d_3dtensor_multichannel(
			OutputPtr output,
			KrnlPtr krnl, size_t k_rows, size_t k_cols, size_t k_depth, size_t nkrnls,
			ImgPtr img,  size_t rows, size_t cols, size_t depth,
			size_t stride=1, size_t padding=0) {

	auto krnl_index = [&](auto krnl_index, auto depth, auto col, auto row) {
		return krnl_index * (k_rows * k_cols * k_depth)
				+ depth * (k_rows * k_cols)
				+ col * (k_rows)
				+ row;
	};

	auto img_index = [&](auto depth, auto col, auto row) {
		return depth * (rows * cols)
				+ col * (rows)
				+ row;
	};

	auto out_index = [&](auto krnl_idx, auto col, auto row) {
		size_t output_columns = (cols + padding - k_cols + 1) / stride;
		size_t output_rows    = (rows + padding - k_rows + 1) / stride;

		return krnl_idx * (output_columns * output_rows)
				+ col * (output_rows)
				+ row;
	};


	for (int c = -padding; c < cols + padding - k_cols + 1; c += stride) {
		for (int r = -padding; r < rows + padding - k_rows + 1; r += stride) {
			for (int k = 0; k < nkrnls; ++k) {
				float sum = 0;
				for (int kc = 0; kc < k_cols; ++kc) {
					for (int kr = 0; kr < k_rows; ++kr) {
						for (int d = 0; d < depth; ++d) {
							if (c+kc >= 0 && c+kc < cols &&
								r+kr >= 0 && r+kr < rows) {
								auto x = img[img_index(d, c+kc, r+kr)];
								auto w = krnl[krnl_index(k, d, kc, kr)];
								sum += w * x;
							}
						}
					}
				}
				output[out_index(k, c, r)] = sum;
			}
		}
	}
}

template<class DeltaPtr, class ImgDelta, class KrnlPtr>
void backward_data_conv2d_3dtensor_multichannel(
			//The deltas
			DeltaPtr output,

			//This is now the 'output'
			KrnlPtr krnl, size_t k_rows, size_t k_cols, size_t k_depth, size_t nkrnls,

			//ImgPtr is now the 'delta' storage
			ImgDelta img,  size_t rows, size_t cols, size_t depth,

			size_t stride=1, size_t padding=0) {

	auto krnl_index = [&](auto krnl_index, auto depth, auto col, auto row) {
		return krnl_index * (k_rows * k_cols * k_depth)
				+ depth * (k_rows * k_cols)
				+ col * (k_rows)
				+ row;
	};

	auto img_index = [&](auto depth, auto col, auto row) {
		return depth * (rows * cols)
				+ col * (rows)
				+ row;
	};

	auto out_index = [&](auto krnl_idx, auto col, auto row) {
		size_t output_columns = (cols + padding - k_cols + 1) / stride;
		size_t output_rows    = (rows + padding - k_rows + 1) / stride;

		return krnl_idx * (output_columns * output_rows)
				+ col * (output_rows)
				+ row;
	};

	//must zero the entire delta array
	for (int i = 0; i < rows * cols * depth; ++i) {
		img[i] = 0;
	}

	for (int c = -padding; c < cols + padding - k_cols + 1; c += stride) {
		for (int r = -padding; r < rows + padding - k_rows + 1; r += stride) {
			for (int k = 0; k < nkrnls; ++k) {
				float sum = 0;
				for (int kc = 0; kc < k_cols; ++kc) {
					for (int kr = 0; kr < k_rows; ++kr) {
						for (int d = 0; d < depth; ++d) {
							//if in bounds
							if (c+kc >= 0 && c+kc < cols &&
								r+kr >= 0 && r+kr < rows) {
								auto w = krnl[krnl_index(k, d, kc, kr)];
								img[img_index(d, c+kc, r+kr)] += w * output[out_index(k, c, r)];
							}
						}
					}
				}
			}
		}
	}
}


template<class DeltaPtr, class ImgDelta, class KrnlPtr>
void backward_filter_conv2d_3dtensor_multichannel(
			//The deltas
			DeltaPtr output,

			//This is now the 'output'
			KrnlPtr krnl, size_t k_rows, size_t k_cols, size_t k_depth, size_t nkrnls,

			//ImgPtr is now the 'delta' storage
			ImgDelta img,  size_t rows, size_t cols, size_t depth,

			size_t stride=1, size_t padding=0) {

	auto krnl_index = [&](auto krnl_index, auto depth, auto col, auto row) {
		return krnl_index * (k_rows * k_cols * k_depth)
				+ depth * (k_rows * k_cols)
				+ col * (k_rows)
				+ row;
	};

	auto img_index = [&](auto depth, auto col, auto row) {
		return depth * (rows * cols)
				+ col * (rows)
				+ row;
	};

	auto out_index = [&](auto krnl_idx, auto col, auto row) {
		size_t output_columns = (cols + padding - k_cols + 1) / stride;
		size_t output_rows    = (rows + padding - k_rows + 1) / stride;

		return krnl_idx * (output_columns * output_rows)
				+ col * (output_rows)
				+ row;
	};

	//must zero the entire delta array
	for (int i = 0; i < k_rows * k_cols * k_depth; ++i) {
		krnl[i] = 0;
	}

	for (int c = -padding; c < cols + padding - k_cols + 1; c += stride) {
		for (int r = -padding; r < rows + padding - k_rows + 1; r += stride) {
			for (int k = 0; k < nkrnls; ++k) {
				float sum = 0;
				for (int kc = 0; kc < k_cols; ++kc) {
					for (int kr = 0; kr < k_rows; ++kr) {
						for (int d = 0; d < depth; ++d) {
							//if in bounds
							if (c+kc >= 0 && c+kc < cols &&
								r+kr >= 0 && r+kr < rows) {
								krnl[krnl_index(k, d, kc, kr)] += img[img_index(d, c+kc, r+kr)] * output[out_index(k, c, r)];
							}
						}
					}
				}
			}
		}
	}
}


}
}
}




#endif /* CONVOLUTION2D_H_ */
