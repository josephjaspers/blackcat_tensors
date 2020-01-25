/*
 * bc_im2col.h
 *
 *  Created on: Jan 9, 2020
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef BC_IM2COL_CU_
#define BC_IM2COL_CU_

#include "../../tensors.h"
#include "common.h"
#include "../../operations.h"
#include "../../common.h"
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace bc {
namespace nn {
namespace functions {

template <typename Dtype> __global__
void im2col_kernel(
		int size,
		const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	using bc::tensors::exprs::Kernel_Array;
	using column_tensor_type = Kernel_Array<bc::Shape<5>, Dtype, bc::device_tag>;
	using image_tensor_type = Kernel_Array<bc::Shape<3>, Dtype, bc::device_tag>;

	int image_h_end = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1));
	int image_w_end =  (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1));

	const int output_h = image_h_end / stride_h + 1;
	const int output_w = image_w_end / stride_w + 1;

	auto column_tensor = column_tensor_type(
			bc::Dim<5> { kernel_h, kernel_w, channels, output_w, output_h },
			data_col);

	auto image_tensor = image_tensor_type(
			bc::Dim<3> { height, width, channels },
			const_cast<Dtype*>(data_im));

	BC_CUDA_KERNEL_LOOP_X(index, size) {
		int kernel_row = index % kernel_h;
		int kernel_col = (index / kernel_h) % kernel_w;
		int channel = (index / kernel_h / kernel_w) % channels;
		int image_row = (index / kernel_h / kernel_w) % (height-kernel_h+1);
		int image_col = (index / kernel_h / kernel_w / (height-kernel_h+1)) % (width-kernel_w+1);

		column_tensor(image_col, image_row, channel, kernel_col, kernel_row)
			= image_tensor(channel, image_col+kernel_col, image_row+kernel_row);
	}
}

template <typename Dtype>
void im2col(
		bc::device_tag,
		const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,

		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,

		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	int image_col_max = width-kernel_w+1;
	int image_row_max = height-kernel_h+1;

	int size = channels
			* image_col_max * image_row_max
			* kernel_h * kernel_w;

	im2col_kernel<<<
			bc::calculate_block_dim(size),
			bc::calculate_threads(size)>>>(
					size,
					data_im,
					channels,
					height, width,
					kernel_h, kernel_w,
					pad_h, pad_w,
					stride_h, stride_w,
					dilation_h, dilation_w,
					data_col);
}

template <typename Dtype> __global__
void col2im_kernel(int size,
		const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,

		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,

		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	using bc::tensors::exprs::Kernel_Array;
	using column_tensor_type = Kernel_Array<bc::Shape<5>, Dtype, bc::device_tag>;
	using image_tensor_type = Kernel_Array<bc::Shape<3>, Dtype, bc::device_tag>;

	int image_h_end = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1));
	int image_w_end =  (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1));

	int output_h = image_h_end / stride_h + 1;
	int output_w = image_w_end / stride_w + 1;

	auto column_tensor = column_tensor_type(
			Dim<5> { kernel_h, kernel_w, channels, output_w, output_h },
			data_col);

	auto image_tensor = image_tensor_type(
			Dim<3> { height, width, channels },
			const_cast<Dtype*>(data_im));

	BC_CUDA_KERNEL_LOOP_X(index, size) {
		int kernel_row = index % kernel_h;
		int kernel_col = (index / kernel_h) % kernel_w;
		int channel = (index / kernel_h / kernel_w) % channels;
		int image_row = (index / kernel_h / kernel_w) % (height-kernel_h+1);
		int image_col = (index / kernel_h / kernel_w / (height-kernel_h+1)) % (width-kernel_w+1);

		bc::oper::Device_Atomic_Add::apply(
			image_tensor(channel, image_col+kernel_col, image_row+kernel_row),
			column_tensor(image_col, image_row, channel, kernel_col, kernel_row));
	}
}

template <typename Dtype>
void col2im(
		bc::device_tag,
		const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,

		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,

		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	int image_col_max = width-kernel_w+1;
	int image_row_max = height-kernel_h+1;

	int size = channels
			* image_col_max * image_row_max
			* kernel_h * kernel_w;

	col2im_kernel<<<
			bc::calculate_block_dim(size),
			bc::calculate_threads(size)>>>(
					size,
					data_im,
					channels,
					height, width,
					kernel_h, kernel_w,
					pad_h, pad_w,
					stride_h, stride_w,
					dilation_h, dilation_w,
					data_col);
}



}
}
}



#endif /* BC_IM2COL_H_ */
#endif
