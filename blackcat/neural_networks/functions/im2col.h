/*
 * bc_im2col.h
 *
 *  Created on: Jan 9, 2020
 *      Author: joseph
 */

#ifndef BC_IM2COL_H_
#define BC_IM2COL_H_

namespace bc {
namespace nn {
namespace functions {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col(
		bc::host_tag,
		const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,

		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,

		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	int image_h_end = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1));
	int image_w_end =  (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1));

	const int output_h = image_h_end / stride_h + 1;
	const int output_w = image_w_end / stride_w + 1;

	using bc::tensors::exprs::Kernel_Array;

	using column_tensor_type = Kernel_Array<bc::Shape<5>, Dtype, bc::host_tag>;
	using image_tensor_type = Kernel_Array<bc::Shape<3>, Dtype, bc::host_tag>;

	auto column_tensor = column_tensor_type(
			{kernel_h, kernel_w, channels, output_w, output_h},
			data_col);

	auto image_tensor = image_tensor_type(
			{height, width, channels},
			const_cast<Dtype*>(data_im));

	//TODO add support for strides, and padding
	for (int channel = 0; channel < channels; channel++) {
		for (int image_col = 0; image_col < width-kernel_w+1; image_col++) {
			for (int image_row = 0; image_row < height-kernel_w+1; image_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
						column_tensor(image_col, image_row, channel, kernel_col, kernel_row)
								= image_tensor(channel, image_col+kernel_col, image_row+kernel_row);
					}
				}
			}
		}
	}
}

template <typename Dtype>
void col2im(
		bc::host_tag,
		const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,

		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,

		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	int image_h_end = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1));
	int image_w_end =  (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1));

	const int output_h = image_h_end / stride_h + 1;
	const int output_w = image_w_end / stride_w + 1;

	using bc::tensors::exprs::Kernel_Array;

	using column_tensor_type = Kernel_Array<bc::Shape<5>, Dtype, bc::host_tag>;
	using image_tensor_type = Kernel_Array<bc::Shape<3>, Dtype, bc::host_tag>;

	auto column_tensor = column_tensor_type(
			{kernel_h, kernel_w, channels, output_w, output_h},
			data_col);

	auto image_tensor = image_tensor_type(
			{height, width, channels},
			const_cast<Dtype*>(data_im));

	//TODO add support for strides, and padding
	for (int channel = 0; channel < channels; channel++) {
		for (int image_col = 0; image_col < width-kernel_w+1; image_col++) {
			for (int image_row = 0; image_row < height-kernel_w+1; image_row++) {
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
					for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
						image_tensor(channel, image_col+kernel_col, image_row+kernel_row)
								+= column_tensor(image_col, image_row, channel, kernel_col, kernel_row);
					}
				}
			}
		}
	}
}


}
}
}



#endif /* BC_IM2COL_H_ */
