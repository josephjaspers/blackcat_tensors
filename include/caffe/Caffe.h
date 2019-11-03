/*
 * Caffe.h
 *
 *  Created on: Oct 31, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_CAFFE_CAFFE_H_
#define BLACKCAT_CAFFE_CAFFE_H_

#include "Img2Col.cu"
#include "Img2Col.h"

namespace BC {

template<
	class Stream,
	class ColumnImage,
	class Image>
void im2col(
		Stream stream,
		ColumnImage col_image,
		Image image,
		BC::Dim<3> krnl_shape,
		BC::Dim<2> padding = BC::Dim<2>(),
		BC::Dim<2> strides = BC::Dim<2>().fill(1),
		BC::Dim<2> dilation = BC::Dim<2>().fill(1)) {

	static_assert(ColumnImage::tensor_dimension == 2,
			"ColumnImage must be a matrix");
	static_assert(Image::tensor_dimension == 3,
			"2d Convolution expects a 3d-image input");

	using system_tag = typename Stream::system_tag;

	stream.enqueue([=]() {
		 im2col(
				system_tag(),
				image.memptr(),
				image.dimension(2),
				image.dimension(1), image.dimension(0),
				krnl_shape[1], krnl_shape[0],
				padding[1], padding[0],
				strides[1], strides[0],
				dilation[1], dilation[0],
				col_image.memptr());
	});
}

template<
	class Stream,
	class ColumnImage,
	class Image,
	int SpacialAxis>
void im2col_nd(
		Stream stream,
		ColumnImage col_image,
		Image image,
		BC::Dim<SpacialAxis> krnl_shape,
		BC::Dim<SpacialAxis> padding = BC::Dim<SpacialAxis>(),
		BC::Dim<SpacialAxis> strides = BC::Dim<SpacialAxis>().fill(1),
		BC::Dim<SpacialAxis> dilation = BC::Dim<SpacialAxis>().fill(1)) {

	constexpr bool is_batched = Image::tensor_dimension == SpacialAxis + 1;
	static_assert(ColumnImage::tensor_dimension == 2 + is_batched,
			"Invalid ColumnImage dimension");
	static_assert(Image::tensor_dimension == SpacialAxis + is_batched,
			"Invalid Image dimension");

	using system_tag = typename Stream::system_tag;

	//assume non-strided data (must be packed format)
	auto img_shape = image.get_shape().inner_shape().reverse();
	auto col_shape = col_image.get_shape().inner_shape().reverse();

	stream.enqueue([=]() {
		 im2col_nd(
				system_tag(),
				image.memptr(),
				SpacialAxis,
				img_shape.data(),
				col_shape.data(),
				krnl_shape.reverse().data(),
				padding.reverse().data(),
				strides.reverse().data(),
				dilation.reverse().data(),
				col_image.memptr());
	});
}
template <typename Dtype>
void im2col(BC::host_tag, const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	BC::caffe::im2col_cpu(
			data_im, channels,
			height, width,
			kernel_h, kernel_w,
			pad_h, pad_w,
			stride_h, stride_w,
			dilation_h, dilation_w, data_col);
}

#ifdef __CUDACC__
template <typename Dtype>
void im2col(BC::device_tag, const Dtype* data_im, const int channels,
		const int height, const int width,
		const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		Dtype* data_col) {

	BC::caffe::im2col_gpu(
			data_im, channels,
			height, width,
			kernel_h, kernel_w,
			pad_h, pad_w,
			stride_h, stride_w,
			dilation_h, dilation_w, data_col);
}
#endif


template <typename Dtype>
void im2col_nd(BC::host_tag, const Dtype* data_im, const int num_spatial_axes,
		const int* im_shape, const int* col_shape,
		const int* kernel_shape, const int* pad, const int* stride,
		const int* dilation, Dtype* data_col) {
	BC::caffe::im2col_nd_cpu(data_im, num_spatial_axes, im_shape, col_shape,
									kernel_shape, pad, stride, dilation, data_col);
}


#ifdef __CUDACC__
template <typename Dtype>
void im2col_nd(BC::device_tag, const Dtype* data_im, const int num_spatial_axes,
		const int* im_shape, const int* col_shape,
		const int* kernel_shape, const int* pad, const int* stride,
		const int* dilation, Dtype* data_col) {
	const bool kIm2Col = true;
	BC::caffe::im2col_nd_gpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
									kernel_shape, pad, stride, dilation, data_col);
}
#endif





}

#endif /* CAFFE_H_ */
