/*
 * Caffe.h
 *
 *  Created on: Oct 31, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_FUNCTIONS_H_
#define BLACKCAT_NEURALNETWORK_FUNCTIONS_H_

#include "maxpooling.h"
#include "maxpooling.cu"
#include "im2col.h"
#include "im2col.cu"

namespace bc {

template<class Stream, class Indexes, class Image, class ImageOut>
void max_pooling_forward(
		Stream stream,
		Image image,
		ImageOut out,
		Indexes mask,
		Dim<2> krnl_shape,
		Dim<2> padding = Dim<2>().fill(0),
		Dim<2> strides = {-1,-1}) {

	if (strides == Dim<2>{ -1,-1 })
		strides = krnl_shape;

	BC_ASSERT((out.inner_shape().template subdim<0,2>() ==
			(image.inner_shape().template subdim<0,2>() + padding*2)/strides),
			"ASSERT MAX_POOLING_FORWARD"
			"\nout.inner_shape() == "
			"(image.inner_shape() + padding)/strides");
	BC_ASSERT(out.dim(2) == image.dim(2), "numb channels must be the same");
	BC_ASSERT(out.dim(3) == image.dim(3), "batch size must be the same");

	stream.enqueue([=]() {
		bc::caffe::MaxPoolForward(
				typename Stream::system_tag(),
				image.data(),
				image.dim(3),
				image.dim(2),
				image.dim(0), image.dim(1),
				out.dim(0), out.dim(1),
				krnl_shape[0], krnl_shape[1],
				strides[0], strides[1],
				padding[0], padding[1],
				out.data(), mask.data());
	});
}

template<
	class Stream,
	class Indexes,
	class Image,
	class ImageOut>
void max_pooling_backward(
		Stream stream,
		Image image,         //output delta (not initialized)
		ImageOut delta,      //delta from upper layer
		Indexes mask,        //indicies of delta from upper layer
		Dim<2> krnl_shape,
		Dim<2> padding = Dim<2>().fill(0 ),
		Dim<2> strides = Dim<2>().fill(-1))
{
	static_assert(std::is_same<
			int,
			typename Indexes::value_type>::value,
			"Mask must be int");

	static_assert(std::is_same<
			typename Image::value_type,
			typename ImageOut::value_type>::value,
			"Delta/Image value_type must be the same");

	if (strides == Dim<2>{ -1,-1 })
		strides = krnl_shape;

	BC_ASSERT((delta.inner_shape().template subdim<0,2>() ==
			(image.inner_shape().template subdim<0,2>() + padding*2)/strides),
			"ASSERT MAX_POOLING_FORWARD"
			"\nout.inner_shape() == "
			"(image.inner_shape() + padding)/strides");
	BC_ASSERT(delta.dim(2) == image.dim(2), "numb channels must be the same");
	BC_ASSERT(delta.dim(3) == image.dim(3), "batch size must be the same");

	using system_tag = typename Stream::system_tag;

	stream.enqueue([=]() {
		bc::algorithms::fill(
				stream,
				image.data(),
				image.data() + image.size(),
				0.0);

		bc::caffe::MaxPoolBackward(
				system_tag(),
				delta.data(), mask.data(),
				image.dim(3), image.dim(2),
				image.dim(0), image.dim(1),
				delta.dim(0), delta.dim(1),
				krnl_shape[0], krnl_shape[1],
				strides[0], strides[1],
				padding[0], padding[1],
				image.data());
	});
}

template<
	class Stream,
	class ColumnImage,
	class Image>
void im2col(
		Stream stream,
		ColumnImage col_image,
		Image image,
		bc::Dim<3> krnl_shape,
		bc::Dim<2> padding = bc::Dim<2>().fill(0),
		bc::Dim<2> strides = bc::Dim<2>().fill(1),
		bc::Dim<2> dilation = bc::Dim<2>().fill(1),
		int numb_spatial_axis=2) {

	static_assert(ColumnImage::tensor_dim == 2,
			"ColumnImage must be a matrix");
	static_assert(Image::tensor_dim == 3,
			"2d Convolution expects a 3d-image input");

	using system_tag = typename Stream::system_tag;

	stream.enqueue([=]() {
		bc::nn::functions::im2col(
				system_tag(),
				image.data(),
				image.dim(2),
				image.dim(1), image.dim(0),
				krnl_shape[1], krnl_shape[0],
				padding[1], padding[0],
				strides[1], strides[0],
				dilation[1], dilation[0],
				col_image.data());
	});
}

template<
	class Stream,
	class ColumnImage,
	class Image>
void col2im(
		Stream stream,
		ColumnImage col_image,
		Image image,
		bc::Dim<3> krnl_shape,
		bc::Dim<2> padding = bc::Dim<2>(),
		bc::Dim<2> strides = bc::Dim<2>().fill(1),
		bc::Dim<2> dilation = bc::Dim<2>().fill(1)) {

	static_assert(ColumnImage::tensor_dim == 2,
			"ColumnImage must be a matrix");
	static_assert(Image::tensor_dim == 3,
			"2d Convolution expects a 3d-image input");

	stream.enqueue([=]() {
		bc::nn::functions::col2im(
				typename Stream::system_tag(),
				image.data(),
				image.dim(2),
				image.dim(1), image.dim(0),
				krnl_shape[1], krnl_shape[0],
				padding[1], padding[0],
				strides[1], strides[0],
				dilation[1], dilation[0],
				col_image.data());
	});
}
}

#endif /* CAFFE_H_ */
