/*
 * Convolution.h
 *
 *  Created on: Aug 24, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_TENSORS_FUNCIONS_CONVOLUTION_H_
#define BLACKCATTENSORS_TENSORS_FUNCIONS_CONVOLUTION_H_

#include <type_traits>
#include "Host.h"

#ifdef __CUDACC__
#include "Device.cu"
#endif

namespace BC {
namespace tensors {
namespace exprs {
namespace functions {
namespace convolutions {

template<class SystemTag>
struct Convolution_Implementation;

/**
 * 2d Convolution of a 3d tensor with multiple 3d kernels.
 * Assume packed format.
 *
 */
template<class Stream, class Output, class Image, class Kernel>
static void convolution_common_assert(
		Stream stream,
		Output output,
		Image krnl,
		Kernel img,
		BC::size_t padding=0,
		BC::size_t stride=1) {

	using system_tag = typename Stream::system_tag;

	static_assert(std::is_same<system_tag, typename Output::system_tag>::value,
			"Output must have same system_tag as Stream argument");
	static_assert(std::is_same<system_tag, typename Kernel::system_tag>::value,
			"Kernel must have same system_tag as Stream argument");
	static_assert(std::is_same<system_tag, typename Image::system_tag>::value,
			"Image must have same system_tag as Stream argument");

	static_assert(Image::tensor_dimension == Kernel::tensor_dimension-1 ||
			Image::tensor_dimension == Kernel::tensor_dimension,
			"img tensor_dimension must equal krnl tensor_dimension - 1");
	static_assert(Output::tensor_dimension == Kernel::tensor_dimension-1 ||
			Output::tensor_dimension == Kernel::tensor_dimension,
			"output tensor_dimension must equal krnl tensor_dimension - 1");

	BC_ASSERT(output.cols() == img.cols() + padding - krnl.cols() + 1,
			"Invalid output column dimension");
	BC_ASSERT(output.rows() == img.rows() + padding - krnl.rows() + 1,
			"Invalid output column dimension");
	BC_ASSERT(output.dimension(2) == krnl.dimension(3),
			"Invalid output column dimension");
}


template<class Stream, class Output, class Image, class Kernel>
static void conv2d(
			Stream stream,
			Output output,
			Kernel krnl,
			Image img,
			BC::size_t stride=1, BC::size_t padding=0) {

	convolution_common_assert(
			stream,
			output,
			krnl,
			img,
			stride,
			padding);

	using system_tag = typename Stream::system_tag;
	using implementation = Convolution_Implementation<system_tag>;
	implementation::conv2d(stream, output, krnl, img);
}

}
}
}
}
}



#endif /* CONVOLUTION_H_ */
