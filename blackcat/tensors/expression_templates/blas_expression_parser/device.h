/*
 * Device.h
 *
 *  Created on: Apr 24, 2019
 *      Author: joseph
 */
#ifdef __CUDACC__
#ifndef BLACKCAT_EXPRESSION_TEMPLATES_BLAS_TOOLS_DEVICE_H_
#define BLACKCAT_EXPRESSION_TEMPLATES_BLAS_TOOLS_DEVICE_H_

#include "device_impl.cu"
#include "common.h"

namespace bc {
namespace tensors {
namespace exprs { 
namespace blas_expression_parser {

template<>
struct Blas_Expression_Parser<device_tag>:
		Common_Tools<Blas_Expression_Parser<device_tag>> {

	template<class Stream, class Scalar, class... Scalars>
	static void scalar_multiply(Stream stream, Scalar output, Scalars... vals)
	{
		device_detail::calculate_alpha<<<1, 1, 0, stream>>>(output, vals...);
	}

	template<class ValueType, int Value>
	static const ValueType* scalar_constant() {

		struct initializer {
			static ValueType* init() {
				ValueType tmp = Value;
				ValueType* scalar = nullptr;

				constexpr int size = sizeof(ValueType);

				BC_CUDA_ASSERT(
						cudaMallocManaged((void**)&scalar, size));
				BC_CUDA_ASSERT(
						cudaMemcpy(scalar, &tmp, size, cudaMemcpyHostToDevice));

				return scalar;
			}
		};

		static ValueType* scalar = initializer::init();
		return scalar;
	}
};

}
}
}
}


#endif /* DEVICE_H_ */
#endif
