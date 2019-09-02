/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifdef __CUDACC__
#ifndef BC_MATHEMATICS_DEVICE_H_
#define BC_MATHEMATICS_DEVICE_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "Device_Impl.cu"
#include <iostream>


namespace BC {
namespace tensors {
namespace exprs {
namespace evaluator {

template<>
struct Evaluator<device_tag> {

	template<int Dimensions, class Expression, class Stream>
	static void nd_evaluate(Expression expression, Stream stream) {

		struct n1 {
			static void eval(Expression expression, cudaStream_t stream) {
				gpu_impl::eval<<<calculate_block_dim(expression.size()), calculate_threads(), 0, stream>>>(expression);
			}
		};
		struct n2 {
			static void eval(Expression expression, cudaStream_t stream) {
				gpu_impl::eval2d<<<calculate_block_dim(expression.size()), calculate_threads(), 0, stream>>>(expression);
			}
		};
		struct n3 {
			static void eval(Expression expression, cudaStream_t stream) {
				gpu_impl::eval3d<<<calculate_block_dim(expression.size()), calculate_threads(), 0, stream>>>(expression);
			}
		};
		struct n4 {
			static void eval(Expression expression, cudaStream_t stream) {
				gpu_impl::eval4d<<<calculate_block_dim(expression.size()), calculate_threads(), 0, stream>>>(expression);
			}
		};
		struct n5 {
			static void eval(Expression expression, cudaStream_t stream) {
				gpu_impl::eval5d<<<calculate_block_dim(expression.size()), calculate_threads(), 0, stream>>>(expression);
			}
		};

		using run = std::conditional_t<(Dimensions <= 1), n1,
						std::conditional_t<(Dimensions == 2), n2,
							std::conditional_t<(Dimensions == 3), n3,
								std::conditional_t<(Dimensions == 4), n4, n5>>>>;

		stream.enqueue([=]() {
			run::eval(expression, stream);
		});
	}

};

}
}
}
}

#endif /* MATHEMATICS_CPU_H_ */

#endif //if cudda cc defined
