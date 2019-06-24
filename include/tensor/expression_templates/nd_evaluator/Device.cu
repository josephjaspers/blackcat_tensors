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
namespace evaluator {

template<>
struct Evaluator<device_tag> {

	 template<int Dimensions>
	 struct nd_evaluator_func {

		struct n1 {
			template<class Expression>
			static void eval(Expression expression, cudaStream_t stream=cudaStream_t()) {
				gpu_impl::eval<<<blocks(expression.size()), threads(), 0, stream>>>(expression);
			}
		};
		struct n2 {
			template<class Expression>
			static void eval(Expression expression, cudaStream_t stream=cudaStream_t()) {
				gpu_impl::eval2d<<<blocks(expression.size()), threads(), 0, stream>>>(expression);
			}
		};
		struct n3 {
			template<class Expression>
			static void eval(Expression expression, cudaStream_t stream=cudaStream_t()) {
				gpu_impl::eval3d<<<blocks(expression.size()), threads(), 0, stream>>>(expression);
			}
		};
		struct n4 {
			template<class Expression>
			static void eval(Expression expression, cudaStream_t stream=cudaStream_t()) {
				gpu_impl::eval4d<<<blocks(expression.size()), threads(), 0, stream>>>(expression);
			}
		};
		struct n5 {
			template<class Expression>
			static void eval(Expression expression, cudaStream_t stream=cudaStream_t()) {
				gpu_impl::eval5d<<<blocks(expression.size()), threads(), 0, stream>>>(expression);
			}
		};

		using run = std::conditional_t<(Dimensions <= 1), n1,
						std::conditional_t<(Dimensions == 2), n2,
							std::conditional_t<(Dimensions == 3), n3,
								std::conditional_t<(Dimensions == 4), n4, n5>>>>;

		template<class Expression>
		static void eval(Expression expression) {
			run::eval(expression);
		}

		template<class Expression, class Stream>
		static void eval(Expression expression, Stream stream) {
				run::eval(expression, stream);

		}
	};

	template<int dimensions, class Expression>
	static void nd_evaluate(Expression expression) {
		nd_evaluator_func<dimensions>::eval(expression);
	}
	template<int dimensions, class Expression, class Stream>
	static void nd_evaluate(Expression expression, Stream stream) {
		nd_evaluator_func<dimensions>::eval(expression, stream);
	}

};

}
}

#endif /* MATHEMATICS_CPU_H_ */

#endif //if cudda cc defined
