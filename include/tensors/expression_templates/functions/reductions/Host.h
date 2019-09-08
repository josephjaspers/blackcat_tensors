/*
 * Host_Add.h
 *
 *  Created on: Aug 27, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_TENSORS_FUNCTIONS_REDUCTIONS_HOST_REDUCE_H_
#define BLACKCATTENSORS_TENSORS_FUNCTIONS_REDUCTIONS_HOST_REDUCE_H_

namespace BC {
namespace tensors {
namespace exprs {
namespace functions {

template<class>
struct Reduce;

template<>
struct Reduce<BC::host_tag> {

	template<class Stream, class ScalarOutput,  class Expression>
	static void sum(Stream stream, ScalarOutput output, Expression expression) {

		auto function = [&]() {

			using value_type = typename Expression::value_type;

			value_type& total = output[0];
			total = 0;
	#if defined(_OPENMP) && !defined(BC_NO_OPENMP)
	#pragma omp parallel for reduction(+:total)
	#endif
			for (BC::size_t i = 0; i < expression.size(); ++i) {
				total += expression[i];
			}
		};
		stream.enqueue(function);
	}
};

}
}
}
}


#endif /* HOST_ADD_H_ */
