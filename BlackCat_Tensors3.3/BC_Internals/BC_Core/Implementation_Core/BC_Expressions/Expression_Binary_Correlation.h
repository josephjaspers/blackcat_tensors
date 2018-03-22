/*
 * Expression_Binary_Correlation.h
 *
 *  Created on: Mar 21, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_CORRELATION_H_
#define EXPRESSION_BINARY_CORRELATION_H_

#include "Expression_Base.h"

template<class filter, class image>
struct binary_expression_correlation : expression<binary_expression_correlation<filter, image>> {

	filter w;
	image  i;

	static_assert(filter::RANK() == image::RANK(), "correlation currently only support for same order tensors");
	static constexpr int RANK = filter::RANK();

	template<int rank>
	auto axpy(T sum) {
		if (rank == 0)
			return sum;
	}

	binary_expression_correlation(filter f_, image i_) : f(f_), i(i_) {}



	T operator [] () const {

	}
};




#endif /* EXPRESSION_BINARY_CORRELATION_H_ */
