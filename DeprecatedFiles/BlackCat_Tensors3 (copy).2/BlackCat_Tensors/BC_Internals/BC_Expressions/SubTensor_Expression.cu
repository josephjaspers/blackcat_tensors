/*
 * SubTensor_Expression.h
 *
 *  Created on: Jan 10, 2018
 *      Author: joseph
 */

#ifndef SUBTENSOR_EXPRESSION_H_
#define SUBTENSOR_EXPRESSION_H_

#include "../BlackCat_Internal_Definitions.h" //__BC_gcpu__


namespace BC {

template<class T, int rows, int cols,  int ld>
struct subMat_expression  {

	T array;


	inline __attribute__((always_inline)) __BC_gcpu__
	const auto& operator [] (int index) const {
		return array[index];
	}

	inline __attribute__((always_inline)) __BC_gcpu__
	auto& operator [] (int index) {
		return array[(index % rows) * ld  + (index % rows)];
	}
};
}

#endif /* SUBTENSOR_EXPRESSION_H_ */
