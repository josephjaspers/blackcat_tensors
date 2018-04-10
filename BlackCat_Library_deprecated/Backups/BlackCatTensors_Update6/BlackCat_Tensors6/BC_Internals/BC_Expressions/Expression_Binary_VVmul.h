/*
 * Expression_Binary_VVmul.h
 *
 *  Created on: Dec 20, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_VVMUL_H_
#define EXPRESSION_BINARY_VVMUL_H_
#include <cmath>
#include "../BlackCat_Internal_GlobalUnifier.h"

namespace BC {

template<
	class T,
	class COL_VECTOR,
	class ROW_VECTOR
>
struct binary_expression_VVmul_outer : expression<T, binary_expression_VVmul_outer<T, COL_VECTOR, ROW_VECTOR>>{

	/*
	 * Outer product (always between Col Vector and Row Vector
	 */

	COL_VECTOR vecL;
	ROW_VECTOR vecR;

	binary_expression_VVmul_outer(const COL_VECTOR& lv, const ROW_VECTOR& rv) : vecL(lv), vecR(rv) {}
	binary_expression_VVmul_outer(const binary_expression_VVmul_outer&) = default;

	T operator [] (int index) const {
		return vecL.array[std::ceil(index / vecL.rows())] * vecR.array[index % vecR.cols()];
	}


};

}

#endif /* EXPRESSION_BINARY_VVMUL_H_ */
