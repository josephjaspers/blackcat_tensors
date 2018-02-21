/*
 * BC_Expression_Binary.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */
#ifdef  __CUDACC__
#ifndef EXPRESSION_BINARY_POINTWISE_SAME_H_
#define EXPRESSION_BINARY_POINTWISE_SAME_H_


#include "Expression_Base.cu"
namespace BC {


template<class T, class op, class lv, class rv>
class binary_expression;

template<class T, class operation, class lv, class rv>
struct binary_expression : public expression<T, binary_expression<T, operation, lv, rv>> {

	operation oper;

	const lv& left;
	const rv& right;

	int rows() const { return left.rows(); }
	int cols() const { return left.cols(); }
	int LD_rows() const { return left.LD_rows(); }

	int size() const { return left.size(); }
	void printDimensions() const {  left.printDimensions(); }
	const int* InnerShape() const { return left.InnerShape(); }
	const auto addressOf(int offset) const { return binary_expression(addressOf(left, offset), addressOf(right, offset)); }


	template<class L, class R>
	inline __attribute__((always_inline)) binary_expression(const L& l, const R& r) :
			left(l), right(r) {

	}



	inline __attribute__((always_inline)) __BC_gcpu__ auto operator [](int index) const -> decltype(oper(left[index], right[index])) {
		return oper(left[index], right[index]);
	}
};


}

#endif /* EXPRESSION_BINARY_POINTWISE_SAME_H_ */
#endif
