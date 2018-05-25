/*
 * Expressions_Binary_Correlation.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONS_BINARY_CORRELATION_STACK_H_
#define EXPRESSIONS_BINARY_CORRELATION_STACK_H_

#include "Expression_Base.h"
#include "Expression_Binary_Pointwise.h"
#include <utility>

namespace BC {

struct inner;
template<int,class> struct _x_corr_stack;

template<class lv, class rv, int corr_dimension>
struct binary_expression<lv, rv, _x_corr_stack<corr_dimension,inner>> : expression_base<binary_expression<lv, rv, _x_corr_stack<corr_dimension,inner>>> {

	__BCinline__ static constexpr int DIMS() { return corr_dimension + 1; }
	__BCinline__ static constexpr int ITERATOR() { return corr_dimension + 1; }
	using scalar = _scalar<lv>;

	static_assert(lv::DIMS() - 1 == rv::DIMS(), "CORRELATION CURRENTLY ONLY SUPPORTED FOR SAME ORDER TENSORS");
	static_assert(DIMS() <= 3, "CORRELATION MOST MOVEMENT IS LIMITED TO 3D");

	int is[DIMS()];
	int os[DIMS()];

	lv left;  //krnl
	rv right; //img

	binary_expression(lv l_, rv r_) :left(l_), right(r_) {
		is[0] = right.dimension(0) - left.dimension(0) + 1;
		os[0] = is[0];
		for (int i = 1; i < DIMS() - 1; ++i) {
			is[i] = right.dimension(i) - left.dimension(i) + 1;
			os[i] = is[i] * os[i - 1];
		}
		is[DIMS() - 1] = left.dimension(DIMS() - 1);
		os[DIMS() - 1] = is[DIMS() - 1] * os[DIMS() - 2];

	}

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	struct DISABLED;
	template<int x>
	using conditional_int = std::conditional_t<x == corr_dimension, int, DISABLED>;

	///The last integer index of the axpy is always the "stack" dimension.
	///therefor we remove the integer (via pop_tail), and call right().

	template<class... ints> __BCinline__
	scalar axpy (int stack, conditional_int<1> x, ints... location) const {
		scalar sum = 0;
		for (int i = 0 ; i < left.rows(); ++i) {
			sum += left(i, stack) * right(i + x, location...);
		}
		return sum;
	}
	template<class... ints> __BCinline__
	scalar axpy (int stack, conditional_int<2> x, int y, ints... indexes) const {
		scalar sum = 0;
		for (int n = 0; n < left.dimension(0); ++n) {
			for (int m = 0 ; m < left.dimension(1); ++m) {
				sum += left(m, n, stack) * right(m + x, n + y, indexes...);

			}
		}
		return sum;
	}
	template<class... ints> __BCinline__
	scalar axpy (int stack, conditional_int<3> x, int y, int z, ints... indexes) const {
		scalar sum = 0;
		for (int k = 0; k < left.dimension(2); ++k) {
			for (int n = 0; n < left.cols(); ++n) {
				for (int m = 0 ; m < left.rows(); ++m) {
					sum += left(m, n, k, stack) * right(m + x, n + y, k + z, indexes...);
				}
			}
		}
		return sum;
	}


	template<class... integers> __BCinline__  scalar operator ()(integers... ints) const {
		return PPack::pop_tail([&](auto&... xs) { return this->axpy(xs...); }, PPack::tail(ints...), ints...);
	}
};

}


#endif /* EXPRESSIONS_BINARY_CORRELATION_H_ */
