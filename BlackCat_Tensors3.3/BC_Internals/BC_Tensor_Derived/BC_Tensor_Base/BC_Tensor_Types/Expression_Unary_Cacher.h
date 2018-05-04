/*
 * Expression_Unary_Pointwise.cu
 *
 *  Created on: Jan 25, 2018
 *      Author: joseph
 */
#ifndef EXPRESSION_UNARY_CACHER_CU_
#define EXPRESSION_UNARY_CACHER_CU_

#include "Expression_Base.h"

namespace BC {


struct _cache;

template<class lv, class rv>
class binary_expression<lv, rv, _cache> : public expression_base<binary_expression<lv, rv, _cache>> {
public:
	using scalar_type =  _scalar<lv>;
	mutable lv left;
	rv right;

	mutable bool* is_cached = new bool[left.size()] {0};
	mutable bool* is_locked = new bool[right.size()] {0};

	__BCinline__ static constexpr int DIMS() { return rv::DIMS(); }
	__BCinline__ static constexpr int CONTINUOUS() { return rv::CONTINUOUS(); }

	__BCinline__  binary_expression(lv l_, rv r_) : left(l_), right(r_) {}

	__BCinline__  const auto innerShape() const 			{ return left.innerShape(); }
	__BCinline__  const auto outerShape() const 			{ return left.outerShape(); }

	__BCinline__ const auto& operator [](int index) const {
		if (is_cached[index])
			return left[index];
		else if (is_locked[index]) {
			while (is_locked[index]) {
			}
			return left[index];
		} else {
			is_locked[index] = true;
			left[index] = right[index];
			is_locked[index] = false;
			is_cached[index] = true;
			return left[index];
		}
	}

	template<class... integers> __BCinline__
	const auto& operator ()(int index, integers... indices) const {

		if (is_cached[index])
			return left(index, indices...);
		else if (is_locked[index]) {
			while (is_locked[index]) {
			}
			return left[index];
		} else {
			is_locked[index] = true;
			left(index, indices...) = right(index, indices...);
			is_locked[index] = false;
			is_cached[index] = true;
			return left[index];
		}
	}
};
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
