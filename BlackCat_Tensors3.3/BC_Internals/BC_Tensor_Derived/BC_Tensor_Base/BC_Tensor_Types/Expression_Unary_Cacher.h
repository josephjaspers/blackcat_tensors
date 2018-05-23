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

template<class mathlib>
struct cache;

template<class lv, class rv, class ml>
class binary_expression<lv, rv, cache<ml>> : public expression_base<binary_expression<lv, rv, cache<ml>>> {
public:
	using scalar_type =  _scalar<lv>;
	mutable lv left;
	rv right;

	mutable bool* is_cached;

	__BCinline__ static constexpr int DIMS() { return rv::DIMS(); }
	__BCinline__ static constexpr int CONTINUOUS() { return rv::CONTINUOUS(); }

	__BCinline__  binary_expression(lv l_, rv r_) : left(l_), right(r_) {
		 ml::zero_initialize(is_cached, right.size());
	}

	__BCinline__  const auto innerShape() const 			{ return left.innerShape(); }
	__BCinline__  const auto outerShape() const 			{ return left.outerShape(); }

	__BCinline__ const auto& operator [](int index) const {
		if (is_cached[index])
			return left[index];
//		else if (is_locked[index]) {
//			while (is_locked[index]) {
//			}
//			return left[index];
		else {
//			is_locked[index] = true;
			left[index] = right[index];
//			is_locked[index] = false;
			is_cached[index] = true;
			return left[index];
		}
	}

	template<class... integers> __BCinline__
	const auto& operator ()(integers... indices) const {
		int index = this->scal_index(indices...);
		if (is_cached[index])
			return left(indices...);
//		else if (is_locked[index]) {
//			while (is_locked[index]) {
//			}
//			return left[index];
//		}
	else {
//			is_locked[index] = true;
			left(indices...) = right(indices...);
//			is_locked[index] = false;
			is_cached[index] = true;
			return left[index];
		}
	}


	void destroy() {
		ml::destroy(is_cached);
	}
};
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
