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
	lv left;
	rv right;

	mutable bool* is_cached = new bool[left.size()] {0};
	mutable bool* is_locked = new bool[right.size()] {0};

	__BCinline__ static constexpr int DIMS() { return lv::DIMS(); }
	__BCinline__ static constexpr int CONTINUOUS() { return rv::CONTINUOUS(); }

	__BCinline__  binary_expression(lv l_, rv r_) : left(l_), right(r_) {
		for (int i = 0; i < array.size(); ++i) {
			is_cached[i] = false;
			mutex[i] = false;
		}
	}

	__BCinline__  const auto innerShape() const 			{ return array.innerShape(); }
	__BCinline__  const auto outerShape() const 			{ return array.outerShape(); }

	__BCinline__ const auto& operator [](int index) const {
		if (is_cached[index])
			return cache[index];
		else if (mutex[index]) {
			while (mutex[index]) {
			}
			return cache[index];
		} else {
			mutex[index] = true;
			cache[index] = array[index];
			mutex[index] = false;
			is_cached[index] = true;
			return cache[index];
		}
	}
	__BCinline__ auto& operator [](int index) {
		if (is_cached[index])
			return cache[index];
		else if (mutex[index]) {
			while (mutex[index]) {
			}
			return cache[index];
		} else {
			mutex[index] = true;
			cache[index] = array[index];
			mutex[index] = false;
			is_cached[index] = true;
			return cache[index];
		}
	}
	template<class... integers>__BCinline__ const auto& operator ()(integers... indices) const {
		int index = array.point_index(indices...);

		if (is_cached[index])
			return cache[index];
		else if (mutex[index]) {
			while (mutex[index]) {
			}
			return cache[index];
		} else {
			mutex[index] = true;
			cache[index] = array(indices...);
			mutex[index] = false;
			is_cached[index] = true;
			return cache[index];
		}
	}
	template<class... integers>	__BCinline__ auto& operator ()(integers... indices) {
		int index = array.point_index(indices...);

		if (is_cached[index])
			return cache[index];
		else if (mutex[index]) {
			while (mutex[index]) {
			}
			return cache[index];
		} else {
			mutex[index] = true;
			cache[index] = array(indices...);
			mutex[index] = false;
			is_cached[index] = true;
			return cache[index];
		}
	}
	__BCinline__ const auto slice(int i) const {
		return unary_expression<decltype(array.slice(0)), _cache>(array.slice(i)); }
	__BCinline__ const auto row(int i) const {
		return unary_expression<decltype(array.row(0)),_cache>(array.row(i)); }
	__BCinline__ const auto col(int i) const {
		return unary_expression<decltype(array.col(0)),_cache>(array.col(i)); }
	__BCinline__ const auto scalar(int i) const {
		return unary_expression<decltype(array.scalar(0)),_cache>(array.scalar(i)); }
};
}
#endif /* EXPRESSION_UNARY_POINTWISE_CU_ */
