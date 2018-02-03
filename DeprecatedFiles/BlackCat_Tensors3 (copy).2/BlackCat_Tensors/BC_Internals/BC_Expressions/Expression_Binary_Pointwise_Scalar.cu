
/*
 * BC_Expression_Binary_Pointwise_ScalarL.h
 *
 *  Created on: Dec 2, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BINARY_POINTWISE_SCALAR_H_
#define EXPRESSION_BINARY_POINTWISE_SCALAR_H_

#include "Expression_Base.h"
#include "../BlackCat_Internal_Definitions.h"

template<class,class,class,class> class binary_expression;

namespace BC {


template<class T, class operation, class lv, class rv>
class binary_expression_scalar_L : expression<T,binary_expression_scalar_L<T, operation, lv, rv>> {
public:

	using this_type = binary_expression_scalar_L<T, operation, lv, rv>;

	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline))  __BC_gcpu__ binary_expression_scalar_L(lv l, rv r) : left(l), right(r) {}
	inline __attribute__((always_inline))  __BC_gcpu__ auto operator [](int index) const { return oper(left[0], right[index]); }
};

template<class T, class operation, class lv, class rv>
class binary_expression_scalar_R : expression<T, binary_expression_scalar_R<T, operation, lv, rv>> {
public:

	using this_type = binary_expression_scalar_R<T, operation, lv, rv>;

	operation oper;

	lv left;
	rv right;

	inline __attribute__((always_inline))  __BC_gcpu__ binary_expression_scalar_R(lv l, rv r) : left(l), right(r) {}
	inline __attribute__((always_inline))  __BC_gcpu__ auto operator [](int index) const { return oper(left[index], right[0]);}
};

template<class T, class operation, class lv, class rv>
class binary_expression_scalar_LR : expression<T, binary_expression_scalar_LR<T, operation, lv, rv>> {
public:
			T* data;
			operation oper;


			template<class ml>
			inline __attribute__((always_inline))  __BC_gcpu__ binary_expression_scalar_LR(lv l, rv r, ml lib) {
				ml::initialize(data, 1);
				ml::copy(data, binary_expression<T, operation, lv, rv>(l, r), 1);
				oper(l, r);

			}
			inline __attribute__((always_inline))  __BC_gcpu__ auto& operator [](int index) const { return data[0];}
};
}

#endif /* EXPRESSION_BINARY_POINTWISE_SCALAR_H_ */
