/*
 * Expression.h
 *
 *  Created on: Feb 15, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_H_
#define EXPRESSION_H_

namespace BC {

template<class derived>
struct expression {};

template<class derived, class lv, class rv>
struct binary_expression : expression<derived> {

	const lv& left;
	const rv& right;

	__attribute__((always_inline)) inline
	binary_expression(const lv& l, const rv& r) : left(l), right(r) {}
};
template<class derived, class lv, class rv>
struct unary_expression : expression<derived> {

	const rv& right;

	const int size() { return right.size(); }

	__attribute__((always_inline)) inline
			unary_expression(const rv& r) : right(r) {}
};

template<class operation, class lv, class rv>
struct bPointwise : binary_expression<bPointwise<operation,lv,rv>, lv, rv> {

	operation oper;
	__attribute__((always_inline)) inline
	bPointwise(const lv& l, const rv& r) : binary_expression<bPointwise<operation,lv,rv>, lv, rv> (l, r) {}

	__attribute__((always_inline)) inline
	auto operator[] (int index) {
		return oper(this->left[index], this->right[index]);
	}
};

template<class o, class l, class r>
using bp_expr = bPointwise<o, l, r>;


}



#endif /* EXPRESSION_H_ */
