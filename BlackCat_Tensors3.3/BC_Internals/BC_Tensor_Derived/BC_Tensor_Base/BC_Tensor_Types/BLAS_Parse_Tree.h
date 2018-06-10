/*
 * BLAS_Parse_Tree.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef BLAS_PARSE_TREE_H_
#define BLAS_PARSE_TREE_H_

#include "Operations/Binary.h"
namespace BC {



namespace oper {

//priority -1 == non-rotateable (injection not legal)
//priority 0 == low
//priority 1 == high
/*
 * a low priority may be after a high priority (ensure priority order)
 * but a low priority preceding a high-priority operation will result in non-injectable segment
 *
 * This module assigns arithmetic values to operands to handle apropriate precedence in injections
 */

enum precendece_level {

	non_rotateable = -1,
	lo_commutative = 0,
	hi_commutative = 1

};

template<class> struct dotproduct;
template<class> struct PRIORITY { static constexpr int value = -1;};
template<> struct PRIORITY<add> { static constexpr int value = 0; };
template<> struct PRIORITY<sub> { static constexpr int value = 0; };
template<> struct PRIORITY<mul> { static constexpr int value = 1; };
template<> struct PRIORITY<div> { static constexpr int value = 1; };
template<> struct PRIORITY<add_assign> { static constexpr int value = 0; };
template<> struct PRIORITY<sub_assign> { static constexpr int value = 0; }; //this will eventually be allowed but for now NO
template<> struct PRIORITY<assign> { static constexpr int value = 1; };

}

enum branch {
	none = -1,
	center = 0,
	left = 1,
	right = 2,
	left_and_right = 3
};

namespace internal {
template<class T> static constexpr bool is_void() { return std::is_same<T, void>::value; }

//default -- this is a core type, no injection
template<class tensor_core, class inject_t = void>
struct traversal {

	static constexpr int priority = -1;
	static constexpr bool substitution = false;
	static constexpr bool injection = false;
	static constexpr branch path = none;
	using type = tensor_core; //no sub _type
};

//binary_expression handler
template<class lv, class rv, class operand, class inject_t>
struct traversal<binary_expression<lv, rv, operand>, inject_t>  {

	///need code for preferencing
	static constexpr int priority 	= oper::PRIORITY<operand>::value;
	static constexpr bool lv_inject = priority <= traversal<lv>::priority;
	static constexpr bool rv_inject = priority <= traversal<rv>::priority;
	using lv_inject_t = std::conditional_t<lv_inject, inject_t, void>;
	using rv_inject_t = std::conditional_t<rv_inject, inject_t, void>;

	using lv_branch = typename traversal<lv, lv_inject_t>::type;
	using rv_branch = typename traversal<rv, rv_inject_t>::type;

	using type = binary_expression<lv_branch, rv_branch, operand>;
};

//unary_expression handler
template<class v, class operand, class inject_t>
struct traversal<unary_expression<v, operand>, inject_t>  {
	static constexpr int priority = 1;
	static constexpr bool injection = priority <= traversal<v, inject_t>::priority && (!is_void<inject_t>());

	using asserted_inject_t = std::conditional_t<injection, inject_t, void>;
	static constexpr bool substitution = traversal<v, asserted_inject_t>::substitution;

	using type = unary_expression<typename traversal<v, asserted_inject_t>::type, operand>;
};

//dotproduct_handler (will need to recreate this overload for each BLAS function)
template<class lv, class rv, class ml, class inject_t>
struct traversal<binary_expression<lv, rv, oper::dotproduct<ml>>, inject_t>  {

	static constexpr int priority = 0;
	static constexpr bool substitution = is_void<inject_t>();
	static constexpr bool injection = !is_void<inject_t>();

	using expr = binary_expression<lv, rv, oper::dotproduct<ml>>;
	using tensor_t = tensor_of_t<expr::DIMS(), _scalar<lv>, _mathlib<lv>>;
	using sub_t 	= Core<tensor_t>;

	using type = std::conditional_t<injection, inject_t, sub_t>;
};

//overload for the correct
template<class lv, class rv, class inject_t>
struct traversal<binary_expression<lv, rv, oper::assign>, inject_t>  {

	using traversal_left = traversal<lv, lv>;
	using traversal_right = traversal<rv, lv>;
	using lv_branch = typename traversal_left::type;
	using rv_branch = typename traversal_right::type;

	static constexpr int priority = 1;


	static constexpr bool substitution = traversal_left::substitution || traversal_right::substitution;
	static constexpr bool injection =  traversal_left::injection || traversal_right::injection;
	using expr_self = binary_expression<lv, rv, oper::assign>;

	using type = binary_expression<lv_branch, rv_branch, oper::assign>;
};

template<class T>
static constexpr bool SUBSTITUTION() {
	return ! std::is_same<T, typename traversal<T>::type>::value;
}

template<class T>
static constexpr bool INJECTION() {
	return ! std::is_same<T, typename traversal<T>::type>::value;
}



}
}



#endif /* BLAS_PARSE_TREE_H_ */
