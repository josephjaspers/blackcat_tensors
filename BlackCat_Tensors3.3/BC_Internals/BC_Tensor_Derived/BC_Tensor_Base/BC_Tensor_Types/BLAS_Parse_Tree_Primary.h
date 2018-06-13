/*
 * BLAS_Parse_Tree.h
 *
 *  Created on: Jun 7, 2018
 *      Author: joseph
 */

#ifndef BLAS_PARSE_TREE_H_
#define BLAS_PARSE_TREE_H_

#include "Operations/Binary.h"
#include "BC_Utility/MetaTemplateFunctions.h"
#include "BLAS_Parse_Tree_Precedence.h"
namespace BC {

//blas function
namespace oper{
template<class> class dotproduct;
}

template<class T> static constexpr bool is_blas_supported_oper  = MTF::is_one_of<T, oper::add, oper::sub>();
template<class T> static constexpr bool is_blas				= std::is_base_of<BLAS_FUNCTION, T>::value;
template<class lv, class rv, class op>
static constexpr bool is_blas<BC::internal::binary_expression<lv, rv, op>> = is_blas<lv> && is_blas<rv>;


namespace internal {
template<class T> static constexpr bool is_void() { return std::is_same<T, void>::value; }


//default -- this is a core type, no injection
template<class tensor_core, class inject_t = void, class enabler = void>
struct traversal {

	static constexpr int priority = -1;
	static constexpr bool substitution = false;
	static constexpr bool injection = false;
	using type = tensor_core; //no sub _type
};

//binary_expression handler
template<class lv, class rv, class operand, class inject_t>
struct traversal<binary_expression<lv, rv, operand>, inject_t>  {

	///need code for preferencing
	static constexpr int priority 	= PRIORITY<operand>::value;
	static constexpr bool lv_inject = priority <= traversal<lv>::priority;
	static constexpr bool rv_inject = priority <= traversal<rv>::priority;
	using lv_inject_t = std::conditional_t<lv_inject, inject_t, void>;
	using rv_inject_t = std::conditional_t<rv_inject, inject_t, void>;

	using lv_branch = typename traversal<lv, lv_inject_t>::type;
	using rv_branch = typename traversal<rv, rv_inject_t>::type;

	using type = binary_expression<lv_branch, rv_branch, operand>;
};

//meh specilization ---
template<class lv1, class rv1, class lv2, class rv2, class inject_t, class ml>
struct traversal<
	binary_expression<
		binary_expression<lv1, rv1, oper::dotproduct<ml>>,	//lv
		binary_expression<lv2, rv2, oper::dotproduct<ml>>,	//rv
			oper::add>, inject_t>  {

	static constexpr int priority = 0;
	static constexpr bool injection 	= !is_void<inject_t>();
	static constexpr bool substitution 	= !is_void<inject_t>();

	using type =inject_t;
};
//meh specilization ---
template<class lv1, class rv1, class lv2, class rv2, class inject_t, class ml>
struct traversal<
	binary_expression<
		binary_expression<lv1, rv1, oper::dotproduct<ml>>,	//lv
		binary_expression<lv2, rv2, oper::dotproduct<ml>>,	//rv
			oper::sub>, inject_t>  {

	static constexpr int priority = 0;
	static constexpr bool injection 	= !is_void<inject_t>();
	static constexpr bool substitution 	= !is_void<inject_t>();

	using type =inject_t;
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

	using expr 		= binary_expression<lv, rv, oper::dotproduct<ml>>;
	using tensor_t 	= tensor_of_t<expr::DIMS(), _scalar<lv>, _mathlib<lv>>;
	using sub_t 	= Core<tensor_t>;

	using type = std::conditional_t<injection, inject_t, sub_t>;
};

template<class T>
static constexpr bool INJECTION() {
	return ! std::is_same<T, typename traversal<T>::type>::value;
}



}
}


#include "BLAS_Parse_Tree_Assignments.h"


#endif /* BLAS_PARSE_TREE_H_ */
