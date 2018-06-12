/*
 * BLAS_Parse_Tree_Assignments.h
 *
 *  Created on: Jun 10, 2018
 *      Author: joseph
 */

#ifndef BLAS_PARSE_TREE_ASSIGNMENTS_H_
#define BLAS_PARSE_TREE_ASSIGNMENTS_H_

#include "BLAS_Parse_Tree_Primary.h"
#include "BLAS_Injection_Wrapper.h"

namespace BC {
namespace internal {

/*
 * Part of the operation_tree, these specializations contains the
 * 	constexpr int method ALPHA/BETA modifiers -- which assist in informing what scalars to inject.
 *
 */

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
	static constexpr int ALPHA_MODIFIER() { return 1; }
	static constexpr int BETA_MODIFIER() { return 0; }

	using type = binary_expression<lv_branch, rv_branch, oper::assign>;
};

template<class lv, class rv, class inject_t>
struct traversal<binary_expression<lv, rv, oper::add_assign>, inject_t>  {

	using traversal_left = traversal<lv, lv>;
	using traversal_right = traversal<rv, lv>;
	using lv_branch = typename traversal_left::type;
	using rv_branch = typename traversal_right::type;

	static constexpr int priority = 0;


	static constexpr bool substitution = traversal_left::substitution || traversal_right::substitution;
	static constexpr bool injection =  traversal_left::injection || traversal_right::injection;
	static constexpr int ALPHA_MODIFIER() { return 1; }
	static constexpr int BETA_MODIFIER() { return 1; }

	using expr_self = binary_expression<lv, rv, oper::assign>;

	using type = binary_expression<lv_branch, rv_branch, oper::add_assign>;
};

template<class lv, class rv, class inject_t>
struct traversal<binary_expression<lv, rv, oper::sub_assign>, inject_t>  {

	using traversal_left = traversal<lv, lv>;
	using traversal_right = traversal<rv, lv>;
	using lv_branch = typename traversal_left::type;
	using rv_branch = typename traversal_right::type;

	static constexpr int priority = 0;


	static constexpr bool substitution = traversal_left::substitution || traversal_right::substitution;
	static constexpr bool injection =  traversal_left::injection || traversal_right::injection;
	static constexpr int ALPHA_MODIFIER() { return -1; }
	static constexpr int BETA_MODIFIER() { return 1; }

	using type = binary_expression<lv_branch, rv_branch, oper::sub_assign>;
};

}
}



#endif /* BLAS_PARSE_TREE_ASSIGNMENTS_H_ */
