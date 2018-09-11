/*
 * Expression_Tree_Functions.h
 *
 *  Created on: Jun 13, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_TREE_FUNCTIONS_H_
#define EXPRESSION_TREE_FUNCTIONS_H_

#include "Operations/Unary.h"
#include "Operations/Binary.h"
#include "Operations/BLAS.h"
#include "Utility/MetaTemplateFunctions.h"

namespace BC{
namespace internal {
namespace tree {



template<class> struct scalar_modifer {
	enum mod {
		alpha = 0,
		beta = 0,
	};
};
template<> struct scalar_modifer<oper::add> {
	enum mod {
		alpha = 1,
		beta = 1,
	};
};
template<> struct scalar_modifer<oper::sub> {
	enum mod {
		alpha = -1,
		beta = 1
	};
};
template<> struct scalar_modifer<oper::add_assign> {
	enum mod {
		alpha = 1,
		beta = 1,
	};
};
template<> struct scalar_modifer<oper::sub_assign> {
	enum mod {
		alpha = -1,
		beta = 1,
	};
};
template<> struct scalar_modifer<oper::assign> {
	enum mod {
		alpha = 1,
		beta = 0,
	};
};

template<class T> static constexpr bool is_blas_func() {
	return std::is_base_of<BC::BLAS_FUNCTION, T>::value;
}

template<class T>
static constexpr bool is_linear_op() {
	return MTF::seq_contains<T, oper::add, oper::sub>;
}

template<class T>
static constexpr bool is_nonlinear_op() {
	return  !MTF::seq_contains<T, oper::add, oper::sub> && !is_blas_func<T>();
}
template<class T>
static constexpr bool is_linear_assignment_op() {
	return MTF::seq_contains<T, oper::add_assign, oper::sub_assign>;
}
template<class T>
static constexpr bool is_assignment_op() {
	return MTF::seq_contains<T, oper::assign, oper::add_assign,oper::sub_assign, oper::mul_assign,oper::div_assign>;
}

template<class T>
static constexpr bool is_standard_assignment_op() {
	return MTF::seq_contains<oper::assign>;
}

template<class T>
static constexpr int alpha_of() {
	return scalar_modifer<std::decay_t<T>>::mod::alpha;
}
template<class T>
static constexpr int beta_of() {
	return scalar_modifer<std::decay_t<T>>::mod::beta;
}


//trivial_blas_evaluation -- detects if the tree is entirely +/- operations with blas functions, --> y = a * b + c * d - e * f  --> true, y = a + b * c --> false
template<class op, class core, int a, int b>//only apply update if right hand side branch
auto update_injection(injector<core,a,b> tensor) {
	static constexpr int alpha = a != 0 ? a * alpha_of<op>() : 1;
	static constexpr int beta = b != 0 ? b * beta_of<op>() : 1;
	return injector<core, alpha, beta>(tensor.data());
}


}
}
}




#endif /* EXPRESSION_TREE_FUNCTIONS_H_ */
