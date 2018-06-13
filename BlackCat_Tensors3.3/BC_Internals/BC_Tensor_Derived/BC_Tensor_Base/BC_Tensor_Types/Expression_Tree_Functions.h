/*
 * Expression_Tree_Functions.h
 *
 *  Created on: Jun 13, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_TREE_FUNCTIONS_H_
#define EXPRESSION_TREE_FUNCTIONS_H_

namespace BC{
namespace oper {
struct add;
struct sub;
struct mul;
struct div;
struct assign;
struct add_assign;
struct mul_assign;
struct div_assign;
struct sub_assign;
template<class ml> struct dotproduct;
}

namespace internal {
namespace tree {



template<class> struct PRECEDENCE {
	enum traits {
		value = -1,
		alpha_modifier = 0,
		beta_modifier = 0,
		injectable_assignment = false,
		blas_rotation = false,
		injector = false
	};
};
template<> struct PRECEDENCE<oper::add> {
	enum traits {
		value = 0,
		alpha_modifier = 1,
		beta_modifier = 1,
		injectable_assignment = false,
		blas_rotation = true,
		injector = false
	};
};
template<> struct PRECEDENCE<oper::sub> {
	enum traits {
		value = 0,
		alpha_modifier = -1,
		beta_modifier = 1,
		injectable_assignment = false,
		blas_rotation = true,
		injector = false
	};
};
template<> struct PRECEDENCE<oper::mul> {
	enum traits {
		value = 1,
		alpha_modifier = 0,
		beta_modifier = 0,
		injectable_assignment = false,
		blas_rotation = false,
		injector = false
	};
};
template<> struct PRECEDENCE<oper::div> {
	enum traits {
		value = 1,
		alpha_modifier = 0,
		beta_modifier = 0,
		injectable_assignment = false,
		blas_rotation = false,
		injector = false
	};
};
template<> struct PRECEDENCE<oper::add_assign> {
	enum traits {
		value = 0,
		alpha_modifier = 1,
		beta_modifier = 1,
		injectable_assignment = true,
		blas_rotation = false,
		injector = true
	};
};
template<> struct PRECEDENCE<oper::sub_assign> {
	enum traits {
		value = 0,
		alpha_modifier = -1,
		beta_modifier = 1,
		injectable_assignment = true,
		blas_rotation = false,
		injector = true
	};
};
template<> struct PRECEDENCE<oper::assign> {
	enum traits {
		value = 1,
		alpha_modifier = 1,
		beta_modifier = 0,
		injectable_assignment = true,
		blas_rotation = false,
		injector = true
	};
};

template<> struct PRECEDENCE<oper::mul_assign> {
	enum traits {
		value = 1,
		alpha_modifier = 0,
		beta_modifier = 0,
		injectable_assignment = false,
		blas_rotation = false,
		injector = false
	};
};
template<> struct PRECEDENCE<oper::div_assign> {
	enum traits {
		value = 1,
		alpha_modifier = 0,
		beta_modifier = 0,
		injectable_assignment = false,
		blas_rotation = false,
		injector = false
	};
};

template<class T>
static constexpr int alpha_of() {
	return PRECEDENCE<std::decay_t<T>>::traits::alpha_modifier;
}
template<class T>
static constexpr int beta_of() {
	return PRECEDENCE<std::decay_t<T>>::traits::beta_modifier;
}
template<class T>
static constexpr int precedence() {
	return PRECEDENCE<std::decay_t<T>>::traits::value;
}

template<class T>
static constexpr int valid_double_inject() {
	return PRECEDENCE<std::decay_t<T>>::traits::blas_rotation;
}

template<class T>
static constexpr bool injectable_assignment() {
	return PRECEDENCE<std::decay_t<T>>::traits::injectable_assignment;
}

template<class T> struct BLAS_FUNCTION_TYPE { static constexpr bool conditional = false; };
template<class T> struct BLAS_FUNCTION_TYPE<BC::oper::dotproduct<T>> { static constexpr bool conditional = true; };
template<class T> struct BLAS_FUNCTION_TYPE<BC::oper::transpose<T>> { static constexpr bool conditional = true; };

template<class T> static constexpr bool is_blas_func() {
	return BLAS_FUNCTION_TYPE<T>::conditional;
}
}
}
}




#endif /* EXPRESSION_TREE_FUNCTIONS_H_ */
