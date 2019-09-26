/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_INTERNAL_FORWARD_DECLS_H_
#define BLACKCAT_INTERNAL_FORWARD_DECLS_H_

#include <type_traits>

namespace BC {
namespace tensors {
namespace exprs { 

template<class,class,class> struct Binary_Expression;
template<class,class>		struct Unary_Expression;

namespace detail {


template<class T> using query_system_tag = typename T::system_tag;
template<class T> using query_value_type = typename T::value_type;
template<class T> using query_allocation_type = typename T::allocation_tag;

template<class T> using query_copy_assignable    = typename T::copy_assignable;
template<class T> using query_copy_constructible = typename T::copy_constructible;
template<class T> using query_move_assignable    = typename T::move_assignable;
template<class T> using query_move_constructible = typename T::move_constructible;
template<class T> using query_requires_greedy_evaluation = typename T::requires_greedy_evaluation;

template<class T>
struct remove_scalar_mul {
	using type = T;
	using scalar_type = typename BC::traits::conditional_detected<query_value_type, T, T>::type *;

	static T rm(T expression) {
		return expression;
	}

	static scalar_type get_scalar(const T& expression) {
		return nullptr;
	};
};

template<class lv, class rv>
struct remove_scalar_mul<Binary_Expression<oper::Scalar_Mul, lv, rv>> {
	using type = std::conditional_t<lv::tensor_dimension == 0, rv, lv>;
	using scalar_type = std::conditional_t<lv::tensor_dimension == 0, lv ,rv>;

	static type rm(Binary_Expression<oper::Scalar_Mul, lv, rv> expression) {
		return BC::traits::constexpr_ternary<lv::tensor_dimension==0>(
				[&]() { return expression.right; },
				[&]() { return expression.left;  }
		);
	}
	static scalar_type get_scalar(Binary_Expression<oper::Scalar_Mul, lv, rv> expression) {
		return BC::traits::constexpr_ternary<lv::tensor_dimension==0>(
				[&]() { return expression.left; },
				[&]() { return expression.right;  }
		);
	}

};

template<class T>
struct remove_transpose {
	using type = T;
	static T rm(T expression) {
		return expression;
	}
};
template<class Array, class SystemTag>
struct remove_transpose<Unary_Expression<oper::transpose<SystemTag>, Array>> {
	using type = Array;

	static type rm(Unary_Expression<oper::transpose<SystemTag>, Array> expression) {
		return expression.array;
	}
};

template<class Array, class SystemTag, class Rv>
struct remove_transpose<
	Binary_Expression<oper::Scalar_Mul, Unary_Expression<oper::transpose<SystemTag>, Array>, Rv>> {
	using type = Array;

	static type rm(Binary_Expression<oper::Scalar_Mul, Unary_Expression<oper::transpose<SystemTag>, Array>, Rv> expression) {
		return expression.left.array;
	}
};

template<class Array, class SystemTag, class Lv>
struct remove_transpose<
	Binary_Expression<oper::Scalar_Mul, Lv, Unary_Expression<oper::transpose<SystemTag>, Array>>> {
	using type = Array;

	static type rm(Binary_Expression<oper::Scalar_Mul, Lv, Unary_Expression<oper::transpose<SystemTag>, Array>> expression) {
		return expression.right.array;
	}
};

}

#define BC_TAG_DEFINITION(name, using_name, default_value)\
struct name { using using_name = default_value; };\
namespace detail { template<class T> using query_##using_name = typename T::using_name; }

BC_TAG_DEFINITION(BC_Type, is_bc_type, std::true_type);
BC_TAG_DEFINITION(BC_Array, is_tensor_type, std::true_type);
BC_TAG_DEFINITION(BC_Expr, is_expression_type, std::true_type);
BC_TAG_DEFINITION(BC_Temporary, is_temporary_value, std::true_type);
BC_TAG_DEFINITION(BC_Stack_Allocated, is_stack_allocated, std::true_type);
BC_TAG_DEFINITION(BC_Noncontinuous, is_noncontinuous_in_memory, std::true_type);
BC_TAG_DEFINITION(BC_Immutable, is_not_mutable, std::true_type);

#undef BC_TAG_DEFINITION

class BC_View {
	using is_view_type = std::true_type;
	using copy_constructible = std::false_type;
	using move_constructible = std::false_type;
    using copy_assignable    = std::true_type;
	using move_assignable    = std::false_type;
};

namespace detail {
template<class T> using query_is_view_type = typename T::is_view_type;
}

template<class T>
struct expression_traits {

	using system_tag = BC::traits::conditional_detected_t<
			detail::query_system_tag, T, host_tag>;

	using allocation_tag = BC::traits::conditional_detected_t<
			detail::query_allocation_type, T, system_tag>;

	using value_type = BC::traits::conditional_detected_t<
			detail::query_value_type, T, void>;

	using is_move_constructible = BC::traits::conditional_detected_t<
			detail::query_move_constructible, T, std::true_type>;

	using is_copy_constructible = BC::traits::conditional_detected_t<
			detail::query_copy_constructible, T, std::true_type>;

	using is_move_assignable = BC::traits::conditional_detected_t<
			detail::query_move_assignable, T, std::true_type>;

	using is_copy_assignable = BC::traits::conditional_detected_t<
			detail::query_copy_assignable, T, std::true_type>;

	using requires_greedy_evaluation = BC::traits::conditional_detected_t<
			detail::query_requires_greedy_evaluation,T, std::false_type>;

	using is_bc_type = BC::traits::conditional_detected_t<
			detail::query_is_bc_type, T, std::false_type>;

	using is_array = BC::traits::conditional_detected_t<
			detail::query_is_tensor_type, T, std::false_type>;

	using is_expr = BC::traits::conditional_detected_t<
			detail::query_is_expression_type, T, std::false_type>;

	using is_temporary = BC::traits::conditional_detected_t<
			detail::query_is_temporary_value, T, std::false_type>;

	using is_stack_allocated = BC::traits::conditional_detected_t<
			detail::query_is_stack_allocated, T, std::false_type>;

	using is_noncontinuous = BC::traits::conditional_detected_t<
			detail::query_is_noncontinuous_in_memory, T, std::false_type>;

	using is_continuous = BC::traits::truth_type<!is_noncontinuous::value>;

};

template<class T>
struct blas_expression_traits : expression_traits<T> {

	using requires_greedy_evaluation = BC::traits::conditional_detected_t<
			 detail::query_requires_greedy_evaluation, T, std::false_type>;

	using remove_scalar_mul_type	= typename detail::remove_scalar_mul<T>::type;
	using remove_transpose_type		= typename detail::remove_transpose<T>::type;
	using remove_blas_features_type	= typename detail::remove_transpose<remove_scalar_mul_type>::type;
	using scalar_multiplier_type	= typename detail::remove_scalar_mul<T>::scalar_type;
	using value_type				= typename T::value_type;

	using is_scalar_multiplied = BC::traits::truth_type<!std::is_same<remove_scalar_mul_type, T>::value>;
	using is_transposed = BC::traits::truth_type<!std::is_same<remove_transpose_type,  T>::value>;

	static remove_transpose_type remove_transpose(T expression) { return detail::remove_transpose<T>::rm(expression); }
	static remove_scalar_mul_type remove_scalar_mul(T expression) { return detail::remove_scalar_mul<T>::rm(expression); }
	static remove_blas_features_type remove_blas_modifiers(T expression) {
		return detail::remove_transpose<remove_scalar_mul_type>::rm(remove_scalar_mul(expression));
	}

	 //If an expression with a scalar, returns the scalar, else returns a nullpointer of the valuetype == to T::value_type
	static auto get_scalar(const T& expression)
		-> decltype(detail::remove_scalar_mul<T>::get_scalar(expression)) {
		return detail::remove_scalar_mul<T>::get_scalar(expression);
	}
};

} //ns BC
} //ns exprs
} //ns tensors

#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
