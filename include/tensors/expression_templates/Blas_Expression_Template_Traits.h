/*
 * Blas_Expression_Traits.h
 *
 *  Created on: Oct 10, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_TENSORS_EXPRS_BLAS_EXPRESSION_TRAITS_H_
#define BLACKCATTENSORS_TENSORS_EXPRS_BLAS_EXPRESSION_TRAITS_H_

#include "Expression_Template_Traits.h"
#include "Array.h"
#include "Array_Kernel_Array.h"
#include "Array_Scalar_Constant.h"
#include "Tree_Evaluator.h"
#include "Tree_Evaluator_Optimizer.h"

namespace BC {
namespace tensors {
namespace exprs {
namespace detail {

template<class T>
struct remove_scalar_mul {
	using type = T;
	using scalar_type = BC::traits::conditional_detected_t<
			query_value_type, T, T>*;

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

	static scalar_type get_scalar(
			Binary_Expression<oper::Scalar_Mul, lv, rv> expression) {
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

	static type rm(
			Unary_Expression<oper::transpose<SystemTag>, Array> expression) {
		return expression.array;
	}
};

template<class Array, class SystemTag, class Rv>
struct remove_transpose<
		Binary_Expression<
			oper::Scalar_Mul,
			Unary_Expression<oper::transpose<SystemTag>,
			Array>,
			Rv>>
{
	using type = Array;

	static type rm(
			Binary_Expression<
					oper::Scalar_Mul,
					Unary_Expression<oper::transpose<SystemTag>,
					Array>,
					Rv> expression) {
		return expression.left.array;
	}
};

template<class Array, class SystemTag, class Lv>
struct remove_transpose<
		Binary_Expression<
				oper::Scalar_Mul,
				Lv,
				Unary_Expression<oper::transpose<SystemTag>,
				Array>>>
{
	using type = Array;

	static type rm(
			Binary_Expression<
					oper::Scalar_Mul,
					Lv,
					Unary_Expression<oper::transpose<SystemTag>,
					Array>> expression) {
		return expression.right.array;
	}
};

} //end of ns detail


namespace blas_expression_parser {
template<class SystemTag>
struct Blas_Expression_Parser;
}

template<class T>
struct blas_expression_traits:
		expression_traits<T> {

	using requires_greedy_evaluation = BC::traits::conditional_detected_t<
			 detail::query_requires_greedy_evaluation, T, std::false_type>;

	using remove_scalar_mul_type    = typename detail::remove_scalar_mul<T>::type;
	using remove_transpose_type     = typename detail::remove_transpose<T>::type;
	using remove_blas_features_type = typename detail::remove_transpose<remove_scalar_mul_type>::type;
	using scalar_multiplier_type    = typename detail::remove_scalar_mul<T>::scalar_type;
	using value_type                = typename T::value_type;

	using is_scalar_multiplied = BC::traits::truth_type<
			!std::is_same<remove_scalar_mul_type, T>::value>;

	using is_transposed = BC::traits::truth_type<
			!std::is_same<remove_transpose_type,  T>::value>;

	static remove_transpose_type remove_transpose(T expression) {
		return detail::remove_transpose<T>::rm(expression);
	}

	static remove_scalar_mul_type remove_scalar_mul(T expression) {
		return detail::remove_scalar_mul<T>::rm(expression);
	}

	static remove_blas_features_type remove_blas_modifiers(T expression) {
		return detail::remove_transpose<remove_scalar_mul_type>::rm(
				remove_scalar_mul(expression));
	}

	//If an expression with a scalar,
	//returns the scalar,
	//else returns a nullpointer of the valuetype == to T::value_type
	static auto get_scalar(const T& expression)
		-> decltype(detail::remove_scalar_mul<T>::get_scalar(expression)) {
		return detail::remove_scalar_mul<T>::get_scalar(expression);
	}

	template<int Alpha, int Beta, class Stream>
	static auto parse_expression(Stream stream, T expression) {
		using system_tag = typename T::system_tag;
		return blas_expression_parser::Blas_Expression_Parser<system_tag>::
				template parse_expression<Alpha, Beta>(
					stream, expression.left, expression.right);
	}

	template<class Stream, class Contents>
	static void post_parse_expression_evaluation(Stream stream, Contents contents) {
		using system_tag = typename T::system_tag;
		blas_expression_parser::Blas_Expression_Parser<system_tag>::
				template post_parse_expression_evaluation(stream, contents);
	}
};


}
}
}

#include "blas_expression_parser/Blas_Expression_Parser.h"

#endif /* BLAS_EXPRESSION_TRAITS_H_ */
