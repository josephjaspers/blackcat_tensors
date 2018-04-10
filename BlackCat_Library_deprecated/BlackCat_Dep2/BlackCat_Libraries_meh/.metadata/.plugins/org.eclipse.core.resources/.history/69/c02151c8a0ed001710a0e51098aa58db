/*
 * BC_MetaTemplate_EssentialMethods.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef ADHOC_H_
#define ADHOC_H_


#include "../BlackCat_Internal_GlobalUnifier.h"
//Expression Forward decs
namespace BC {
template<class, class >
class expression;

template<class, class, class, class >
class binary_expression;

//Array Forward decs
template<class T, class p>
struct unary_expression_transpose_mat;


namespace MTF {

	/*
	 * Contains meta-template functions that are designed specifically for this library
	 * All other meta-template functions are generalizable
	 */

	template<class >
	struct isArrayType {
		static constexpr bool conditional = false;
	};

	template<class T, class p>
	struct isArrayType<unary_expression_transpose_mat<T, p>> {
		static constexpr bool conditional = true;
		using type = unary_expression_transpose_mat<T, p>;
	};

	template<class > struct isExpressionType {
		static constexpr bool conditional = false;
	};

	template<class T, class deriv> struct isExpressionType<expression<T, deriv>> {
		static constexpr bool conditional = true;
	};

	template<class T, class O, class L, class R> struct isExpressionType<binary_expression<T, O, L, R>> {
		static constexpr bool conditional = true;
	};

	template<class,class>
	struct unaryExpression_negation;

	template<class T, class O> struct isExpressionType<unaryExpression_negation<T, O>> {
		static constexpr bool conditional = true;
	};

/*
 * ADD ARRAY_TYPE CLASSES HERE ---  IF A CLASS IS
 */

}
}

#endif /* ADHOC_H_ */
