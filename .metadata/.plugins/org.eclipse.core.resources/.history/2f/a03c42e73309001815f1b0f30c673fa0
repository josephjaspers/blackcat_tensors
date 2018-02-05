/*
 * Transpose_Wrapper.h
 *
 *  Created on: Dec 19, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#define EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_
#include <cmath>
#include "Expression_Base.h"
#include "../BC_MetaTemplateFunctions/Adhoc.h"
#include "../BlackCat_Internal_Definitions.h"

namespace BC {
template<class T, class parent>
struct unary_expression_transpose : expression<T, unary_expression_transpose<T, parent>>
{
	// Rows and Cols are the CURRENT rows and columns of the
	// Utilizing CUDA's experimental-flag for calling constexpr expression (seems to be working fine)

	using this_type = unary_expression_transpose<T,parent>;
	using functor_type = typename MTF::determine_functor<T>::type;


	const parent& root;

	unary_expression_transpose(const parent& p) : root(p) {}

	__attribute__((always_inline)) __BC_gcpu__  auto& operator [](int index) {
		return root.data()[(int)floor(index / root.cols()) + (index % root.cols()) * root.rows()];
	}
	__attribute__((always_inline))  __BC_gcpu__ const auto& operator[](int index) const {
		return root.data()[(int) floor(index / root.cols()) + (index % root.cols()) * root.rows()];
	}
};
template<class,class> class Vector;

template<class T, class U, class ml>
struct unary_expression_transpose<T, Vector<U, ml>> : expression<T, unary_expression_transpose<T, Vector<U, ml>>>
{
	// Rows and Cols are the CURRENT rows and columns of the
	// Utilizing CUDA's experimental-flag for calling constexpr expression (seems to be working fine)

	using this_type = unary_expression_transpose<T, Vector<U, ml>>;
	using functor_type = typename MTF::determine_functor<T>::type;
	using parent = Vector<U, ml>;

	const Vector<U, ml>& root;

	unary_expression_transpose(const parent& p) : root(p) {}

	__attribute__((always_inline)) __BC_gcpu__  auto& operator [](int index) {
		return root.data()[index];
	}
	__attribute__((always_inline))  __BC_gcpu__ const auto& operator[](int index) const {
		return root.data()[index];
	}
};
}
#endif /* EXPRESSION_UNARY_MATRIXTRANSPOSITION_H_ */
