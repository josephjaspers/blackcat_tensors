/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GEMM_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GEMM_H_

#include "Expression_Template_Base.h"
#include "Tree_Evaluator.h"
#include "Blas_Expression_Template_Traits.h"

namespace BC {
namespace tensors {
namespace exprs { 


template<class lv, class rv, class SystemTag>
struct Binary_Expression<oper::gemm<SystemTag>, lv, rv>:
		Expression_Base<Binary_Expression<oper::gemm<SystemTag>, lv, rv>>,
		oper::gemm<SystemTag> {

	static_assert(std::is_same<
				typename lv::value_type,
				typename rv::value_type>::value,
			"GEMM arguments must have the same value_type");

	static_assert(lv::tensor_dimension==2 && rv::tensor_dimension==2,
			"Error: GEMM Expression initialized with non matrix tensor");

	using value_type = typename lv::value_type;
	using system_tag = SystemTag;

	static constexpr int tensor_dimension = rv::tensor_dimension;
	static constexpr int tensor_iterator_dimension  = 1;

	lv left;
	rv right;

	BCHOT Binary_Expression(lv left, rv right):
			left(left),
			right(right) {}

	BCINLINE BC::size_t  size() const { return left.rows() * right.cols(); }
	BCINLINE BC::size_t  dimension(int i) const {
		return i == 0 ? left.rows() : i == 1 ? right.cols() : 1;
	}

	BCINLINE BC::size_t rows() const { return dimension(0); }
	BCINLINE BC::size_t cols() const { return dimension(1); }

	template<class Core, int Alpha, int Beta, class Stream>
	void eval(Output_Data<Core, Alpha, Beta> output, Stream stream) const {

		using traits = blas_expression_traits<
				Binary_Expression<oper::gemm<SystemTag>, lv, rv>>;

		static_assert(Core::tensor_dimension == 2,
				"Gemm out must be a matrix");
		BC_ASSERT(left.cols() == right.rows(),
				"gemm requires left.cols() == right.rows()");

	//get the data of the out --> Output_Data simply stores the alpha/beta scalar modifiers
	auto& out = output.data();

	auto contents = traits::template parse_expression<Alpha, Beta>(stream, *this);
	auto A = contents.left;
	auto B = contents.right;
	auto alpha = contents.alpha;
	auto beta  = contents.beta;
	auto transA = contents.lv_is_transposed;
	auto transB = contents.rv_is_transposed;

		//call matrix_mul
	BC::blas::BLAS<system_tag>::gemm(
				stream, transA, transB,  left.rows(), right.cols(), left.cols(),
				alpha.memptr(), A.memptr(), A.leading_dimension(1),
				B.memptr(), B.leading_dimension(1),
				beta.memptr(), out.memptr(), out.leading_dimension(1));

	traits::template post_parse_expression_evaluation(stream, contents);
	}
};


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
