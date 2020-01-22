/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GEMM_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GEMM_H_

#include "expression_template_base.h"
#include "tree_evaluator.h"
#include "blas_expression_template_traits.h"

namespace bc {
namespace tensors {
namespace exprs { 


template<class lv, class rv, class SystemTag>
struct Bin_Op<oper::gemm<SystemTag>, lv, rv>:
		Expression_Base<Bin_Op<oper::gemm<SystemTag>, lv, rv>>,
		oper::gemm<SystemTag> {

	static_assert(std::is_same<
				typename lv::value_type,
				typename rv::value_type>::value,
			"GEMM arguments must have the same value_type");

	static_assert(lv::tensor_dim==2 && rv::tensor_dim==2,
			"Error: GEMM Expression initialized with non matrix tensor");

	using value_type = typename lv::value_type;
	using system_tag = SystemTag;

	static constexpr int tensor_dim = rv::tensor_dim;
	static constexpr int tensor_iterator_dim  = 1;

	lv left;
	rv right;

	BCHOT Bin_Op(lv left, rv right):
			left(left),
			right(right)
	{
		BC_ASSERT(left.cols() == right.rows(),
				"gemm requires left.cols() == right.rows()");
	}

	static oper::gemm<SystemTag> get_operation() {
		return oper::gemm<SystemTag>();
	}

	BCINLINE bc::size_t  size() const { return left.rows() * right.cols(); }
	BCINLINE bc::size_t  dim(int i) const {
		return i == 0 ? left.rows() : i == 1 ? right.cols() : 1;
	}

	BCINLINE bc::size_t rows() const { return dim(0); }
	BCINLINE bc::size_t cols() const { return dim(1); }

	template<class Core, int Alpha, int Beta, class Stream>
	void eval(Output_Data<Core, Alpha, Beta> output, Stream stream) const
	{
		auto& out = output.data();

		static_assert(Core::tensor_dim == 2,
				"Gemm out must be a matrix");
		BC_ASSERT(out.rows() == left.rows(),
				"Output dim (rows) mismatch for GEMM");
		BC_ASSERT(out.cols() == right.cols(),
				"Output dim (cols) mismatch for GEMM");

		using self_t = Bin_Op<oper::gemm<SystemTag>, lv, rv>;
		using traits = blas_expression_traits<self_t>;

		auto contents = traits::template parse_expression<Alpha, Beta>(stream, *this);
		auto A = contents.left;
		auto B = contents.right;
		auto alpha = contents.alpha;
		auto beta  = contents.beta;
		auto transA = contents.lv_is_transposed;
		auto transB = contents.rv_is_transposed;

		bc::blas::BLAS<system_tag>::gemm(
					stream, transA, transB, out.rows(), out.cols(), left.cols(),
					alpha.data(), A.data(), A.leading_dim(1),
					B.data(), B.leading_dim(1),
					beta.data(), out.data(), out.leading_dim(1));

		traits::template post_parse_expression_evaluation(stream, contents);
	}
};


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
