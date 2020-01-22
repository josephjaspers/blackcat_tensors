/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_GEMV_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_GEMV_H_

#include "expression_template_base.h"
#include "tree_evaluator.h"
#include "blas_expression_template_traits.h"


namespace bc {
namespace tensors {
namespace exprs { 


template<class lv, class rv, class System_Tag>
struct Bin_Op<oper::gemv<System_Tag>, lv, rv>:
			Expression_Base<Bin_Op<oper::gemv<System_Tag>, lv, rv>>,
			oper::gemv<System_Tag> {

	static_assert(std::is_same<
			typename lv::value_type,
			typename rv::value_type>::value,
			"GEMV arguments must have the same value_type");

	static_assert(lv::tensor_dim == 2 && rv::tensor_dim == 1,
			"Lv must be a Matrix and Rv must be a Vector");

	using value_type = typename lv::value_type;
	using system_tag = System_Tag;

	static constexpr int tensor_dim = 1;
	static constexpr int tensor_iterator_dim = 1;

	lv left;
	rv right;

	Bin_Op(lv left, rv right):
			left(left),
			right(right)
	{
		BC_ASSERT(left.cols() == right.rows(),
				"gemv requires left.cols() == right.rows()");
	}

	static oper::gemv<System_Tag> get_operation() {
		return oper::gemv<System_Tag>();
	}

	BCINLINE bc::size_t size() const { return left.rows(); }
	BCINLINE bc::size_t dim(int i) const { return i == 0 ? left.rows() : 1; }
	BCINLINE bc::size_t rows() const { return dim(0); }
	BCINLINE bc::size_t cols() const { return dim(1); }

	template<class core, int Alpha, int Beta, class Stream>
	void eval(Output_Data<core, Alpha, Beta> output, Stream stream) const {
		static_assert(core::tensor_dim==1, "Gemv out must be a vector");

		using self_t = Bin_Op<oper::gemv<System_Tag>, lv, rv>;
		using traits = blas_expression_traits<self_t>;

		//evaluate the left and right branches (computes only if necessary)
		auto contents = traits::template parse_expression<Alpha, Beta>(stream, *this);
		auto A = contents.left;
		auto X = contents.right;
		auto alpha = contents.alpha;
		auto beta  = contents.beta;
		bool transA = contents.lv_is_transposed;

		auto& out = output.data();

		//gemv uses the [m,n] to refer to dim ignoring op(A)
		//http://www.netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gafc92361b74c6d41c7e5afa0aa5d13ec9.html#gafc92361b74c6d41c7e5afa0aa5d13ec9
		bc::blas::BLAS<system_tag>::gemv(
				stream, transA, A.rows(), A.cols(),
				alpha.data(), A.data(), A.leading_dim(1),
				X.data(), X.leading_dim(0)/*inc_X*/,
				beta.data(),
				out.data()/*Y*/, out.leading_dim(0)/*incy*/);

		traits::post_parse_expression_evaluation(stream, contents);
	}
};


} //ns BC
} //ns exprs
} //ns tensors


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
