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
#include "Tree_Lazy_Evaluator.h"
#include "blas_tools/Blas_tools.h"

namespace BC {
namespace exprs {


template<class lv, class rv, class System_Tag>
struct Binary_Expression<lv, rv, oper::gemm<System_Tag>>
: Expression_Base<Binary_Expression<lv, rv, oper::gemm<System_Tag>>>,
  oper::gemm<System_Tag> {

	static_assert(std::is_same<typename lv::value_type, typename rv::value_type>::value,
    		"MATRIX MULTIPLICATION ONLY AVAILABLE TO SAME TYPE TENSORS (FLOAT/DOUBLE)");
	static_assert(lv::tensor_dimension==2 && rv::tensor_dimension==2,
    		"GEMM Expression initialized with non matrix tensor");


    using value_type	= typename lv::value_type;
    using system_tag	= System_Tag;
    using blas_impl		= BC::blas::implementation<system_tag>;
    using blas_util	    = BC::exprs::blas_tools::implementation<system_tag>;

    static constexpr int tensor_dimension 	   = rv::tensor_dimension;
    static constexpr int tensor_iterator_dimension  = 1;

    lv left;
    rv right;

    BCHOT Binary_Expression(lv left, rv right)
     : left(left), right(right) {}

    BCINLINE BC::size_t  size() const { return left.rows() * right.cols(); }
    BCINLINE BC::size_t  rows() const { return left.rows();  }
    BCINLINE BC::size_t  cols() const { return right.cols(); }
    BCINLINE BC::size_t  dimension(int i) const {
		return i == 0 ? left.rows() : i == 1 ? right.cols() : 1;
    }
    BCINLINE BC::size_t  block_dimension(int i) const {
			return i == 0 ? left.rows() : i == 1 ? size() : 1;
    }

    template<class core, int alpha_mod, int beta_mod, class Stream>
    void eval(tree::injector<core, alpha_mod, beta_mod> injection_values, Stream& alloc) const {

        //get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
        auto& injection = injection_values.data();

        auto contents = blas_util::template parse_expression<alpha_mod, beta_mod>(alloc, left, right);
        auto A = contents.left;
        auto B = contents.right;
        auto alpha = contents.alpha;
        auto beta  = contents.beta;
        auto transA = contents.lv_is_transposed;
        auto transB = contents.rv_is_transposed;

		//call matrix_mul
        blas_impl::gemm(alloc, transA, transB,  left.rows(), right.cols(), left.cols(),
					alpha, A, A.leading_dimension(0),
					B, B.leading_dimension(0),
					beta, injection, injection.leading_dimension(0));

        blas_util::post_parse_expression_evaluation(alloc, contents);


    }
};


}
}


#endif /* EXPRESSION_BINARY_DOTPRODUCT_CU_ */
