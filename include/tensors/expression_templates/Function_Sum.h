/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_EXPRESSION_TEMPLATES_FUNCTION_SUM_H_
#define BC_EXPRESSION_TEMPLATES_FUNCTION_SUM_H_

#include "Expression_Template_Base.h"
#include "Tree_Evaluator.h"
#include "blas_tools/Blas_tools.h"

#include "functions/reductions/Reductions.h"


namespace BC {
namespace tensors {
namespace exprs { 

template<class SystemTag>
struct Sum {};

template<class ArrayType, class SystemTag>
struct Unary_Expression<Sum<SystemTag>, ArrayType>
: Expression_Base<Unary_Expression<Sum<SystemTag>, ArrayType>>, Shape<0>, Sum<SystemTag> {

    using value_type = typename ArrayType::value_type;
    using system_tag = SystemTag;
    using requires_greedy_evaluation = std::true_type;


    static constexpr int tensor_dimension  = 0;
    static constexpr int tensor_iterator_dimension = 0;

    ArrayType array;

    using Shape<0>::inner_shape;
    using Shape<0>::block_shape;

    Unary_Expression(ArrayType array_, Sum<SystemTag> sum=Sum<SystemTag>()):
    	array(array_) {}

    template<class core, int alpha_mod, int beta_mod, class Stream>
    void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {
    	static_assert(core::tensor_dimension==0, "Injection must be a scalar type");

    	BC::tensors::exprs::functions::Reduce<system_tag>::sum(
    			stream,
    			injection_values.data(),
    			array);
    }
};


} //ns BC
} //ns exprs
} //ns tensors



#endif /* FUNCTION_DOT_H_ */
