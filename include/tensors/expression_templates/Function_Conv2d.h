/*
 * Function_Conv2d.h
 *
 *  Created on: Aug 21, 2019
 *      Author: joseph
 */

#ifndef FUNCTION_CONV2D_H_
#define FUNCTION_CONV2D_H_

#include "Expression_Template_Base.h"
#include "Tree_Evaluator.h"
#include "blas_tools/Blas_tools.h"

namespace BC {
namespace tensors {
namespace exprs {

//Not actually a BLAS function but we want the optimizer to treat it as if it was one
template<class SystemTag>
struct multichannel_conv2d : BC::oper::BLAS_Function {};

template<class lv, class rv, class System_Tag>
struct Binary_Expression<multichannel_conv2d<System_Tag>, lv, rv>
: Expression_Base<Binary_Expression<multichannel_conv2d<System_Tag>, lv, rv>>, multichannel_conv2d<System_Tag> {

    static_assert((lv::tensor_dimension == 3 || lv::tensor_dimension==4) && rv::tensor_dimension==3,
    		"DOT DIMENSION MISMATCH, INTERNAL BUG, REPORT PLEASE");

    using value_type = typename lv::value_type;
    using system_tag = System_Tag;
    using blas_impl  = BC::blas::implementation<system_tag>;
    using blas_util	 = BC::tensors::exprs::blas_tools::implementation<system_tag>;

    static constexpr bool lv_scalar = blas_expression_traits<lv>::is_scalar_multiplied;
    static constexpr bool rv_scalar = blas_expression_traits<rv>::is_scalar_multiplied;

    static constexpr int tensor_dimension  = 3;
    static constexpr int tensor_iterator_dimension = 3;

    lv left;
    rv right;

    BC::size_t stride = 1;
    BC::size_t padding = 0;

    BCINLINE BC::size_t  size() const { return rows() * cols() * this->dimension(2); }
    BCINLINE BC::size_t  rows() const { return right.rows() - left.rows() + padding + 1;  }
    BCINLINE BC::size_t  cols() const { return right.cols() - left.cols() + padding + 1; }
    BCINLINE BC::size_t  dimension(int i) const {
    	if (i == 0)
    		return rows();
    	else if (i == 1)
    		return cols();
    	else if (i == 2)
    		return left.dimension(3);
    	else
    		return 1;
    }
    BCINLINE BC::size_t  block_dimension(int i) const {
    	if (i == 0)
    		return rows();
    	else if (i == 1)
    		return cols()*rows();
    	else if (i == 2)
    		return left.dimension(3)*rows()*cols();
    	else
    		return 1;
    }

    Binary_Expression(lv left, rv right, BC::size_t stride_, BC::size_t padding_):
    	left(left), right(right), stride(stride_), padding(padding_) {}

    Binary_Expression(lv left, rv right, multichannel_conv2d<system_tag> default_=multichannel_conv2d<system_tag>()):
    	left(left), right(right), stride(1), padding(0) {}


    template<class core, int alpha_mod, int beta_mod, class Stream>
    void eval(injector<core, alpha_mod, beta_mod> injection_values, Stream stream) const {

    	for (int i = 0; i < tensor_dimension; ++i) {
    		BC_ASSERT(dimension(0) == injection_values.data().dimension(0),
    				"INVALID TENSOR INJECTION INCORRECT DIMENSIONS");
    	}

        //get the data of the injection --> injector simply stores the alpha/beta scalar modifiers
        auto& injection = injection_values.data();

        //greedy evaluate the whole expression, currently we do not support transposition/scalars/etc
        auto left_evaluated = greedy_evaluate(left, stream);
        auto right_evaluated = greedy_evaluate(right, stream);

		//call convolution (technically correlation)
        blas_impl::conv2d_3dtensor_multichannel(stream, injection.memptr(),
        		left_evaluated.memptr(), left_evaluated.rows(), left_evaluated.cols(), left_evaluated.dimension(2), left_evaluated.dimension(3),
        		right_evaluated.memptr(), right_evaluated.rows(), right_evaluated.cols(), right_evaluated.dimension(2));

        //deallocate if need be
        if (expression_traits<decltype(left_evaluated)>::is_temporary::value) {
        	using vt = typename decltype(left_evaluated)::value_type;

        	stream.template get_allocator_rebound<vt>().deallocate(left_evaluated.memptr(), left_evaluated.size());
        }
        if (expression_traits<decltype(right_evaluated)>::is_temporary::value) {
        	using vt = typename decltype(right_evaluated)::value_type;

        	stream.template get_allocator_rebound<vt>().deallocate(right_evaluated.memptr(), right_evaluated.size());
		}
    }
};

}
}
}


#endif /* FUNCTION_CONV2D_H_ */
