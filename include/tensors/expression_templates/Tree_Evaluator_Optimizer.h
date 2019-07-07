/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "Tree_Struct_Injector.h"
#include "Array.h"
#include "Expression_Binary.h"
#include "Expression_Unary.h"

namespace BC {
namespace tensors {
namespace exprs { 

template<class T, class voider=void>
struct optimizer;

template<class T>
struct optimizer_default {
	/*
	 * entirely_blas_expr -- if we may replace this branch entirely with a temporary/cache
	 * partial_blas_expr  -- if part of this branch contains a replaceable branch nested inside it
	 * nested_blas_expr   -- if a replaceable branch is inside a function (+=/-= won't work but basic assign = can work)
	 */

    static constexpr bool entirely_blas_expr = false;			//An expression of all +/- operands and BLAS calls				IE w*x + y*z
    static constexpr bool partial_blas_expr = false;			//An expression of element-wise +/- operations and BLAS calls	IE w + x*y
    static constexpr bool nested_blas_expr  = false;			//An expression containing a BLAS expression nested in a unary_functor IE abs(w * x)
    static constexpr bool requires_greedy_eval = false;			//Basic check if any BLAS call exists at all

    template<class core, int a, int b, class Stream>
    static auto linear_evaluation(T branch, injector<core, a, b> tensor, Stream) {
        return branch;
    }

    template<class core, int a, int b, class Stream>
    static auto injection(T branch, injector<core, a, b> tensor, Stream) {
        return branch;
    }

    template<class Stream>
    static auto temporary_injection(T branch, Stream alloc) {
        return branch;
    }

    template<class Stream>
    static void deallocate_temporaries(T, Stream alloc) {
        return;
    }
};

//-------------------------------- Array ----------------------------------------------------//
template<class T>
struct optimizer<T, std::enable_if_t<expression_traits<T>::is_array && !expression_traits<T>::is_temporary>>
: optimizer_default<T> {};

//--------------Temporary---------------------------------------------------------------------//

template<class Array>
struct optimizer<Array, std::enable_if_t<expression_traits<Array>::is_temporary>>
 : optimizer_default<Array> {

	template<class Stream>
    static void deallocate_temporaries(Array tmp, Stream alloc) {
        destroy_temporary_kernel_array(tmp, alloc);
    }
};


//-----------------------------------------------BLAS----------------------------------------//

template<class op, class lv, class rv>
struct optimizer<Binary_Expression<op, lv, rv>, std::enable_if_t<oper::operation_traits<op>::is_blas_function>> {

	static constexpr bool entirely_blas_expr = true;
    static constexpr bool partial_blas_expr = true;
    static constexpr bool nested_blas_expr = true;
    static constexpr bool requires_greedy_eval = true;

    template<class core, int a, int b, class Stream>
    static auto linear_evaluation(Binary_Expression<op, lv, rv> branch, injector<core, a, b> tensor, Stream alloc) {
    	branch.eval(tensor, alloc);
        return tensor.data();
    }
    template<class core, int a, int b, class Stream>
    static auto injection(Binary_Expression<op, lv, rv> branch, injector<core, a, b> tensor, Stream alloc) {
        branch.eval(tensor, alloc);
        return tensor.data();
    }

    //if no replacement is used yet, auto inject
    template<class Stream>
    static auto temporary_injection(Binary_Expression<op, lv, rv> branch, Stream alloc) {
    	using value_type = typename Binary_Expression<op, lv, rv>::value_type;
    	auto temporary = make_temporary_kernel_array<value_type>(make_shape(branch.inner_shape()), alloc);
        branch.eval(make_injection<1, 0>(temporary), alloc);
        return temporary;
    }

    template<class Stream>
    static void deallocate_temporaries(Binary_Expression<op, lv, rv> branch, Stream alloc) {
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
    }
};


//-------------------------------- Linear ----------------------------------------------------//

template<class op, class lv, class rv>
struct optimizer<Binary_Expression<op, lv, rv>, std::enable_if_t<oper::operation_traits<op>::is_linear_operation>> {
    static constexpr bool entirely_blas_expr 	= optimizer<lv>::entirely_blas_expr && optimizer<rv>::entirely_blas_expr;
    static constexpr bool partial_blas_expr 	= optimizer<lv>::partial_blas_expr || optimizer<rv>::partial_blas_expr;
    static constexpr bool nested_blas_expr 		= optimizer<lv>::nested_blas_expr || optimizer<rv>::nested_blas_expr;
    static constexpr bool requires_greedy_eval 	= optimizer<lv>::requires_greedy_eval || optimizer<rv>::requires_greedy_eval;


    template<class core, int a, int b, class Stream>
    static auto linear_evaluation(Binary_Expression<op, lv, rv>& branch, injector<core, a, b> tensor, Stream alloc) {
        return
        		BC::meta::constexpr_if<entirely_blas_expr>(
        				[&](){
							optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
							optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
							return tensor.data();
        				},
				BC::meta::constexpr_else_if<optimizer<lv>::entirely_blas_expr>(
        				[&]() {
        					return BC::meta::constexpr_ternary<std::is_same<op, oper::Sub>::value>(
        							[&]() {
										/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
										auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
										return make_un_expr<oper::negation>(right);

        							},
        							[&]() {
										/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
										auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
										return right;
        							}
        					);
        				},
				BC::meta::constexpr_else_if<optimizer<rv>::entirely_blas_expr>(
        				[&]() {
								auto left  = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
							  /*auto right = */ optimizer<rv>::linear_evaluation(branch.right, update_injection<op, b != 0>(tensor), alloc);
								return left;
						},
				BC::meta::constexpr_else_if<optimizer<rv>::nested_blas_expr>(
        				[&]() {
				            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
				            return make_bin_expr<op>(branch.left, right);
						},
				BC::meta::constexpr_else(
						[&]() {
				        	static constexpr bool left_evaluated = optimizer<lv>::partial_blas_expr || b != 0;
				        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
				            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, left_evaluated>(tensor), alloc);
				            return make_bin_expr<op>(left, right);
						}
				)))));
    }

    //---------------partial blas expr branches-------------------------//
    struct evaluate_branch {
        template<class core, int a, int b, class Stream>
        static auto function(Binary_Expression<op, lv, rv> branch, injector<core, a, b> tensor, Stream alloc) {
        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor);
        	auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
            return make_bin_expr<op>(left, right);
        }
    };


    template<class core, int a, int b, class Stream>
    static auto injection(Binary_Expression<op, lv, rv> branch, injector<core, a, b> tensor, Stream alloc) {

    	auto basic_eval = [&]() {
        	static constexpr bool left_evaluated = optimizer<lv>::partial_blas_expr || b != 0;
        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, left_evaluated>(tensor), alloc);
            return make_bin_expr<op>(left, right);
    	};


        return
        		BC::meta::constexpr_if<entirely_blas_expr>(
        				[&](){
							optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
							optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
							return tensor.data();
        				},
				BC::meta::constexpr_else_if<optimizer<rv>::partial_blas_expr && optimizer<lv>::partial_blas_expr>(
        					basic_eval,
				BC::meta::constexpr_else_if<optimizer<lv>::nested_blas_expr>(
        				[&]() {
							auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
							return make_bin_expr<op>(left, branch.right);
						},
				BC::meta::constexpr_else_if<optimizer<rv>::nested_blas_expr>(
        				[&]() {
				            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
				            return make_bin_expr<op>(branch.left, right);
						},
				BC::meta::constexpr_else(
						basic_eval
				)))));
    }


    //---------substitution implementation---------//

    template<class Stream>
    static auto temporary_injection(Binary_Expression<op, lv, rv> branch, Stream alloc) {
    	auto left  = optimizer<lv>::template temporary_injection<Stream>(branch.left, alloc);
    	auto right = optimizer<rv>::template temporary_injection<Stream>(branch.right, alloc);
    	return make_bin_expr<op>(left, right);
    }


    template<class Stream>
    static void deallocate_temporaries(Binary_Expression<op, lv, rv> branch, Stream alloc) {
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
    }

};

//------------------------------Non linear-------------------------------------------//

template<class op, class lv, class rv>
struct optimizer<Binary_Expression<op, lv, rv>, std::enable_if_t<oper::operation_traits<op>::is_nonlinear_operation >> {
    static constexpr bool entirely_blas_expr = false;
    static constexpr bool partial_blas_expr = false;
    static constexpr bool nested_blas_expr = optimizer<lv>::nested_blas_expr || optimizer<rv>::nested_blas_expr;
    static constexpr bool requires_greedy_eval = optimizer<lv>::requires_greedy_eval || optimizer<rv>::requires_greedy_eval;


    template<class core, int a, int b, class Stream>
    static auto linear_evaluation(Binary_Expression<op, lv, rv> branch, injector<core, a, b> tensor, Stream) {
        return branch;
    }

    template<class core, int a, int b, class Stream>
    static auto injection(Binary_Expression<op, lv, rv> branch, injector<core, a, b> tensor, Stream alloc) {
    	return BC::meta::constexpr_ternary<optimizer<lv>::partial_blas_expr || optimizer<lv>::nested_blas_expr>(
					[&]() {
						auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
						auto right = branch.right;
						return make_bin_expr<op>(left, right, branch.get_operation());
					},
					[&]() {
						auto left = branch.left;
						auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
						return make_bin_expr<op>(left, right, branch.get_operation());
			}
    	);
    }

    template<class Stream>
    static auto temporary_injection(Binary_Expression<op, lv, rv> branch, Stream& alloc) {
    	auto left  = optimizer<lv>::template temporary_injection<Stream>(branch.left, alloc);
    	auto right = optimizer<rv>::template temporary_injection<Stream>(branch.right, alloc);
    	return make_bin_expr<op>(left, right, branch.get_operation());
    }

    template<class Stream>
    static void deallocate_temporaries(Binary_Expression<op, lv, rv> branch, Stream alloc) {
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
    	optimizer<lv>::deallocate_temporaries(branch.left, alloc);
    }
};




//--------------Unary Expression---------------------------------------------------------------------//

template<class Op, class Array>
struct optimizer<Unary_Expression<Op, Array>>
{
    static constexpr bool entirely_blas_expr 	= false;
    static constexpr bool partial_blas_expr 	= false;
    static constexpr bool nested_blas_expr 		= optimizer<Array>::nested_blas_expr;
    static constexpr bool requires_greedy_eval 	= optimizer<Array>::requires_greedy_eval;

    template<class core, int a, int b, class Stream>
    static auto linear_evaluation(Unary_Expression<Op, Array> branch, injector<core, a, b> tensor, Stream) {
        return branch;
    }
    template<class core, int a, int b, class Stream>
    static auto injection(Unary_Expression<Op, Array> branch, injector<core, a, b> tensor, Stream alloc) {
        auto array =  optimizer<Array>::injection(branch.array, tensor, alloc);
        return make_un_expr(array, branch.get_operation());
    }

    template<class Stream>
    static auto temporary_injection(Unary_Expression<Op, Array> branch, Stream& alloc) {
    	auto expr = optimizer<Array>::template temporary_injection<Stream>(branch.array, alloc);
    	return make_un_expr(expr, branch.get_operation());

    }
    template<class Stream>
     static void deallocate_temporaries(Unary_Expression<Op, Array> branch, Stream alloc) {
        optimizer<Array>::deallocate_temporaries(branch.array, alloc);
    }
};

} //ns exprs
} //ns tensors
} //ns BC



#endif /* PTE_ARRAY_H_ */
