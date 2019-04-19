/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#include "Common.h"
#include "Tree_Struct_Injector.h"

#include "Array.h"
#include "Expression_Binary.h"
#include "Expression_Unary.h"

namespace BC {
namespace exprs {
namespace tree {

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

    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const T& branch, injector<core, a, b> tensor, Context) {
        return branch;
    }

    template<class core, int a, int b, class Context>
    static auto injection(const T& branch, injector<core, a, b> tensor, Context) {
        return branch;
    }

    template<class Context>
    static auto temporary_injection(const T& branch, Context alloc) {
        return branch;
    }

    template<class Context>
    static void deallocate_temporaries(const T, Context alloc) {
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

	template<class Context>
    static void deallocate_temporaries(Array tmp, Context alloc) {
        destroy_temporary_tensor_array(tmp, alloc);
    }
};


//-----------------------------------------------BLAS----------------------------------------//

template<class lv, class rv, class op>
struct optimizer<Binary_Expression<lv, rv, op>, std::enable_if_t<oper::operation_traits<op>::is_blas_function>> {
    static constexpr bool entirely_blas_expr = true;
    static constexpr bool partial_blas_expr = true;
    static constexpr bool nested_blas_expr = true;
    static constexpr bool requires_greedy_eval = true;


    using branch_t = Binary_Expression<lv, rv, op>;

    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
    	branch.eval(tensor, alloc);
        return tensor.data();
    }
    template<class core, int a, int b, class Context>
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        branch.eval(tensor, alloc);
        return tensor.data();
    }

    //if no replacement is used yet, auto inject
    template<class Context>
    static auto temporary_injection(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
    	using value_type = typename Binary_Expression<lv, rv, op>::value_type;
    	auto temporary = make_temporary_tensor_array<value_type>(make_shape(branch.inner_shape()), alloc);
        branch.eval(make_injection<1, 0>(temporary), alloc);
        return temporary;
    }

    template<class Context>
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
    }
};


//-------------------------------- Linear ----------------------------------------------------//

template<class lv, class rv, class op>
struct optimizer<Binary_Expression<lv, rv, op>, std::enable_if_t<oper::operation_traits<op>::is_linear_operation>> {
    static constexpr bool entirely_blas_expr 	= optimizer<lv>::entirely_blas_expr && optimizer<rv>::entirely_blas_expr;
    static constexpr bool partial_blas_expr 	= optimizer<lv>::partial_blas_expr || optimizer<rv>::partial_blas_expr;
    static constexpr bool nested_blas_expr 		= optimizer<lv>::nested_blas_expr || optimizer<rv>::nested_blas_expr;
    static constexpr bool requires_greedy_eval 	= optimizer<lv>::requires_greedy_eval || optimizer<rv>::requires_greedy_eval;


    //-------------Linear evaluation branches---------------------//
    struct remove_branch {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
        	optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
        	return tensor.data();
        }
    };
    struct basic_eval {
        template<class core, int a, int b, class Context>
    	static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	static constexpr bool left_evaluated = optimizer<lv>::partial_blas_expr || b != 0;
        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, left_evaluated>(tensor), alloc);
            return make_bin_expr<op>(left, right);
        }
    };

    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        struct remove_left_branch {
            static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            	/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
                auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
                return right;
            }
        };
        struct remove_left_branch_and_negate {
            static auto function(const Binary_Expression<lv, rv, oper::sub>& branch, injector<core, a, b> tensor, Context alloc) {
            	/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
                auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
                return make_un_expr<oper::negation>(right);
            }
        };
        struct remove_right_branch {
        	static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        		auto left  = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
              /*auto right = */ optimizer<rv>::linear_evaluation(branch.right, update_injection<op, b != 0>(tensor), alloc);
                return left;
            }
        };

        using impl =
        		std::conditional_t<entirely_blas_expr, remove_branch,
        		std::conditional_t<optimizer<lv>::entirely_blas_expr,
        		std::conditional_t<std::is_same<op, oper::sub>::value, remove_left_branch_and_negate, remove_left_branch>,
        		std::conditional_t<optimizer<rv>::entirely_blas_expr, remove_right_branch,
        		basic_eval>>>;

        return impl::function(branch, tensor, alloc);
    }

    //---------------partial blas expr branches-------------------------//
    struct evaluate_branch {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor);
        	auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
            return make_bin_expr<op>(left, right);
        }
    };


    //------------nontrivial injections, DO NOT UPDATE, scalars-mods will enact during elementwise-evaluator----------------//
    struct left_nested_blas_expr {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
            return make_bin_expr<op>(left, branch.right);
        }
    };
    struct right_nested_blas_expr {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
            return make_bin_expr<op>(branch.left, right);
        }
    };


    template<class core, int a, int b, class Context>
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        using impl =
          		std::conditional_t<entirely_blas_expr, remove_branch,
        		std::conditional_t<optimizer<rv>::partial_blas_expr && optimizer<lv>::partial_blas_expr, basic_eval,
        		std::conditional_t<optimizer<lv>::nested_blas_expr, left_nested_blas_expr,
                std::conditional_t<optimizer<rv>::nested_blas_expr, right_nested_blas_expr, basic_eval>>>>;
        return impl::function(branch, tensor, alloc);
    }


    //---------substitution implementation---------//

    template<class Context>
    static auto temporary_injection(const Binary_Expression<lv,rv,op>& branch, Context alloc) {
    	auto left  = optimizer<lv>::template temporary_injection<Context>(branch.left, alloc);
    	auto right = optimizer<rv>::template temporary_injection<Context>(branch.right, alloc);
    	return make_bin_expr<op>(left, right);
    }


    template<class Context>
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
    }

};

//------------------------------Non linear-------------------------------------------//

template<class lv, class rv, class op>
struct optimizer<Binary_Expression<lv, rv, op>, std::enable_if_t<oper::operation_traits<op>::is_nonlinear_operation >> {
    static constexpr bool entirely_blas_expr = false;
    static constexpr bool partial_blas_expr = false;
    static constexpr bool nested_blas_expr = optimizer<lv>::nested_blas_expr || optimizer<rv>::nested_blas_expr;
    static constexpr bool requires_greedy_eval = optimizer<lv>::requires_greedy_eval || optimizer<rv>::requires_greedy_eval;


    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context) {
        return branch;
    }

    template<class core, int a, int b, class Context>
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            //dont need to update injection
            //trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
        struct left {
            static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
                auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
                auto right = branch.right;
                return make_bin_expr<op>(left, right);
            }
        };
        struct right {
            static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
                auto left = branch.left;
                auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
                return make_bin_expr<op>(left, right);
            }
        };

    	using impl  =
                std::conditional_t<optimizer<lv>::partial_blas_expr, left,
                std::conditional_t<optimizer<lv>::nested_blas_expr,  left,
                std::conditional_t<optimizer<rv>::partial_blas_expr, right,
                std::conditional_t<optimizer<rv>::nested_blas_expr,  right, void>>>>;

            return impl::function(branch, tensor, alloc);
    }

    template<class Context>
    static auto temporary_injection(const Binary_Expression<lv,rv,op>& branch, Context& alloc) {
    	auto left  = optimizer<lv>::template temporary_injection<Context>(branch.left, alloc);
    	auto right = optimizer<rv>::template temporary_injection<Context>(branch.right, alloc);
    	return make_bin_expr<op>(left, right);
    }

    template<class Context>
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
    	optimizer<lv>::deallocate_temporaries(branch.left, alloc);
    }
};




//--------------Unary Expression---------------------------------------------------------------------//

template<class array_t, class op>
struct optimizer<Unary_Expression<array_t, op>>
{
    static constexpr bool entirely_blas_expr 	= false;
    static constexpr bool partial_blas_expr 	= false;
    static constexpr bool nested_blas_expr 		= optimizer<array_t>::nested_blas_expr;
    static constexpr bool requires_greedy_eval 	= optimizer<array_t>::requires_greedy_eval;

    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Unary_Expression<array_t, op>& branch, injector<core, a, b> tensor, Context) {
        return branch;
    }
    template<class core, int a, int b, class Context>
    static auto injection(const Unary_Expression<array_t, op>& branch, injector<core, a, b> tensor, Context alloc) {
        auto array =  optimizer<array_t>::injection(branch.array, tensor, alloc);
        return Unary_Expression<decltype(array), op>(array);
    }

    template<class Context>
    static auto temporary_injection(const Unary_Expression<array_t, op>& branch, Context& alloc) {
    	auto expr = optimizer<array_t>::template temporary_injection<Context>(branch.array, alloc);
    	return Unary_Expression<std::decay_t<decltype(expr)>, op>(expr);

    }
    template<class Context>
     static void deallocate_temporaries(const Unary_Expression<array_t, op>& branch, Context alloc) {
        optimizer<array_t>::deallocate_temporaries(branch.array, alloc);
    }
};




}
}
}



#endif /* PTE_ARRAY_H_ */
