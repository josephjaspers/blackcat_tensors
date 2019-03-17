/*  Project: BlackCat_Tensors
 *  Author: Joseph Jaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef PTE_ARRAY_H_
#define PTE_ARRAY_H_

#ifdef BC_TREE_OPTIMIZER_DEBUG
#define BC_TREE_OPTIMIZER_STDOUT(literal) std::cout << literal << std::endl;
#else
#define BC_TREE_OPTIMIZER_STDOUT(literal)
#endif

#include <type_traits>
#include "Common.h"
#include "Tree_Struct_Injector.h"
#include "Array.h"
#include "Expression_Binary.h"
#include "Expression_Unary.h"


namespace BC {
namespace exprs {
namespace tree {
using namespace oper;

//pure_linear_expr -- detects if the tree is entirely +/- operations with blas functions, --> y = a * b + c * d - e * f  --> true, y = a + b * c --> false
template<class op, bool prior_eval, class core, BC::size_t  a, BC::size_t  b>//only apply update if right hand side branch
auto update_injection(injector<core,a,b> tensor) {
    static constexpr BC::size_t  alpha =  a * BC::oper::operation_traits<op>::alpha_modifier;
    static constexpr BC::size_t  beta  = prior_eval ? 1 : 0;
    return injector<core, alpha, beta>(tensor.data());
}

template<class T>
struct optimizer_default {
	/*
	 * pure_linear_expr -- if we may replace this branch entirely with a temporary/cache
	 * linear_expr  -- if part of this branch contains a replaceable branch nested inside it
	 * nested_linear_expr   -- if a replaceable branch is inside a function (+=/-= won't work but basic assign = can work)
	 */

	//REMEMBER TO UPDATE TREE_GREEDY_EVALUATOR.H && TREE_LAZY_EVALUATOR.H IF THESE ARE REFACTORED
    template<int output_dimension> static constexpr bool pure_linear_expr = false;		//An expression of all +/- operands and BLAS calls				IE w*x + y*z
    template<int output_dimension> static constexpr bool linear_expr = false;			//An expression of element-wise +/- operations and BLAS calls	IE w + x*y
    template<int output_dimension> static constexpr bool nested_linear_expr  = false;	//An expression containing a BLAS expression nested in a unary_functor IE abs(w * x)
    static constexpr bool contains_blas_function = false;			//Basic check if any BLAS call exists at all

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
    	BC_TREE_OPTIMIZER_STDOUT("DEFAULT DEALLOCATE TEMPORARIES");
        return;
    }
};



template<class T, class voider=void>
struct optimizer;

template<class Expression, class Context>
auto evaluate_linear(Expression expr, Context context) {
	return optimizer<Expression>::linear_evaluation(expr, context);
}
template<class Expression, class Context>
auto evaluate_temporary_destruction(Expression expr, Context context) {
	return optimizer<Expression>::linear_evaluation(expr, context);
}
template<class Expression, class Context>
auto evaluate_injection(Expression expr, Context context) {
	return optimizer<Expression>::injection(expr, context);
}
template<class Expression, class Context>
auto evaluate_temporary_injection(Expression expr, Context context) {
	return optimizer<Expression>::temporary_injection(expr, context);
}



//-------------------------------- Array ----------------------------------------------------//
template<class T>
struct optimizer<T, std::enable_if_t<expression_traits<T>::is_array && !expression_traits<T>::is_temporary>>
: optimizer_default<T> {};

//--------------Temporary---------------------------------------------------------------------//

template<int x, class Scalar, class Context>
struct optimizer<
	ArrayExpression<x, Scalar, Context, BC_Temporary>,
	std::enable_if_t<expression_traits<ArrayExpression<x, Scalar, Context, BC_Temporary>>::is_temporary>>
 : optimizer_default<ArrayExpression<x, Scalar, Context, BC_Temporary>> {


	template<class Context_>
    static void deallocate_temporaries(ArrayExpression<x, Scalar, Context, BC_Temporary> tmp, Context_ alloc) {
        alloc.get_allocator().deallocate(tmp.array, tmp.size());
    }
};


//-----------------------------------------------BLAS----------------------------------------//



template<class lv, class rv, class op>
struct optimizer<Binary_Expression<lv, rv, op>, std::enable_if_t<oper::operation_traits<op>::is_blas_function>> {
	template<int output_dimension> static constexpr bool pure_linear_expr 		= output_dimension == op::DIMS;
	template<int output_dimension> static constexpr bool linear_expr 			= output_dimension == op::DIMS;
	template<int output_dimension> static constexpr bool nested_linear_expr 	= output_dimension == op::DIMS;
	static constexpr bool contains_blas_function = true;


    using branch_t = Binary_Expression<lv, rv, op>;

    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: linear_evaluation" << "alpha=" << a << "beta=" << b);

    	branch.eval(tensor, alloc);
        return tensor.data();
    }
    template<class core, int a, int b, class Context>
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: injection");
        branch.eval(tensor, alloc);
        return tensor.data();
    }

    //if no replacement is used yet, auto inject
    template<class Context>
    static auto temporary_injection(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: temporary_injection");

    	using param = Binary_Expression<lv, rv, op>;
    	using allocator_t = typename Context::allocator_t; //refactor allocator to context!!
    	using value_type = typename allocator_t::value_type;
    	constexpr int dims = param::DIMS;

    	using tmp_t = Array<dims, value_type, allocator_t, BC_Temporary>;
        tmp_t tmp(branch.inner_shape(), alloc.get_allocator());


        //ISSUE HERE
        branch.eval(make_injection<1, 0>(tmp.internal()), alloc);
        return tmp.internal();
    }

    template<class Context>
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("BLAS_EXPR: deallocate_temporaries");

        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
    }
};


//-------------------------------- Linear ----------------------------------------------------//


template<class lv, class rv, class op>
struct optimizer<Binary_Expression<lv, rv, op>, std::enable_if_t<operation_traits<op>::is_linear_operation>> {

	template<int output_dim>
    static constexpr bool pure_linear_expr 	= optimizer<lv>::template pure_linear_expr<output_dim> &&
    											optimizer<rv>::template pure_linear_expr<output_dim>;

    template<int output_dim>
    static constexpr bool linear_expr = optimizer<lv>::template linear_expr<output_dim> ||
    										optimizer<rv>::template linear_expr<output_dim>;

    template<int output_dim>
    static constexpr bool nested_linear_expr = optimizer<lv>::template nested_linear_expr<output_dim> ||
    											optimizer<rv>::template nested_linear_expr<output_dim>;

    static constexpr bool contains_blas_function 	= optimizer<lv>::contains_blas_function ||
    													optimizer<rv>::contains_blas_function;


    //-------------Linear evaluation branches---------------------//


    //needed by injection and linear_evaluation
    struct remove_branch {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_branch (entire)");

        	optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
        	optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
        	return tensor.data();
        }
    };

    //if left is entirely blas_expr
    struct remove_left_branch {

        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_left_branch");

        	/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
            return right;
        }
    };
    struct remove_left_branch_and_negate {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, oper::sub>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_left_branch");

        	/*auto left = */ optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);
            return make_un_expr<oper::negation>(right);
        }
    };

    //if right is entirely blas_expr (or if no blas expr)

    struct remove_right_branch {

    	template<class core, int a, int b, class Context>
    	static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- remove_right_branch");

    		auto left  = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
          /*auto right = */ optimizer<rv>::linear_evaluation(branch.right, update_injection<op, b != 0>(tensor), alloc);
            return left;
        }
    };

    struct basic_eval {

        template<class core, int a, int b, class Context>
    	static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- basic_eval");

        	static constexpr bool left_evaluated = optimizer<lv>::template linear_expr<core::DIMS> || b != 0;
        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, left_evaluated>(tensor), alloc);
            return make_bin_expr<op>(left, right);
        }
    };


    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: linear_evaluation");

        using impl =
        		std::conditional_t<pure_linear_expr<core::DIMS>, remove_branch,
        		std::conditional_t<optimizer<lv>::template pure_linear_expr<core::DIMS>,
        		std::conditional_t<std::is_same<op, oper::sub>::value, remove_left_branch_and_negate, remove_left_branch>,
        		std::conditional_t<optimizer<rv>::template pure_linear_expr<core::DIMS>, remove_right_branch,
        		basic_eval>>>;

        return impl::function(branch, tensor, alloc);
    }

    //---------------partial blas expr branches-------------------------//
    struct evaluate_branch {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- evaluate_branch");
        	auto left = optimizer<lv>::linear_evaluation(branch.left, tensor);
        	auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor));
            return make_bin_expr<op>(left, right);
        }
    };


    struct left_blas_expr {

        struct trivial_injection {
            template<class l, class r, class Context>
            static auto function(const l& left, const r& right, Context) {
	        	BC_TREE_OPTIMIZER_STDOUT("-- trivial_injection");
            	return left;
            }
        };

		struct non_trivial_injection {
			template<class LeftVal, class RightVal, class Context>
			static auto function(const LeftVal& left, const RightVal& right, Context) {
	        	BC_TREE_OPTIMIZER_STDOUT("-- non_trivial_injection");
	            return make_bin_expr<op>(left, right);
			}
		};

        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- left_blas_expr");

            auto left = optimizer<lv>::injection(branch.left, tensor, alloc);

            //check this ?
            auto right = optimizer<rv>::linear_evaluation(branch.right, update_injection<op, true>(tensor), alloc);

            using impl = std::conditional_t<optimizer<rv>::template pure_linear_expr<core::DIMS>,
                    trivial_injection, non_trivial_injection>;

            return impl::function(left, right, alloc);
        }
    };
    struct right_blas_expr {
        struct trivial_injection {
            template<class l, class r, class Context>
            static auto function(const l& left, const r& right, Context alloc) {
            	BC_TREE_OPTIMIZER_STDOUT("-- trivial_injection");
                return right;
            }
        };
        struct non_trivial_injection {
                template<class l, class r, class Context>
                static auto function(const l& left, const r& right, Context alloc) {
                	BC_TREE_OPTIMIZER_STDOUT("-- non_trivial_injection");
                    return make_bin_expr<op>(left, right);
                }
            };
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- right_blas_expr");

            auto left = optimizer<lv>::linear_evaluation(branch.left, tensor, alloc);
            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);

            using impl = std::conditional_t<optimizer<lv>::template pure_linear_expr<core::DIMS>,
                    trivial_injection, non_trivial_injection>;

            return impl::function(left, right, alloc);
        }
    };

    //------------nontrivial injections, DO NOT UPDATE, scalars-mods will enact during elementwise-evaluator----------------//
    struct left_nested_linear_expr {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- left_nested_linear_expr");

            auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
            rv right = branch.right; //rv
            return make_bin_expr<op>(left, right);
        }
    };
    struct right_nested_linear_expr {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
        	BC_TREE_OPTIMIZER_STDOUT("- right_nested_linear_expr");

            lv left = branch.left; //lv
            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
            return make_bin_expr<op>(left, right);
        }
    };


    template<class core, int a, int b, class Context>
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: Injection");

    	struct to_linear_eval {
    		static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
    	    	BC_TREE_OPTIMIZER_STDOUT("-flip to linear_eval");

    			return linear_evaluation(branch, tensor, alloc);
    		}
    	};

        using impl =
          		std::conditional_t<pure_linear_expr<core::DIMS>, remove_branch,
        		std::conditional_t<optimizer<rv>::template linear_expr<core::DIMS> && optimizer<lv>::template linear_expr<core::DIMS>, basic_eval,
        		std::conditional_t<optimizer<lv>::template nested_linear_expr<core::DIMS>, left_nested_linear_expr,
                std::conditional_t<optimizer<rv>::template nested_linear_expr<core::DIMS>, right_nested_linear_expr, basic_eval>>>>;

        static_assert(!std::is_same<void, impl>::value, "EXPRESSION_REORDERING COMPILATION FAILURE, USE 'ALIAS' AS A WORKAROUND");
        return impl::function(branch, tensor, alloc);
    }


    //---------substitution implementation---------//

    template<class Context>
    static auto temporary_injection(const Binary_Expression<lv,rv,op>& branch, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: Temporary Injection");

    	auto left  = optimizer<lv>::template temporary_injection<Context>(branch.left, alloc);
    	auto right = optimizer<rv>::template temporary_injection<Context>(branch.right, alloc);
    	return make_bin_expr<op>(left, right);
    }


    template<class Context>
    static void deallocate_temporaries(const Binary_Expression<lv, rv, op>& branch, Context alloc) {
    	BC_TREE_OPTIMIZER_STDOUT("Binary Linear: Deallocate Temporaries");

        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
    }

};

//------------------------------Non linear-------------------------------------------//

template<class lv, class rv, class op>
struct optimizer<Binary_Expression<lv, rv, op>, std::enable_if_t<operation_traits<op>::is_nonlinear_operation >> {
    template<int output_dim> static constexpr bool pure_linear_expr = false;
    template<int output_dim> static constexpr bool linear_expr = false;

    template<int output_dim>
    static constexpr bool nested_linear_expr = optimizer<lv>::template nested_linear_expr<output_dim> ||
    											optimizer<rv>::template nested_linear_expr<output_dim>;

    static constexpr bool contains_blas_function = optimizer<lv>::contains_blas_function || optimizer<rv>::contains_blas_function;


    template<class core, int a, int b, class Context>
    static auto linear_evaluation(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context) {
        return branch;
    }


    struct left_trivial_injection {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
            auto right = branch.right;
            return make_bin_expr<op>(left, right);
        }
    };
    struct right_trivial_injection {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            auto left = branch.left;
            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
            return make_bin_expr<op>(left, right);
        }
    };
    struct left_nontrivial_injection {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            auto left = optimizer<lv>::injection(branch.left, tensor, alloc);
            auto right = branch.right; //rv
            return make_bin_expr<op>(left, right);
        }
    };
    struct right_nontrivial_injection {
        template<class core, int a, int b, class Context>
        static auto function(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            auto left = branch.left; //lv
            auto right = optimizer<rv>::injection(branch.right, tensor, alloc);
            return make_bin_expr<op>(left, right);
        }
    };

    template<class core, int a, int b, class Context>
    static auto injection(const Binary_Expression<lv, rv, op>& branch, injector<core, a, b> tensor, Context alloc) {
            //dont need to update injection
            //trivial injection left_hand side (we attempt to prefer trivial injections opposed to non-trivial)
            using impl  =
                std::conditional_t<optimizer<lv>::template linear_expr<lv::DIMS>,         left_trivial_injection,
                std::conditional_t<optimizer<rv>::template linear_expr<lv::DIMS>,         right_trivial_injection,
                std::conditional_t<optimizer<lv>::template nested_linear_expr<lv::DIMS>,     left_nontrivial_injection,
                std::conditional_t<optimizer<rv>::template nested_linear_expr<lv::DIMS>,     right_nontrivial_injection, void>>>>;

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
    	BC_TREE_OPTIMIZER_STDOUT("Binary NonLinear: DEALLOCATE TEMPORARIES");
        optimizer<lv>::deallocate_temporaries(branch.left, alloc);
        optimizer<rv>::deallocate_temporaries(branch.right, alloc);
    }
};




//--------------Unary Expression---------------------------------------------------------------------//


template<class array_t, class op>
struct optimizer<Unary_Expression<array_t, op>>
{
	template<int output_dim> static constexpr bool pure_linear_expr   = false;
	template<int output_dim> static constexpr bool linear_expr 		  = false;
	template<int output_dim> static constexpr bool nested_linear_expr
	= optimizer<array_t>::template nested_linear_expr<output_dim>;
    static constexpr bool contains_blas_function 	= optimizer<array_t>::contains_blas_function;

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
    	BC_TREE_OPTIMIZER_STDOUT("Unary Expression: Deallocate Temporaries");
        optimizer<array_t>::deallocate_temporaries(branch.array, alloc);
    }
};




}
}
}



#endif /* PTE_ARRAY_H_ */
