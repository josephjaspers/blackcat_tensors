#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "nd_evaluator/Evaluator.h"
#include "Tree_Greedy_Evaluator.h"

namespace BC {
namespace exprs {

struct Lazy_Evaluator {

	template<class Expression, class Context>
	static void nd_evaluate(const Expression expr, Context context) {
		using system_tag = typename Context::system_tag;
		using evaluator_impl  = typename evaluator::template implementation<system_tag>;
		evaluator_impl::template nd_evaluator<Expression::ITERATOR>(expr, context);
	}

    template<class T>
    static constexpr bool requires_greedy_eval() {
        return exprs::tree::optimizer<std::decay_t<T>>::requires_greedy_eval;
    }

    //------------------------------------------------Purely lazy evaluation----------------------------------//
    template< class expression, class Context>
    static std::enable_if_t<!requires_greedy_eval<expression>()>
    evaluate(const expression& expr, Context context_) {
    	nd_evaluate(expr, context_);
    }
    //------------------------------------------------Purely lazy alias evaluation----------------------------------//
    template< class expression, class Context>
    static std::enable_if_t<!requires_greedy_eval<expression>()>
    evaluate_aliased(const expression& expr, Context context_) {
    	nd_evaluate(expr, context_);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected)----------------------------------//
    template< class expression, class Context>
    static std::enable_if_t<requires_greedy_eval<expression>()>
    evaluate(const expression& expr, Context context_) {
        auto greedy_evaluated_expr = exprs::tree::Greedy_Evaluator::evaluate(expr, context_);

        if (BC::exprs::expression_traits<decltype(greedy_evaluated_expr)>::is_expr) {
        	nd_evaluate(greedy_evaluated_expr, context_);
        }
        exprs::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr, context_);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
    template< class expression, class Context>
    static std::enable_if_t<requires_greedy_eval<expression>()>
    evaluate_aliased(const expression& expr, Context context_) {
        auto greedy_evaluated_expr = exprs::tree::Greedy_Evaluator::evaluate_aliased(expr, context_);        //evaluate the internal tensor_type

        if (BC::exprs::expression_traits<decltype(greedy_evaluated_expr)>::is_expr) {
        	nd_evaluate(greedy_evaluated_expr, context_);
        }
        exprs::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr, context_);
    }
};



//-------------------------------- Evaluator Endpoints ---------------------------------- //
template<class array_t, class expression_t, class Context>
auto evaluate_to(array_t array, expression_t expr, Context context) {
    static_assert(expression_traits<array_t>::is_array, "MAY ONLY EVALUATE TO ARRAYS");
    return Lazy_Evaluator::evaluate(
    		exprs::make_bin_expr<oper::assign>(array, expr), context);
}

template<class expression_t, class Context>
auto evaluate(expression_t expr, Context context) {
    return Lazy_Evaluator::evaluate(expr, context);
}
template<class expression_t, class Context>
auto evaluate_aliased(expression_t expr, Context context) {
    return Lazy_Evaluator::evaluate_aliased(expr, context);
}

//-------------------------------- Cache Evaluator Endpoints (used in blas functions) ---------------------------------- //

template<class Context>
struct CacheEvaluator {

    template<class branch, class=std::enable_if_t<expression_traits<branch>::is_array>> //The branch is an array, no evaluation required
    static auto evaluate(const branch& expression, Context context_) { return expression; }


    template<class branch, class=std::enable_if_t<expression_traits<branch>::is_expr>, int=0> //Create and return an array_core created from the expression
    static auto evaluate(const branch& expression, Context context) {
    	using value_type = typename branch::value_type;
    	auto shape = BC::make_shape(expression.inner_shape());
    	auto temporary = BC::exprs::make_temporary_tensor_array<value_type>(shape, context);

    	return evaluate_to(temporary, expression, context);
    }

};



}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
