#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "nd_evaluator/Evaluator.h"
#include "Tree_Greedy_Evaluator.h"

namespace BC {
namespace exprs {

template<class Expression, class Context>
static void nd_evaluate(const Expression expr, Context context) {
	using system_tag = typename Context::system_tag;
	using evaluator_impl  = typename BC::evaluator::template implementation<system_tag>;
	evaluator_impl::template nd_evaluator<Expression::ITERATOR>(expr, context);
}

template<class T>
static constexpr bool requires_greedy_eval() {
	return BC::exprs::tree::optimizer<T>::requires_greedy_eval;
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
	auto greedy_evaluated_expr = BC::exprs::tree::evaluator_paths::evaluate(expr, context_);

	if (expression_traits<decltype(greedy_evaluated_expr)>::is_expr) {
		nd_evaluate(greedy_evaluated_expr, context_);
	}
	BC::exprs::tree::evaluator_paths::deallocate_temporaries(greedy_evaluated_expr, context_);
}
//------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
template< class expression, class Context>
static std::enable_if_t<requires_greedy_eval<expression>()>
evaluate_aliased(const expression& expr, Context context_) {
	auto greedy_evaluated_expr = BC::exprs::tree::evaluator_paths::evaluate_aliased(expr, context_);        //evaluate the internal tensor_type

	if (expression_traits<decltype(greedy_evaluated_expr)>::is_expr) {
		nd_evaluate(greedy_evaluated_expr, context_);
	}
	BC::exprs::tree::evaluator_paths::deallocate_temporaries(greedy_evaluated_expr, context_);
}



//-------------------------------- Evaluator Endpoints ---------------------------------- //
template<class Array, class Expression, class Context>
void greedy_evaluate(Array array, Expression expr, Context context) {
    static_assert(expression_traits<Array>::is_array, "MAY ONLY EVALUATE TO ARRAYS");
    evaluate(make_bin_expr<oper::assign>(array, expr), context);
}

//-------------------------------- Cache Evaluator Endpoints (used in blas functions) ---------------------------------- //

template<class Expression, class Context, class=std::enable_if_t<expression_traits<Expression>::is_array>> //The branch is an array, no evaluation required
static auto greedy_evaluate(const Expression& expression, Context context) { return expression; }

template<class Expression, class Context, class=std::enable_if_t<expression_traits<Expression>::is_expr>, int=0> //Create and return an array_core created from the expression
static auto greedy_evaluate(const Expression& expression, Context context) {
	using value_type = typename Expression::value_type;
	auto shape = BC::make_shape(expression.inner_shape());
	auto temporary = make_temporary_tensor_array<value_type>(shape, context);

	return evaluate_to(temporary, expression, context);
}

}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
