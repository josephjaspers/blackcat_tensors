#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "nd_evaluator/Evaluator.h"
#include "Tree_Greedy_Evaluator.h"

namespace BC {
namespace exprs {

template<class Expression, class Stream>
static void nd_evaluate(const Expression expr, Stream stream) {
	using system_tag = typename Stream::system_tag;
	using evaluator_impl  = typename BC::evaluator::template implementation<system_tag>;
	evaluator_impl::template nd_evaluator<Expression::tensor_iterator_dimension>(expr, stream);
}

template<class T>
static constexpr bool requires_greedy_eval() {
	return BC::exprs::tree::optimizer<T>::requires_greedy_eval;
}

//------------------------------------------------Purely lazy evaluation----------------------------------//
template<class Expression, class Stream>
static std::enable_if_t<!requires_greedy_eval<Expression>()>
evaluate(Expression expression, Stream stream) {
	nd_evaluate(expression, stream);
}
//------------------------------------------------Purely lazy alias evaluation----------------------------------//
template< class Expression, class Stream>
static std::enable_if_t<!requires_greedy_eval<Expression>()>
evaluate_aliased(Expression expression, Stream stream_) {
	nd_evaluate(expression, stream_);
}
//------------------------------------------------Greedy evaluation (BLAS function call detected)----------------------------------//
template< class Expression, class Stream>
static std::enable_if_t<requires_greedy_eval<Expression>()>
evaluate(Expression expr, Stream stream_) {
	auto greedy_evaluated_expr = BC::exprs::tree::evaluator_paths::evaluate(expr, stream_);

	if (expression_traits<decltype(greedy_evaluated_expr)>::is_expr) {
		nd_evaluate(greedy_evaluated_expr, stream_);
	}
	BC::exprs::tree::evaluator_paths::deallocate_temporaries(greedy_evaluated_expr, stream_);
}
//------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
template<class Expression, class Stream>
static std::enable_if_t<requires_greedy_eval<Expression>()>
evaluate_aliased(Expression expr, Stream stream_) {
	auto greedy_evaluated_expr = BC::exprs::tree::evaluator_paths::evaluate_aliased(expr, stream_);        //evaluate the internal tensor_type

	if (expression_traits<decltype(greedy_evaluated_expr)>::is_expr) {
		nd_evaluate(greedy_evaluated_expr, stream_);
	}
	BC::exprs::tree::evaluator_paths::deallocate_temporaries(greedy_evaluated_expr, stream_);
}



//-------------------------------- Evaluator Endpoints ---------------------------------- //
template<class Array, class Expression, class Stream>
auto greedy_evaluate(Array array, Expression expr, Stream stream) {
    static_assert(expression_traits<Array>::is_array, "MAY ONLY EVALUATE TO ARRAYS");
    evaluate(make_bin_expr<oper::assign>(array, expr), stream);
    return array;
}

//-------------------------------- Cache Evaluator Endpoints (used in blas functions) ---------------------------------- //

template<class Expression, class Stream, class=std::enable_if_t<expression_traits<Expression>::is_array>> //The branch is an array, no evaluation required
static auto greedy_evaluate(Expression expression, Stream stream) { return expression; }

template<class Expression, class Stream, class=std::enable_if_t<expression_traits<Expression>::is_expr>, int=0> //Create and return an array_core created from the expression
static auto greedy_evaluate(Expression expression, Stream stream) {
	using value_type = typename Expression::value_type;
	auto shape = BC::make_shape(expression.inner_shape());
	auto temporary = make_temporary_tensor_array<value_type>(shape, stream);
	return greedy_evaluate(temporary, expression, stream);
}

}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
