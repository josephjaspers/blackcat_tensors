#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "Tree_Evaluator_Runner_Impl.h"

namespace BC {
namespace et {

template<class context_type>
struct Lazy_Evaluator {

	template<class lv, class rv>
	static constexpr bool decay_same = std::is_same<std::decay_t<lv>, std::decay_t<rv>>::value;

	using impl = typename evaluator::template implementation<context_type>;

    template<class T>
    static constexpr bool INJECTION() {
        return et::tree::evaluator<std::decay_t<T>>::requires_greedy_eval;
    }

    //------------------------------------------------Purely lazy evaluation----------------------------------//
    template< class expression, class FullContext>
    static std::enable_if_t<!INJECTION<expression>()>
    evaluate(const expression& expr, FullContext context_) {
        static constexpr int iterator_dimension = expression::ITERATOR;
        impl::template nd_evaluator<iterator_dimension>(expr, context_);
    }
    //------------------------------------------------Purely lazy alias evaluation----------------------------------//
    template< class expression, class FullContext>
    static std::enable_if_t<!INJECTION<expression>()>
    evaluate_aliased(const expression& expr, FullContext context_) {
        static constexpr int iterator_dimension = expression::ITERATOR;
        impl:: nd_evaluator<iterator_dimension>(expr, context_);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected)----------------------------------//
    template< class expression, class FullContext>
    static std::enable_if_t<INJECTION<expression>()>
    evaluate(const expression& expr, FullContext context_) {
        auto greedy_evaluated_expr = et::tree::Greedy_Evaluator::evaluate(expr, context_);

        if (!decay_same<decltype(greedy_evaluated_expr.right), decltype(expr.left)>) {
            static constexpr int iterator_dimension = expression::ITERATOR;
            impl::template nd_evaluator<iterator_dimension>(greedy_evaluated_expr, context_);
        }
        //decontext_ate any temporaries made by the tree
        et::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr, context_);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
    template< class expression, class FullContext>
    static std::enable_if_t<INJECTION<expression>()>
    evaluate_aliased(const expression& expr, FullContext context_) {
        static constexpr int iterator_dimension = expression::ITERATOR;    //the iterator for the evaluation of post inject_t

        auto greedy_evaluated_expr = et::tree::Greedy_Evaluator::evaluate_aliased(expr, context_);        //evaluate the internal tensor_type
        if (!decay_same<decltype(greedy_evaluated_expr.right), decltype(expr.left)>) {
        	impl::template nd_evaluator<iterator_dimension>(greedy_evaluated_expr, context_);
        }
        //decontext_ate any temporaries made by the tree
        et::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr, context_);
    }
};

template<class Context>
struct CacheEvaluator {
    template<class branch> using sub_t  = BC::et::Array<branch::DIMS, BC::scalar_of<branch>, typename Context::allocator_t, BC_Temporary>;
//    template<class branch> using eval_t = BC::et::Binary_Expression<sub_t<branch>, branch, BC::oper::assign>;

    template<class branch> //The branch is an array, no evaluation required
    static std::enable_if_t<BC::is_array<std::decay_t<branch>>(), const branch&>
    evaluate(const branch& expression, Context context_) { return expression; }


    template<class branch> //Create and return an array_core created from the expression
    static std::enable_if_t<!BC::is_array<std::decay_t<branch>>(), sub_t<std::decay_t<branch>>>
    evaluate(const branch& expression, Context context_)
    {
        auto cached_branch = make_tensor_array(
        		expression.inner_shape(),			//context 'get_allocator' returns a referenec not a copy
        		BC::allocator_traits<typename Context::allocator_t>::select_on_temporary_construction(context_.get_allocator()));

        auto expr = make_bin_expr<BC::oper::assign>(cached_branch.internal(), expression);
        Lazy_Evaluator<Context>::evaluate(expr, context_);
        return cached_branch;
    }

};

template<class array_t, class expression_t, class Context>
auto evaluate_to(array_t array, expression_t expr, Context context) {
    static_assert(is_array<array_t>(), "MAY ONLY EVALUATE TO ARRAYS");
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate(
    		et::make_bin_expr<oper::assign>(array, expr), context);
}

template<class expression_t, class Context>
auto evaluate(expression_t expr, Context context) {
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate(expr, context);
}
template<class expression_t, class Context>
auto evaluate_aliased(expression_t expr, Context context) {
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate_aliased(expr, context);
}


}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
