#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "Tree_Evaluator_Runner_Impl.h"

namespace BC {
namespace et  {

template<class allocator_type>
struct Lazy_Evaluator {

	template<class lv, class rv>
	static constexpr bool decay_same = std::is_same<std::decay_t<lv>, std::decay_t<rv>>::value;

	using impl = typename evaluator::template implementation<allocator_type>;

    template<class T> __BChot__
    static constexpr bool INJECTION() {
        return et::tree::evaluator<std::decay_t<T>>::requires_greedy_eval;
    }

    //------------------------------------------------Purely lazy evaluation----------------------------------//
    template< class expression>
    static std::enable_if_t<!INJECTION<expression>()>
    evaluate(const expression& expr) {
        static constexpr int iterator_dimension = expression::ITERATOR();
        impl::template nd_evaluator<iterator_dimension>(expr);
    }
    //------------------------------------------------Purely lazy alias evaluation----------------------------------//
    template< class expression>
    static std::enable_if_t<!INJECTION<expression>()>
    evaluate_aliased(const expression& expr) {
        static constexpr int iterator_dimension = expression::ITERATOR();
        impl:: nd_evaluator<iterator_dimension>(expr);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected)----------------------------------//
    template< class expression>
    static std::enable_if_t<INJECTION<expression>()>
    evaluate(const expression& expr) {
        auto greedy_evaluated_expr = et::tree::Greedy_Evaluator::evaluate(expr);

        if (!decay_same<decltype(greedy_evaluated_expr.right), decltype(expr.left)>) {
            static constexpr int iterator_dimension = expression::ITERATOR();
            impl::template nd_evaluator<iterator_dimension>(greedy_evaluated_expr);
        }
        //deallocate any temporaries made by the tree
        et::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
    template< class expression>
    static std::enable_if_t<INJECTION<expression>()>
    evaluate_aliased(const expression& expr) {
        static constexpr int iterator_dimension = expression::ITERATOR();    //the iterator for the evaluation of post inject_t

        auto greedy_evaluated_expr = et::tree::Greedy_Evaluator::evaluate_aliased(expr);        //evaluate the internal tensor_type
        if (!decay_same<decltype(greedy_evaluated_expr.right), decltype(expr.left)>) {
        	impl:: nd_evaluator<iterator_dimension>(greedy_evaluated_expr);
        }
        //deallocate any temporaries made by the tree
        et::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr);
    }
};

template<class allocator>
struct CacheEvaluator {
    template<class branch> using sub_t     = BC::et::Array<branch::DIMS(), BC::et::scalar_of<branch>, allocator>;
    template<class branch> using eval_t = BC::et::Binary_Expression<sub_t<branch>, branch, BC::et::oper::assign>;

    template<class branch>__BChot__ //The branch is an array, no evaluation required
    static std::enable_if_t<BC::et::is_array<std::decay_t<branch>>(), const branch&> evaluate(const branch& expression) { return expression; }


    template<class branch>__BChot__ //Create and return an array_core created from the expression
    static std::enable_if_t<!BC::et::is_array<std::decay_t<branch>>(), sub_t<std::decay_t<branch>>> evaluate(const branch& expression)
    {
        sub_t<std::decay_t<branch>> cached_branch(expression.inner_shape());
        eval_t<std::decay_t<branch>> assign_to_expression(cached_branch, expression);
        Lazy_Evaluator<allocator>::evaluate(assign_to_expression);
        return cached_branch;
    }

};

template<class array_t, class expression_t>__BChot__
auto evaluate_to(array_t array, expression_t expr) {
    static_assert(is_array<array_t>(), "MAY ONLY EVALUATE TO ARRAYS");
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate(et::Binary_Expression<array_t, expression_t, et::oper::assign>(array, expr));
}

template<class expression_t>__BChot__
auto evaluate(expression_t expr) {
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate(expr);
}


}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
