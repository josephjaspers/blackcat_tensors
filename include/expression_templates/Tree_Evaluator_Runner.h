#ifndef TRIVIAL_EVALUATOR_H_
#define TRIVIAL_EVALUATOR_H_

#include "Tree_Evaluator_Runner_Impl.h"

namespace BC {
namespace et {

template<class allocator_type>
struct Lazy_Evaluator {

	template<class lv, class rv>
	static constexpr bool decay_same = std::is_same<std::decay_t<lv>, std::decay_t<rv>>::value;

	using impl = typename evaluator::template implementation<allocator_type>;

    template<class T>
    static constexpr bool INJECTION() {
        return et::tree::evaluator<std::decay_t<T>>::requires_greedy_eval;
    }

    //------------------------------------------------Purely lazy evaluation----------------------------------//
    template< class expression, class Allocator>
    static std::enable_if_t<!INJECTION<expression>()>
    evaluate(const expression& expr, Allocator& alloc) {
        static constexpr int iterator_dimension = expression::ITERATOR;
        impl::template nd_evaluator<iterator_dimension>(expr);
    }
    //------------------------------------------------Purely lazy alias evaluation----------------------------------//
    template< class expression, class Allocator>
    static std::enable_if_t<!INJECTION<expression>()>
    evaluate_aliased(const expression& expr, Allocator& alloc) {
        static constexpr int iterator_dimension = expression::ITERATOR;
        impl:: nd_evaluator<iterator_dimension>(expr);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected)----------------------------------//
    template< class expression, class Allocator>
    static std::enable_if_t<INJECTION<expression>()>
    evaluate(const expression& expr, Allocator& alloc) {
        auto greedy_evaluated_expr = et::tree::Greedy_Evaluator::evaluate(expr, alloc);

        if (!decay_same<decltype(greedy_evaluated_expr.right), decltype(expr.left)>) {
            static constexpr int iterator_dimension = expression::ITERATOR;
            impl::template nd_evaluator<iterator_dimension>(greedy_evaluated_expr);
        }
        //deallocate any temporaries made by the tree
        et::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr, alloc);
    }
    //------------------------------------------------Greedy evaluation (BLAS function call detected), skip injection optimization--------------------//
    template< class expression, class Allocator>
    static std::enable_if_t<INJECTION<expression>()>
    evaluate_aliased(const expression& expr, Allocator& alloc) {
        static constexpr int iterator_dimension = expression::ITERATOR;    //the iterator for the evaluation of post inject_t

        auto greedy_evaluated_expr = et::tree::Greedy_Evaluator::evaluate_aliased(expr, alloc);        //evaluate the internal tensor_type
        if (!decay_same<decltype(greedy_evaluated_expr.right), decltype(expr.left)>) {
        	impl:: nd_evaluator<iterator_dimension>(greedy_evaluated_expr);
        }
        //deallocate any temporaries made by the tree
        et::tree::Greedy_Evaluator::deallocate_temporaries(greedy_evaluated_expr, alloc);
    }
};

template<class allocator>
struct CacheEvaluator {
    template<class branch> using sub_t  = BC::et::Array<branch::DIMS, BC::scalar_of<branch>, allocator, BC_Temporary>;
    template<class branch> using eval_t = BC::et::Binary_Expression<sub_t<branch>, branch, BC::et::oper::assign>;

    template<class branch> //The branch is an array, no evaluation required
    static std::enable_if_t<BC::is_array<std::decay_t<branch>>(), const branch&>
    evaluate(const branch& expression, allocator& alloc) { return expression; }


    template<class branch> //Create and return an array_core created from the expression
    static std::enable_if_t<!BC::is_array<std::decay_t<branch>>(), sub_t<std::decay_t<branch>>>
    evaluate(const branch& expression, allocator& alloc)
    {
        sub_t<std::decay_t<branch>> cached_branch(
        		expression.inner_shape(),
        		BC::allocator_traits<allocator>::select_on_temporary_construction(alloc));
        eval_t<std::decay_t<branch>> assign_to_expression(cached_branch.internal(), expression);
        Lazy_Evaluator<allocator>::evaluate(assign_to_expression);
        return cached_branch;
    }

};

template<class array_t, class expression_t, class allocator>
auto evaluate_to(array_t array, expression_t expr, allocator& alloc) {
    static_assert(is_array<array_t>(), "MAY ONLY EVALUATE TO ARRAYS");
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate(
    		et::make_bin_expr<et::oper::assign>(array, expr),
    		alloc);
}

template<class expression_t, class allocator>
auto evaluate(expression_t expr, allocator& alloc) {
    return Lazy_Evaluator<typename expression_t::system_tag>::evaluate(expr, alloc);
}


}
}


#endif /* TRIVIAL_EVALUATOR_H_ */
