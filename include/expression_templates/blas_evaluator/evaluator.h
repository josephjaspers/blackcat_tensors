/*
 * evaluator.h
 *
 *  Created on: Nov 28, 2018
 *      Author: joseph
 */

#ifndef EVALU2ATOR_H_
#define EVALU2ATOR_H_

namespace BC{
namespace et {
namespace evaluator {

#define BC_ET_INNER_FUNC(funcname) 															\
struct funcname##_t { int operator () (); };	\
 int funcname::operator () () \


template<class> struct scalar_modifer {
    enum mod {
        alpha = 0,
        beta = 0,
    };
};
template<> struct scalar_modifer<et::oper::add> {
    enum mod {
        alpha = 1,
        beta = 0,
    };
};
template<> struct scalar_modifer<et::oper::sub> {
    enum mod {
        alpha = -1,
        beta = 0
    };
};
template<> struct scalar_modifer<et::oper::add_assign> {
    enum mod {
        alpha = 1,
        beta = 1,
    };
};
template<> struct scalar_modifer<et::oper::sub_assign> {
    enum mod {
        alpha = -1,
        beta = 1,
    };
};
template<> struct scalar_modifer<et::oper::assign> {
    enum mod {
        alpha = 1,
        beta = 0,
    };
};


template<class T> static constexpr bool is_blas_func() {
    return std::is_base_of<BC::BLAS_FUNCTION, T>::value;
}

template<class T>
static constexpr bool is_linear_op() {
    return MTF::seq_contains<T, et::oper::add, et::oper::sub>;
}

template<class T>
static constexpr bool is_linear_assignment_op() {
    return MTF::seq_contains<T, et::oper::add_assign, et::oper::sub_assign>;
}



template<class T>
static constexpr bool is_nonlinear_op() {
    return  !MTF::seq_contains<T, et::oper::add, et::oper::sub> && !is_blas_func<T>();
}
template<class T>
static constexpr int alpha_of() {
    return scalar_modifer<std::decay_t<T>>::mod::beta;
}
template<class T>
static constexpr int beta_of() {
    return scalar_modifer<std::decay_t<T>>::mod::alpha;
}



template<class output, int alpha, int beta, class... scalars> class OutputWrapper;

template<class output, int a, int b, class... scalars>
auto make_wrapper(output o, scalars... s) {
	return OutputWrapper<output, a, b, scalars...>(o, s...);
}


	struct evaluator {

		template<class expression_t, class output_t>
		static void impl(expression_t expr, const OutputWrapper& output) {

			if (is_linear_op<expression_t>()) {
				if (is_scalar_mul<expression_t>()) {
					auto output = make_wrapper(output, expr.scalar);
					return evaluator(expr.branch, output);

				} else if (is_blas_func<decltype(expr.left)>() && is_blas_func<decltype(expr.right)>()) {
					expr.left.evaluate(output);
					expr.right.evaluate(update_injection(output));
					return output;
				} else if (is_blas_func<decltype(expr.left)>()) {
					expr.right.evaluate(output);
//					return bin_func(exp)
				}

			}


		}

	};





}
}
}



#endif /* EVALUATOR_H_ */
