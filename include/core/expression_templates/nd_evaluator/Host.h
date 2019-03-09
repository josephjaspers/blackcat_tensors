/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef MATHEMATICS_CPU_H_
#define MATHEMATICS_CPU_H_

#include <future>

namespace BC {
namespace evaluator {

#define BC_OPENMP_REDUCTION_FUNCTION(oper, name, base_value)						\
template<class Expression, class... indexes> static									\
typename Expression::value_type name(Expression value, indexes... indicies) {		\
	using value_type = typename Expression::value_type;								\
																					\
	value_type total = base_value;													\
	__BC_omp_for_reduction__(oper, name)											\
	for (BC::size_t i = 0; i < value.dimension(Dimension-1); ++i) {					\
		total oper##= evaluator_impl<Dimension-1>::name(value, i, indicies...);		\
	}																				\
	return total;																	\
}
#define BC_OPENMP_REDUCTION_BASE_CASE_FUNCTION(oper, name, base_value)				\
template<class Expression, class... indexes> static									\
typename Expression::value_type name(Expression value, indexes... indicies) {		\
	using value_type = typename Expression::value_type;								\
																					\
	value_type total = base_value;													\
	__BC_omp_for_reduction__(oper, name)											\
	for (BC::size_t i = 0; i < value.dimension(0); ++i) {							\
		total oper##= value(i, indicies...);										\
	}																				\
	return total;																	\
}
#define BC_OPENMP_REDUCTION_ITERATOR0_CASE_FUNCTION(oper, name, base_value)			\
template<class Expression> static													\
typename Expression::value_type name(Expression value) {							\
	using value_type = typename Expression::value_type;								\
																					\
	value_type total = base_value;													\
	__BC_omp_for_reduction__(oper, name)											\
	for (BC::size_t i = 0; i < value.size; ++i) {									\
		total oper##= value[i];														\
	}																				\
	return total;																	\
}

template<int Dimension>
struct evaluator_impl {
    template<class Expression, class... indexes>
    static void impl(Expression expression, indexes... indicies) {
        __BC_omp_for__
        for (int i = 0; i < expression.dimension(Dimension-1); ++i) {
        	evaluator_impl<Dimension-1>::impl(expression, i, indicies...);
        }
    }
    BC_OPENMP_REDUCTION_FUNCTION(+, sum, 0)
    BC_OPENMP_REDUCTION_FUNCTION(*, prod, 1)

};
template<>
struct evaluator_impl<1> {
    template<class Expression, class... indexes>
    static void impl(Expression expression, indexes... indicies) {
        __BC_omp_for__
        for (int i = 0; i < expression.dimension(0); ++i) {
            expression(i, indicies...);
        }
    }
    template<class Expression>
    static void impl(Expression expression) {
        __BC_omp_for__
        for (int i = 0; i < expression.size(); ++i) {
            expression[i];
        }
    }

    BC_OPENMP_REDUCTION_BASE_CASE_FUNCTION(+, sum, 0)
    BC_OPENMP_REDUCTION_BASE_CASE_FUNCTION(*, prod, 0)

    BC_OPENMP_REDUCTION_ITERATOR0_CASE_FUNCTION(+, sum, 0)
    BC_OPENMP_REDUCTION_ITERATOR0_CASE_FUNCTION(*, prod, 0)

};
template<>
struct evaluator_impl<0> {
    template<class Expression>
    static void impl(Expression expression) {
        __BC_omp_for__
        for (int i = 0; i < expression.size(); ++i) {
            expression[i];
        }
    }


    BC_OPENMP_REDUCTION_ITERATOR0_CASE_FUNCTION(+, sum, 0)
    BC_OPENMP_REDUCTION_ITERATOR0_CASE_FUNCTION(*, prod, 0)
};


struct Host {

	template<int Dimension, class Expression, class Context>
	static void nd_evaluator(Expression expression, Context context) {
		auto job = [=]() {
			evaluator_impl<Dimension>::impl(expression);
		};
		context.get_stream().push_job(job);
	}


	//!!!!ScalarOutput should be a TensorBase type
	template<int Dimension, class ScalarOutput, class Expression, class Context>
	static std::future<ScalarOutput> reduce_sum(ScalarOutput scalar, Expression expression, Context context) {
		std::promise<ScalarOutput>* promise = new std::promise<ScalarOutput>();

		auto job = [=]() {
			scalar.internal()[0] = evaluator_impl<Dimension>::sum(expression);
			promise->set_value(scalar);
			delete promise;
		};

		context.get_stream().push_job(job);
		return promise->get_future();
	}
	template<int Dimension, class ScalarOutput, class Expression, class Context>
	static std::future<ScalarOutput> reduce_prod(ScalarOutput scalar, Expression expression, Context context) {
		std::promise<ScalarOutput>* promise = new std::promise<ScalarOutput>();

		auto job = [=]() {
			scalar.internal()[0] = evaluator_impl<Dimension>::sum(expression);
			promise->set_value(scalar);
			delete promise;
		};
		context.get_stream().push_job(job);
		return promise->get_future();
	}
	template<int Dimension, class ScalarOutput, class Expression, class Context>
	static std::future<ScalarOutput> reduce_mean(ScalarOutput scalar, Expression expression, Context context) {
		std::promise<ScalarOutput>* promise = new std::promise<ScalarOutput>();

		auto job = [=]() {
			scalar.internal()[0] = evaluator_impl<Dimension>::sum(expression) / expression.size();
			promise->set_value(scalar);
			delete promise;
		};
		context.get_stream().push_job(job);
		return promise->get_future();
	}

};
}
}
#endif
 /* MATHEMATICS_CPU_H_ */
