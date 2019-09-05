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
namespace tensors {
namespace exprs {
namespace evaluator {

template<int Dimension>
struct evaluator_impl {
    template<class Expression, class... indexes>
    static void impl(Expression& expression, indexes... indicies) {
        BC_omp_for__
        for (int i = 0; i < expression.dimension(Dimension-1); ++i) {
        	evaluator_impl<Dimension-1>::impl(expression, indicies..., i);
        }
    }
    template<class Expression, class... indexes>
    static void endpoint_call(Expression expression, indexes... indicies) {
        BC_omp_for__
        for (int i = 0; i < expression.dimension(Dimension-1); ++i) {
        	evaluator_impl<Dimension-1>::impl(expression, indicies..., i);
        }
    }
};
template<>
struct evaluator_impl<1> {
    template<class Expression, class... indexes>
    static void impl(Expression& expression, indexes... indicies) {
        BC_omp_for__
        for (int i = 0; i < expression.dimension(0); ++i) {
            expression(indicies..., i);
        }
    }
    template<class Expression>
    static void impl(Expression& expression) {
        BC_omp_for__
        for (int i = 0; i < expression.size(); ++i) {
            expression[i];
        }
    }
    template<class Expression, class... indexes>
    static void endpoint_call(Expression expression, indexes... indicies) {
        BC_omp_for__
        for (int i = 0; i < expression.dimension(0); ++i) {
            expression(indicies..., i);
        }
    }
    template<class Expression>
    static void endpoint_call(Expression expression) {
        BC_omp_for__
        for (int i = 0; i < expression.size(); ++i) {
            expression[i];
        }
    }
};
template<>
struct evaluator_impl<0> {
    template<class Expression>
    static void impl(Expression& expression) {
        BC_omp_for__
        for (int i = 0; i < expression.size(); ++i) {
            expression[i];
        }
    }
    template<class Expression>
    static void endpoint_call(Expression expression) {
        BC_omp_for__
        for (int i = 0; i < expression.size(); ++i) {
            expression[i];
        }
    }
};


template<>
struct Evaluator<host_tag> {

	template<int Dimension, class Expression, class Stream>
	static void nd_evaluate(Expression expression, Stream stream) {
		auto job = [=]() {
			evaluator_impl<Dimension>::endpoint_call(expression);
		};
		stream.enqueue(job);
	}


};

}
}
}
}
#endif
 /* MATHEMATICS_CPU_H_ */
