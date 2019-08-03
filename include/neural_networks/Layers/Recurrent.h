/*
 * Recurrent.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef RECURRENT_FEEDFORWARD_CU_
#define RECURRENT_FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
struct Recurrent : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;

    using forward_requires_outputs = std::true_type;
    using backward_requires_outputs = std::true_type;
    using greedy_evaluate_delta = std::true_type;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
	using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:

    ValueType lr = 0.0003;

    mat w;  //weights
    mat r;
    vec b;  //biases

public:

    Recurrent(int inputs, int outputs) :
        Layer_Base(inputs, outputs),
		w(outputs, inputs),
		r(outputs, outputs),
		b(outputs)
    {
        w.randomize(-2, 2);
        b.randomize(-2, 2);
    }

    template<class Matrix>
    auto forward_propagation(const Matrix& x) {
    	return w * x + b;
    }

    template<class X, class Delta>
    auto back_propagation(const X& x, const Delta& dy) {
    	w -= lr * dy  * x.t();
    	b -= lr * dy;
    	return w.t() * dy;
    }

    template<class X, class Y>
    auto forward_propagation(const X& x, const Y& y) {
    	return w * x + r * y + b;
    }

    template<class X, class Y, class Delta>
    auto back_propagation(const X& x, const Y& y, const Delta& dy) {
    	w -= lr * dy  * x.t();
    	b -= lr * dy;
    	r -= lr * dy * y.t();
    	return w.t() * dy;
    }
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
Recurrent<SystemTag, ValueType> recurrent(SystemTag system_tag, int inputs, int outputs) {
	return Recurrent<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto recurrent(SystemTag system_tag, int inputs, int outputs) {
	return Recurrent<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto recurrent(int inputs, int outputs) {
	return Recurrent<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
