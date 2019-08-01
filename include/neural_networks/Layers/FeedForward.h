/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef FEEDFORWARD_CU_
#define FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
struct FeedForward : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;
    using greedy_evaluate_delta = std::true_type;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
	using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:

    ValueType lr = 0.003;

    mat w;  //weights
    vec b;  //biases

public:

    FeedForward(int inputs, BC::size_t  outputs) :
        Layer_Base(inputs, outputs),
		w(outputs, inputs),
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
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
FeedForward<SystemTag, ValueType> feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto feedforward(int inputs, int outputs) {
	return FeedForward<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
