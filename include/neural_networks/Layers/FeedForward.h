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
class FeedForward : public Layer_Base {

public:

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
	using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;
    using mat_view = BC::Matrix_View<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:

    ValueType lr = 0.003;

    mat dy;          //error
    mat y;           //outputs
    mat_view x;             //inputs

    mat w;                  //weights
    vec b;                  //biases

public:

    FeedForward(int inputs, BC::size_t  outputs) :
        Layer_Base(inputs, outputs),
		w(outputs, inputs),
		b(outputs)
    {
        w.randomize(-2, 2);
        b.randomize(-2, 2);
    }

    template<class Expression>
    const auto& forward_propagation(const BC::MatrixXpr<Expression>& x_) {
    	x = mat_view(x_);
        y = w * x + b;
        return y;
    }


    template<class Expression>
    auto back_propagation(const BC::MatrixXpr<Expression>& dy_) {
    	dy = dy_;
        return w.t() * dy;
    }
    /*
     * y =  w * x;
     *
     * derivative of w == dy * x.t
     * derivative of x == w.t * dy
     *
     *
     */

    void update_weights() {
    	auto rate = lr / this->batch_size();
    	w -= dy * rate * x.t();
        b -= dy * rate;
    }

    void set_batch_size(int x) {
        y = mat(this->numb_outputs(), x);
        dy = mat(this->numb_outputs(), x);
    }
};

template<class ValueType, class SystemTag>
FeedForward<SystemTag, ValueType> feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, ValueType>(inputs, outputs);
}
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
