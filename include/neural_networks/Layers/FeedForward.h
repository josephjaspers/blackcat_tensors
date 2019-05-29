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

    ValueType lr = 0.03;

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
        w.randomize(-1, 1);
        b.randomize(-1, 1);
    }
    template<class Matrix>
    const auto& forward_propagation(const Matrix& x_) {
    	x = mat_view(x_);
        y = BC::logistic(w * x + b);
        return y;
    }
    template<class Matrix>
    auto back_propagation(const Matrix& dy_) {
    	dy = dy_;
        return w.t() * dy % BC::cached_dx_logistic(x);
    }
    void update_weights() {
    	w -= dy * lr * x.t();
        b -= dy * lr;
    }

    void set_batch_size(int x) {
        y = mat(this->numb_outputs(), x);
        dy = mat(this->numb_outputs(), x);
    }
};
}
}

#endif /* FEEDFORWARD_CU_ */
