/*
 * RecurrentFeedForward.h
 *
 *  Created on: Jul 17, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORKS_RECURRENTFEEDFORWARD_H_
#define BLACKCAT_NEURALNETWORKS_RECURRENTFEEDFORWARD_H_

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
class FeedForward : public Layer_Base {

public:

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
    using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;
    using cube = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

    using mat_view = BC::Matrix_View<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:

    ValueType lr = 0.003;

    BC::size_t curr_timestamp;


    mat dy; //error
    mat y;  //outputs
    cube x; //inputs

    mat w;      //weights
    vec b;      //biases

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
    const auto& forward_propagation(const Matrix& x_) {
    	x = mat_view(x_);
        y = w * x + b;
        return y;
    }
    template<class Matrix>
    auto back_propagation(const Matrix& dy_) {
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

#endif /* RECURRENTFEEDFORWARD_H_ */
