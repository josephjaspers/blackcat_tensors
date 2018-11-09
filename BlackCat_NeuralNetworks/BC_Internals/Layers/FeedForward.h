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
namespace NN {

struct FeedForward : public Layer_Base {

    using Layer_Base::lr;    //the learning rate

    mat dy;          //error
    mat y;           //outputs
    mat_view x;             //inputs

    mat w;                  //weights
    vec b;                  //biases


    FeedForward(int inputs, int outputs) :
        Layer_Base(inputs, outputs),
            w(outputs, inputs),
            b(outputs)
    {
        w.randomize(-1, 1);
        b.randomize(-1, 1);
    }
    template<class t>
    const auto& forward_propagation(const expr::mat<t>& x_) {
        x = mat_view(x_);
        return y = g(w * x + b);
    }
    template<class t>
    auto back_propagation(const expr::mat<t>& dy_) {
        dy = dy_;
        return w.t() * dy % gd(x);
    }
    void update_weights() {
        w -= dy * lr * x.t();
        b -= dy * lr;
    }

    void set_batch_size(int x) {
        y = mat(this->numb_outputs(), x);
        dy = mat(this->numb_outputs(), x);
    }

    auto& inputs()  { return x; }
    auto& outputs() { return y; }
    auto& deltas()  { return dy;}
    auto& weights()    { return w; }
    auto& bias()    { return b; }


};
}
}

#endif /* FEEDFORWARD_CU_ */
