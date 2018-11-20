/*
 * Recurrent_FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef Recurrent_FeedForward_CU_
#define Recurrent_FeedForward_CU_

#include "Recurrent_Layer_Base.h"

namespace BC {
namespace NN {

struct Recurrent_FeedForward : public Recurrent_Layer_Base {

    using Recurrent_Layer_Base::lr;    //the learning rate
    using Recurrent_Layer_Base::t;

    mat dy;          //error
    cube y;           //outputs
    std::vector<mat_view> x = std::vector<mat_view>(this->get_max_bptt_length());             //inputs

    mat w;                  //weights
    vec b;                  //biases
    mat wd;                    //weight delta
    vec bd;                    //bias delta


    Recurrent_FeedForward(int inputs, int outputs) :
        Recurrent_Layer_Base(inputs, outputs),
            w(outputs, inputs),
            b(outputs),
            wd(outputs, inputs),
            bd(outputs)
    {
        w.randomize(-1, 1);
        b.randomize(-1, 1);
    }
    template<class T>
    const auto forward_propagation(const expr::mat<T>& x_) {
        x[t] = mat_view(x_);
         y[t] = g(w * x[t] + b);
         t++;
         return y[t];
    }
    template<class T>
    auto back_propagation(const expr::mat<T>& dy_) {
        dy = dy_;
        return w.t() * dy % gd(x[t]);
    }
    void update_weights() {
        w += wd;
        b += bd;
        wd.zero();
        bd.zero();
    }
    void cache_gradients() {
        wd -= dy * lr * x[t].t();
        bd -= dy * lr;
    }

    void set_batch_size(int batch_sz) {
        y = cube(this->numb_outputs(), batch_sz, this->get_max_bptt_length());
        dy = mat(this->numb_outputs(), batch_sz);
    }

    auto& inputs()  { return x; }
    auto& outputs() { return y; }
    auto& deltas()  { return dy;}
    auto& weights() { return w; }
    auto& bias()    { return b; }


};
}
}

#endif /* Recurrent_FeedForward_CU_ */
