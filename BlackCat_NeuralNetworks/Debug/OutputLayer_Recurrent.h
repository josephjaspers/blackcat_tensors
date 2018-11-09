/*
 * OutputLayer_Recurrent.h
 *
 *  Created on: Aug 26, 2018
 *      Author: joseph
 */

#ifndef OUTPUTLAYER_RECURRENT_H_
#define OUTPUTLAYER_RECURRENT_H_

#include "Layer_Base_Recurrent.h"

namespace BC {
namespace NN {

template<class derived>
struct OutputLayer_Recurrent : Layer_Base_Recurrent<derived> {

    vec zero = vec(this->OUTPUTS);

public:

    OutputLayer(int inputs) : Layer_Base<derived>(inputs) {}

    template<class t> auto forward_propagation(const expr::mat<t>& x) {
        return x;
    }
    template<class t> auto forward_propagation_express(const expr::mat<t>& x) {
        return x;
    }
    template<class t> auto back_propagation(const expr::mat<t>& exp) {
        return this->prev().back_propagation(this->prev().y[this->curr_timestamp] - exp);
    }

    void set_batch_size(int) {}
    void update_weights() {}
    void clear_stored_delta_gradients() {}
    void write(std::ofstream& is) {}
    void read(std::ifstream& os) {}
    void set_learning_rate(fp_type learning_rate) {}

};

}
}
