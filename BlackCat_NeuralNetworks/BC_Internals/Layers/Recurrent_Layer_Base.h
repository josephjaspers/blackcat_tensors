/*
 * Layer_Base_Recurrent.h
 *
 *  Created on: Sep 25, 2018
 *      Author: joseph
 */

#ifndef LAYER_BASE_RECURRENT_H_
#define LAYER_BASE_RECURRENT_H_

#include "Utility.h"
#include "Layer_Base.h"

namespace BC {
namespace NN {

struct Recurrent_Layer_Base : Layer_Base{

    static constexpr bool is_recurrent = true;

    circular_int t = 1; //timestamp
    int max_backprop_length = 1;

    Recurrent_Layer_Base(int x, int y) : Layer_Base(x, y) {}

    void set_max_bptt_length(int x) {
        max_backprop_length = x;
    }
    void increment_timestamp() {
        t++;
    }
    void decrement_timestamp() {
        t--;
    }
    int get_timestamp() {
        return t;
    }
    int get_max_bptt_length() {
        return max_backprop_length;
    }
};
}
}




#endif /* LAYER_BASE_RECURRENT_H_ */
