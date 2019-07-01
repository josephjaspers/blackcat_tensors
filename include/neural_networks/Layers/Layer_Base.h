 /*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

//#include "Layer_Utility_Functions.h"
namespace BC {
namespace nn {

static const BC::tensors::exprs::Shape<0> NULL_TENSOR = BC::tensors::exprs::Shape<0>();

//template<class derived>
class Layer_Base {

    static constexpr bool is_recurrent = false;

    const BC::size_t  INPUTS;
    const BC::size_t  OUTPUTS;
    BC::size_t  BATCH_SIZE;
public:
    Layer_Base(int inputs, BC::size_t  outputs)
        : INPUTS(inputs),
          OUTPUTS(outputs),
          BATCH_SIZE(1)
    {}

    BC::size_t  numb_inputs() const { return INPUTS; }
    BC::size_t  numb_outputs() const { return OUTPUTS; }
    BC::size_t  batch_size()   const { return BATCH_SIZE; }

    void set_batch_size(int bs) {
        BATCH_SIZE = bs;
    }
};

}
}



#endif /* LAYER_H_ */
