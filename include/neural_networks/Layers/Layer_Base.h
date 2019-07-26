 /*
 * Layer.h
 *
 *  Created on: Mar 1, 2018
 *      Author: joseph
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "Layer_Traits.h"

namespace BC {
namespace nn {

class Layer_Base {

    const BC::size_t m_input_sz;
    const BC::size_t m_output_sz;
    BC::size_t batch_sz;

public:

    Layer_Base(int inputs, BC::size_t outputs):
    	m_input_sz(inputs),
        m_output_sz(outputs),
        batch_sz(1) {}

    BC::size_t  input_size() const { return m_input_sz; }
	BC::size_t  output_size() const { return m_output_sz; }
    BC::size_t  batch_size()   const { return batch_sz; }

	BC::size_t  batched_input_size() const { return m_input_sz * batch_sz; }
	BC::size_t  batched_output_size() const { return m_output_sz * batch_sz; }

    void set_batch_size(int bs) {
        batch_sz = bs;
    }

    void update_weights() {}
};

}
}



#endif /* LAYER_H_ */
