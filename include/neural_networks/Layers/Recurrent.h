/*
 * Recurrent.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_LAYERS_RECURRENT_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_LAYERS_RECURRENT_H_

#include "Layer_Base.h"
#include <vector>

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
class Recurrent : public Layer_Base {

public:

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
	using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

	using forward_requires_outputs = std::true_type;
	using backwards_requires_outputs = std::true_type;

private:

    ValueType lr = 0.003;

    mat dc;          //cellstate error

    mat w, w_gradients; //weights
    mat r, r_gradients;	//recurrent weights
    vec b, b_gradients; //biases



public:

    Recurrent(int inputs, BC::size_t outputs):
        Layer_Base(inputs, outputs),
		w(outputs, inputs),
		r(outputs, outputs),
		b(outputs),

		w_gradients(outputs, inputs),
		r_gradients(outputs, outputs),
		b_gradients(outputs)
    {
    	r.randomize(-2, 1); //slightly bias recurrent weights to contribute less
        w.randomize(-2, 2);
        b.randomize(-2, 2);

        w_gradients.zero();
        r_gradients.zero();
        b_gradients.zero();
    }

    template<class X, class Y>
    auto forward_propagation(const X& x, const Y& y) {
    	std::cout << " recurrent fp " << std::endl;
    	return w * x + r * y;
    }
    template<class X>
    auto forward_propagation(const X& x) {
    	std::cout << " single recurrent fp " << std::endl;
    	return w * x;
    }
    template<class Allocator, class DeltaY>
    auto back_propagation(const BC::Matrix<value_type, Allocator>& x, const DeltaY& dy) {
    	std::cout << " single recurrent BP MATRIX" << std::endl;
    	x.print_dimensions();
    	dc = dy;
    	std::cout << " 1 " << std::endl;

    	w_gradients -= dc * x.t();
    	std::cout << " 2 " << std::endl;

    	b_gradients -= dc;
    	std::cout << " 3 " << std::endl;

    	(w.t() * dy).print_dimensions();
    	return w.t() * dy;
    }
    template<class Allocator, class DeltaY>
    auto back_propagation(const BC::Vector<value_type, Allocator>& x, const DeltaY& dy) {
    	std::cout << " single recurrent BP VECTOR" << std::endl;
    	std::cout << " set dc[0] = dy" << std::endl;

    	dc[0] = dy;

    	std::cout << " 1 " << std::endl;

    	w_gradients -= dc[0] * x.t();
    	b_gradients -= dc[0];

    	std::cout << " 2 "  << std::endl;

    	return w.t() * dy;
    }

    template<class X, class Y, class DeltaY>
    auto back_propagation(const X& x, const Y& y, const DeltaY& dy) {
    	std::cout << " recurrent BP " << std::endl;

    	dc += dy;
    	w_gradients -= dc * x.t();
    	r_gradients -= dc * y.t();
    	b_gradients -= dc;

    	return w.t() * dy;
    }

    void update_weights() {
    	w += w_gradients * lr; w_gradients.zero();
    	r += r_gradients * lr; r_gradients.zero();
    	b += b_gradients * lr; b_gradients.zero();


    	dc.zero();
    }

    void set_batch_size(int x) {
        dc = mat(this->output_size(), x);
    }
};

template<class ValueType, class SystemTag>
Recurrent<SystemTag, ValueType> recurrent(SystemTag system_tag, int inputs, int outputs) {
	return Recurrent<SystemTag, ValueType>(inputs, outputs);
}
template<class SystemTag>
auto recurrent(SystemTag system_tag, int inputs, int outputs) {
	return Recurrent<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto recurrent(int inputs, int outputs) {
	return Recurrent<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
