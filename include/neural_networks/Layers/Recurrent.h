/*
 * Recurrent.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef RECURRENT_FEEDFORWARD_CU_
#define RECURRENT_FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
struct Recurrent : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;

    using forward_requires_outputs = std::true_type;
    using backward_requires_outputs = std::true_type;
    using greedy_evaluate_delta = std::true_type;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
	using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:

    ValueType lr = 0.03;

    mat dc; //delta cell_state
    mat w, w_gradients;  //weights
    mat r, r_gradients;
    vec b, b_gradients;  //biases

public:

    Recurrent(int inputs, int outputs) :
        Layer_Base(inputs, outputs),
		w(outputs, inputs),
		w_gradients(outputs, inputs),
		r(outputs, outputs),
		r_gradients(outputs, outputs),
		b(outputs),
		b_gradients(outputs)
	{
        w.randomize(-2, 2);
        b.randomize(-2, 2);
    }

    template<class Matrix>
    auto forward_propagation(const Matrix& x) {
    	return w * x + b;
    }

    template<class X, class Delta>
    auto back_propagation(const X& x, const Delta& dy) {
    	dc = dy;
    	w_gradients -= lr * dc  * x.t();
    	b_gradients -= lr * dc;
    	return w.t() * dc;
    }

    template<class X, class Y>
    auto forward_propagation(const X& x, const Y& y) {
    	return w * x + r * y + b;
    }

    template<class X, class Y, class Delta>
    auto back_propagation(const X& x, const Y& y, const Delta& dy) {
    	dc = dy + r.t() * dc;
    	w_gradients -= lr * dc  * x.t();
    	b_gradients -= lr * dc;
    	r_gradients -= lr * dc * y.t();
    	return w.t() * dc;
    }

    void update_weights() {
    	w += w_gradients / this->batch_size();
    	b += b_gradients / this->batch_size();
    	r += r_gradients / this->batch_size();

    	w_gradients.zero();
    	b_gradients.zero();
    	r_gradients.zero();
    	dc.zero();
    }

    void set_batch_size(BC::size_t bs) {
    	Layer_Base::set_batch_size(bs);
    	dc = mat(this->output_size(), bs);
    }
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
Recurrent<SystemTag, ValueType> recurrent(SystemTag system_tag, int inputs, int outputs) {
	return Recurrent<SystemTag, ValueType>(inputs, outputs);
}
#endif

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
