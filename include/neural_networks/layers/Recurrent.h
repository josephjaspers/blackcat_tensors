/*
 * Recurrent.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef RECURRENT_FEEDFORWARD_CU_
#define RECURRENT_FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType, class RecurrentNonLinearity=BC::Tanh>
struct Recurrent:
		public Layer_Base<
				Recurrent<SystemTag, ValueType, RecurrentNonLinearity>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;
	using parent_type = Layer_Base<
			Recurrent<SystemTag, ValueType, RecurrentNonLinearity>>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using forward_requires_outputs = std::true_type;
	using backward_requires_outputs = std::true_type;
	using greedy_evaluate_delta = std::true_type;

	RecurrentNonLinearity g;
	ValueType lr = parent_type::default_learning_rate;

	mat dc; //delta cell_state
	mat w, w_gradients;  //weights
	mat r, r_gradients;
	vec b, b_gradients;  //biases

	Recurrent(int inputs, int outputs) :
		parent_type(__func__, inputs, outputs),
		w(outputs, inputs),
		w_gradients(outputs, inputs),
		r(outputs, outputs),
		r_gradients(outputs, outputs),
		b(outputs),
		b_gradients(outputs)
	{
		w.randomize(-2, 2);
		b.randomize(-2, 2);
		r.randomize(-2, 2);
		zero_gradients();
	}

	template<class X>
	auto forward_propagation(const X& x) {
		return w * x + b;
	}

	template<class X, class Y>
	auto forward_propagation(const X& x, const Y& y) {
		return w * x + r * g(y) + b;
	}

	template<class X, class Y, class Delta>
	auto back_propagation(const X& x, const Y& y, const Delta& dy) {
		r_gradients -= dc * g.dx(y).t();

		dc.alias() = dy + r.t() * dc;
		w_gradients -= dy  * x.t();
		b_gradients -= dy;
		return w.t() * dy;
	}

	void update_weights() {
		auto lr = this->lr / this->batch_size();

		w += w_gradients * lr;
		b += b_gradients * lr;
		r += r_gradients * lr;

		zero_deltas();
		zero_gradients();
	}

	void set_batch_size(BC::size_t bs) {
		parent_type::set_batch_size(bs);
		dc = mat(this->output_size(), bs);
		zero_deltas();
	}

	void zero_deltas() {
		dc.zero();
	}
	void zero_gradients() {
		w_gradients.zero();
		b_gradients.zero();
		r_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(w, "w");
		loader.save_variable(r, "r");
		loader.save_variable(b, "b");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(w, "w");
		loader.load_variable(r, "r");
		loader.load_variable(b, "b");
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
