/*
 * GRU.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef GRU_FEEDFORWARD_CU_
#define GRU_FEEDFORWARD_CU_

#include "Layer_Base.h"

namespace BC {
namespace nn {
namespace not_implemented {

template<class SystemTag, class ValueType, class GRUNonLinearity=BC::Tanh>
struct GRU : public Layer_Base<GRU<SystemTag, ValueType, GRUNonLinearity>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<GRU<SystemTag, ValueType, GRUNonLinearity>>;

	using forward_requires_outputs = std::true_type;
	using backward_requires_outputs = std::true_type;
	using greedy_evaluate_delta = std::true_type;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
	using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

	GRUNonLinearity g;
	ValueType lr = 0.03;

	mat dc; //delta cell_state
	mat c;

	mat w, w_gradients;  //weights
	mat r, r_gradients;
	vec b, b_gradients;  //biases

	mat wf, wf_gradients;  //weights
	mat rf, rf_gradients;
	vec bf, bf_gradients;  //biases


	GRU(int inputs, int outputs) :
		parent_type(inputs, outputs),
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
	}

	template<class X>
	auto forward_propagation(const X& x) {
		auto f = BC::logistic(wf * x + bf);
		c %= f;
		return c +=  w * x + b;
	}

	template<class X, class Y>
	auto forward_propagation(const X& x, const Y& y) {
		auto f = BC::logistic(wf * x + rf * g(y) + bf);
		c %= f;
		return c += w * x + r * g(y) + b;
	}

	template<class X, class Y, class Delta>
	auto back_propagation(const X& x, const Y& y, const Delta& dy) {
		auto f = BC::logistic(wf * x + rf * g(y) + bf);

		r_gradients -= dc * g.dx(y).t();

		dc.alias() = dy + r.t() * dc + rf.t() * dy;
		w_gradients -= dy  * x.t();
		b_gradients -= dy;
		return w.t() * dy;
	}

	void update_weights() {
		value_type lr = this->get_batched_learning_rate();

		w += w_gradients * lr;
		b += b_gradients * lr;
		r += r_gradients * lr;

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
GRU<SystemTag, ValueType> gru(SystemTag system_tag, int inputs, int outputs) {
	return GRU<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto gru(SystemTag system_tag, int inputs, int outputs) {
	return GRU<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto gru(int inputs, int outputs) {
	return GRU<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}

}
}
}

#endif /* FEEDFORWARD_CU_ */
