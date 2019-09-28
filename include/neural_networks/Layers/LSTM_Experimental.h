/*
 * LSTM.h
 *
 *  Created on: Aug 3, 2019
 *	  Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_EXPERIMENTAL_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_EXPERIMENTAL_H_

namespace BC {
namespace nn {
namespace experimental {

using BC::algorithms::reference_list;

template<class SystemTag,
		class ValueType,
		class ForgetGateNonlinearity=BC::Logistic,
		class WriteGateNonlinearity=BC::Tanh,
		class InputGateNonlinearity=BC::Logistic,
		class OutputGateNonlinearity=BC::Logistic,
		class CellStateNonLinearity=BC::Tanh>
struct LSTM : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;
	using forward_requires_outputs = std::true_type;
	using backward_requires_outputs = std::true_type;


private:

	ValueType lr = Layer_Base::default_learning_rate;

	CellStateNonLinearity  c_g;
	ForgetGateNonlinearity f_g;
	WriteGateNonlinearity z_g;
	InputGateNonlinearity i_g;
	OutputGateNonlinearity o_g;

	//back-propagation index, relative to the size of the number of stored back-prop activations
	unsigned t_minus_index = 0;

	mat w;
	mat w_gradients;

	vec b;
	vec b_gradients;

	mat r;
	mat r_gradients;

	mat dc, dy;
	mat delta;
	mat c;

	std::vector<mat> cs, xs;

public:

	LSTM(int inputs, BC::size_t  outputs):
		Layer_Base(__func__, inputs, outputs),
		w(outputs*4, inputs),
		w_gradients(outputs*4, inputs),
		r(outputs*4, outputs),
		r_gradients(outputs*4, outputs),
		b(outputs*4),
		b_gradients(outputs*4) {

		w.randomize(-.1, .1);
		b.randomize(-.1, .1);
		r.randomize(-.1, .1);

		zero_gradients();
	}

	auto forget_x(int modifier=0) const {
		return xs[xs.size()-1-t_minus_index+modifier][{{0, 0}, {this->output_size(), this->batch_size()}}];
	}
	auto input_x(int modifier=0) const {
		return xs[xs.size()-1-t_minus_index+modifier][{{0, this->output_size()}, {this->output_size(), this->batch_size()}}];
	}
	auto write_x(int modifier=0) const {
		return xs[xs.size()-1-t_minus_index+modifier][{{0, this->output_size()*2}, {this->output_size(), this->batch_size()}}];
	}
	auto output_x(int modifier=0) const {
		return xs[xs.size()-1-t_minus_index+modifier][{{0, this->output_size()*3}, {this->output_size(), this->batch_size()}}];
	}
	auto& cellstate_x(int modifier=0) {
		return cs[cs.size() - 1 - t_minus_index + modifier];
	}

	auto delta_forget_x() const {
		return delta[{{0, 0}, {this->output_size(), this->batch_size()}}];
	}
	auto delta_input_x() const {
		return delta[{{0, this->output_size()}, {this->output_size(), this->batch_size()}}];
	}
	auto delta_write_x() const {
		return delta[{{0, this->output_size()*2}, {this->output_size(), this->batch_size()}}];
	}
	auto delta_output_x() const {
		return delta[{{0, this->output_size()*3}, {this->output_size(), this->batch_size()}}];
	}

	template<class X, class Y>
	auto forward_propagation(const X& x, const Y& y) {
		if (cs.empty()) {
			cs.push_back(c);
		}

		t_minus_index = 0;
		xs.push_back( f_g(w * x + r * y + b) );

		auto f = forget_x();
		auto i = input_x();
		auto z = write_x();
		auto o = output_x();

		c %= f; 	//%= element-wise assign-multiplication
		c += z % i; //%  element-wise multiplication
		cs.push_back(c);

		return c_g(c) % o;
	}

	template<class X, class Y, class Delta>
	auto back_propagation(const X& x, const Y& y,  const Delta& delta_outputs) {
		//LSTM Backprop reference
		//Reference: https://arxiv.org/pdf/1503.04069.pdf
		//delta_t+1 (deltas haven't been updated from previous step yet)

		if (t_minus_index != 0) {
			r_gradients -= delta * y.t();
		}

		auto f = forget_x();
		auto z = write_x();
		auto i = input_x();
		auto o = output_x();
		mat& cm1 = cellstate_x(-1);
		mat& c = cellstate_x();

		dy = delta_outputs + r.t() * delta;
		delta_output_x() = dy % c_g(c) % o_g.cached_dx(o);

		if (t_minus_index != 0) {
			dc = dy % o % c_g.dx(c) + dc % forget_x(1);
		} else {
			dc = dy % o % c_g.dx(c);
		}

		delta_forget_x() = dc % cm1  % f_g.cached_dx(f);
		delta_input_x() = dc % z % f_g.cached_dx(i);
		delta_write_x() = dc % i % f_g.cached_dx(z);

		w_gradients -= delta * x.t();
		b_gradients -= delta;

		//Increment the current index
		//of the internal cell-state
		t_minus_index++;
		return w.t() * delta;
	}

	void update_weights() {
		ValueType lr = this->lr / this->batch_size();

		w += w_gradients * lr;
		r += r_gradients * lr;
		b += b_gradients * lr;

		zero_gradients();
		clear_bp_storage();
	}

	void set_batch_size(int bs) {
		Layer_Base::set_batch_size(bs);
		for (auto& d: reference_list(dc, c)) {
			d = mat(this->output_size(), bs);
		}
		delta = mat(this->output_size() * 4, bs);
		dy = mat(this->output_size(), bs);

		zero_deltas();
	}

	void clear_bp_storage() {
		//remove all but the last element
		for (auto& bp_storage : reference_list(cs, xs)) {
			bp_storage.erase(bp_storage.begin(), bp_storage.end()-1);
		}
	}

	void zero_deltas() {
		for (auto& delta : reference_list(dc, delta, dy)) {
			delta.zero();
		}
	}

	void zero_gradients() {
		for (auto& grad : reference_list(
				w_gradients,
				r_gradients)) {
			grad.zero();
		}

		b_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(b, "b");
		loader.save_variable(w, "w");
		loader.save_variable(c, "c");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(b, "b");
		loader.load_variable(w, "w");
		loader.load_variable(c, "c");
		loader.load_variable(c, "c");
	}

	auto get_learning_rate() const { return lr; }
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
LSTM<SystemTag, ValueType> lstm(SystemTag system_tag, int inputs, int outputs) {
	return LSTM<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto lstm(SystemTag system_tag, int inputs, int outputs) {
	return LSTM<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto lstm(int inputs, int outputs) {
	return LSTM<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}
}

#endif /* LSTM_H_ */
