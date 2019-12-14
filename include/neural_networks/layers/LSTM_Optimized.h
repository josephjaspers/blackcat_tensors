/*
 * LSTM.h
 *
 *  Created on: Aug 3, 2019
 *	  Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_OPTIMIZED_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_OPTIMIZED_H_

#include "../Layer_Cache.h"

namespace BC {
namespace nn {

using BC::algorithms::reference_list;

template<class SystemTag,
		class ValueType,
		class Optimizer=Stochastic_Gradient_Descent,
		class ForgetGateNonlinearity=BC::Logistic,
		class WriteGateNonlinearity=BC::Tanh,
		class InputGateNonlinearity=BC::Logistic,
		class OutputGateNonlinearity=BC::Logistic,
		class CellStateNonLinearity=BC::Tanh>
struct LSTM_Opt:
		public Layer_Base<LSTM_Opt<
				SystemTag,
				ValueType,
				Optimizer,
				ForgetGateNonlinearity,
				WriteGateNonlinearity,
				InputGateNonlinearity,
				OutputGateNonlinearity,
				CellStateNonLinearity>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<LSTM_Opt<
			SystemTag,
			ValueType,
			Optimizer,
			ForgetGateNonlinearity,
			WriteGateNonlinearity,
			InputGateNonlinearity,
			OutputGateNonlinearity,
			CellStateNonLinearity>>;

	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;
	using optimizer_type = Optimizer;

	using cube = BC::Cube<value_type, allocator_type>;
	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using is_recurrent = std::true_type;
	using greedy_evaluate_delta = std::true_type;
	using forward_requires_outputs = std::true_type;
	using backward_requires_outputs = std::true_type;
	using requires_extra_cache = std::true_type;

	CellStateNonLinearity  c_g;
	ForgetGateNonlinearity f_g;
	WriteGateNonlinearity z_g;
	InputGateNonlinearity i_g;
	OutputGateNonlinearity o_g;

	using cube_opt_t = typename Optimizer::template Optimizer<cube>;
	using mat_opt_t = typename Optimizer::template Optimizer<mat>;
	using vec_opt_t = typename Optimizer::template Optimizer<vec>;

	mat w; //mat wf, wz, wi, wo;
	mat w_gradients; //mat wf_gradients, wz_gradients, wi_gradients, wo_gradients;

	mat r; //mat rf, rz, ri, ro;
	mat r_gradients; //rf_gradients, rz_gradients, ri_gradients, ro_gradients;

	vec b; //vec bf, bz, bi, bo;
	vec b_gradients; //vec bf_gradients, bz_gradients, bi_gradients, bo_gradients;

	mat_opt_t w_opt;// wf_opt, wz_opt, wi_opt, wo_opt;
	mat_opt_t r_opt;// rf_opt, rz_opt, ri_opt, ro_opt;
	vec_opt_t b_opt;// bf_opt, bz_opt, bi_opt, bo_opt;

	mat deltas;
	mat dc, dy;

	enum gates {
		 forget=0,
		 input=1,
		 write=2,
		 output=3,
	};

	auto gate_idx(gates gate) const {
		return std::make_pair(
			BC::dim(0, this->output_size() * gate),
			BC::dim(this->output_size(), this->batch_size())
		);
	}

	auto f_idx() const { return gate_idx(gates::forget); }
	auto i_idx() const { return gate_idx(gates::input);  }
	auto z_idx() const { return gate_idx(gates::write);  }
	auto o_idx() const { return gate_idx(gates::output); }

	template<char... C>
	using key_type = BC::nn::cache_key<
			BC::utility::Name<C...>, mat, is_recurrent>;

	using cell_key = key_type<'c'>;
	using fzio_key = key_type<'f','z','i','o'>;
	using delta_key = key_type<'d','e','l','t', 'a'>;

	using predict_cell_key = BC::nn::cache_key<
			BC::utility::Name<'p','c'>, vec, is_recurrent>;

public:

	LSTM_Opt(int inputs, BC::size_t  outputs):
			parent_type(__func__, inputs, outputs),

			w(outputs * 4, inputs),
			w_gradients(outputs * 4, inputs),

			r(outputs * 4, outputs),
			r_gradients(outputs * 4, outputs),

			b(outputs * 4),
			b_gradients(outputs * 4),

			w_opt(outputs * 4, inputs),
			r_opt(outputs * 4, outputs),
			b_opt(outputs * 4)
	{
		randomize_weights();
		zero_gradients();
	}

	void randomize_weights()
	{
		w.randomize(-.1, .1);
		r.randomize(-.1, .1);
		b.randomize(-.1, .1);
	}

	template<class X, class Y>
	auto forward_propagation(const X& x, const Y& y, Cache& cache)
	{
		mat& fzio = cache.store(fzio_key(), w * x + r * y + b);

		auto f = fzio[f_idx()];
		auto z = fzio[z_idx()];
		auto i = fzio[i_idx()];
		auto o = fzio[o_idx()];
		f = f_g(f);
		z = z_g(z);
		i = i_g(i);
		o = o_g(o);

		auto& c = cache.load(cell_key(), cellstate_factory());
		c = c % f + z % i; //% element-wise multiplication

		mat& c_ = cache.store(cell_key(), c);
		return c_g(c_) % o;
	}

	template<class X, class Y, class Delta>
	auto back_propagation(const X& x, const Y& y,
			const Delta& delta_outputs, class Cache& cache)
	{
		//LSTM Backprop reference
		//Reference: https://arxiv.org/pdf/1503.04069.pdf
		if (cache.get_time_index() != 0) {
			r_gradients -= deltas * y.t();
		}

		auto& fzio = cache.load(fzio_key(), fzio_tensor_factory());
		auto f = fzio[f_idx()];
		auto z = fzio[z_idx()];
		auto i = fzio[i_idx()];
		auto o = fzio[o_idx()];

		auto df = deltas[f_idx()];
		auto dz = deltas[z_idx()];
		auto di = deltas[i_idx()];
		auto do_ = deltas[o_idx()];

		auto& cm1 = cache.load(cell_key(), -1, cellstate_factory());
		auto& c = cache.load(cell_key(), cellstate_factory());

		dy = delta_outputs + r.t() * deltas;
		do_ = dy % c_g(c) % o_g.cached_dx(o);

		if (cache.get_time_index() != 0) {
			auto fp1 = cache.load(fzio_key(), 1, fzio_tensor_factory())[f_idx()];
			dc = dy % o % c_g.dx(c) + dc % fp1;
		} else {
			dc = dy % o % c_g.dx(c);
		}

		df = dc % cm1  % f_g.cached_dx(f);
		di = dc % z % i_g.cached_dx(i);
		dz = dc % i % z_g.cached_dx(z);

		w_gradients -= deltas * x.t();
		b_gradients -= deltas;

		return w.t() * deltas;
	}

	void update_weights()
	{
		w_opt.update(w, w_gradients);
		r_opt.update(r, r_gradients);
		b_opt.update(b, b_gradients);

		zero_gradients();
		zero_deltas();
	}

	void set_learning_rate(value_type lr)
	{
		parent_type::set_learning_rate(lr);
		value_type batched_lr = this->get_batched_learning_rate();

		w_opt.set_learning_rate(batched_lr);
		b_opt.set_learning_rate(batched_lr);
		r_opt.set_learning_rate(batched_lr);
	}

	void set_batch_size(int bs)
	{
		parent_type::set_batch_size(bs);
		deltas = make_default();
		dy = make_cellstate();
		dc = make_cellstate();
	}

	void zero_deltas()
	{
		deltas.zero();
	}

	void zero_gradients()
	{
		w_gradients.zero();
		b_gradients.zero();
		r_gradients.zero();
	}

	void clear_bp_storage(Cache& m_cache)
	{
		m_cache.clear_bp_storage(cell_key());
		m_cache.clear_bp_storage(fzio_key());
	}

	void save(Layer_Loader& loader)
	{
	}

	void save_from_cache(Layer_Loader& loader, Cache& cache)
	{
	}

	void load(Layer_Loader& loader)
	{
	}

private:

	mat make_default() const {
		return mat(this->output_size() * 4, this->batch_size()).zero();
	}

	mat make_cellstate() const {
		return mat(this->output_size(), this->batch_size()).zero();
	}

	auto cellstate_factory() const {
		return [&]() {
			return make_cellstate();
		};
	}

	auto fzio_tensor_factory() const
	{
		return [&]() {
			return make_default();
		};
	}
};

template<class SystemTag, class Optimizer=nn_default_optimizer_type>
auto lstm2(SystemTag system_tag, int inputs, int outputs, Optimizer=Optimizer()) {
	return LSTM_Opt<
			SystemTag,
			typename SystemTag::default_floating_point_type,
			Optimizer>(inputs, outputs);
}

template<class Optimizer=nn_default_optimizer_type>
auto lstm2(int inputs, int outputs, Optimizer=Optimizer()) {
	return LSTM_Opt<
			BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type,
			Optimizer>(inputs, outputs);
}


}
}



#endif /* LSTM_H_ */
