/*
 * LSTM.h
 *
 *  Created on: Aug 3, 2019
 *	  Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_H_

#include "../layer_cache.h"
#include "layer_base.h"

namespace bc {
namespace nn {

using bc::algorithms::reference_list;

template<class SystemTag,
		class ValueType,
		class Optimizer=Stochastic_Gradient_Descent,
		class ForgetGateNonlinearity=bc::Logistic,
		class WriteGateNonlinearity=bc::Tanh,
		class InputGateNonlinearity=bc::Logistic,
		class OutputGateNonlinearity=bc::Logistic,
		class CellStateNonLinearity=bc::Tanh>
struct LSTM:
		public Layer_Base<LSTM<
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
	using parent_type = Layer_Base<LSTM<
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

	using greedy_evaluate_delta = std::true_type;
	using forward_requires_outputs = std::true_type;
	using backward_requires_outputs = std::true_type;
	using requires_extra_cache = std::true_type;
	using is_recurrent = std::true_type;

#ifndef _MSC_VER
	using defines_predict = std::true_type;
#endif

	using defines_single_predict = std::true_type;

private:

	using mat = bc::Matrix<value_type, allocator_type>;
	using vec = bc::Vector<value_type, allocator_type>;

	using mat_opt_t = typename Optimizer::template Optimizer<mat>;
	using vec_opt_t = typename Optimizer::template Optimizer<vec>;

	CellStateNonLinearity  c_g;
	ForgetGateNonlinearity f_g;
	WriteGateNonlinearity  z_g;
	InputGateNonlinearity  i_g;
	OutputGateNonlinearity o_g;

	mat wf, wz, wi, wo;
	mat wf_gradients, wz_gradients, wi_gradients, wo_gradients;

	mat rf, rz, ri, ro;
	mat rf_gradients, rz_gradients, ri_gradients, ro_gradients;

	vec bf, bz, bi, bo;
	vec bf_gradients, bz_gradients, bi_gradients, bo_gradients;

	mat_opt_t wf_opt, wz_opt, wi_opt, wo_opt;
	mat_opt_t rf_opt, rz_opt, ri_opt, ro_opt;
	vec_opt_t bf_opt, bz_opt, bi_opt, bo_opt;

	mat dc, df, dz, di, do_, dy;

	template<char C>
	using key_type = bc::nn::cache_key<
			bc::utility::Name<C>, mat, is_recurrent>;

	using cell_key = key_type<'c'>;
	using forget_key = key_type<'f'>;
	using input_key = key_type<'i'>;
	using write_key = key_type<'z'>;
	using output_key = key_type<'o'>;

	using predict_cell_key = bc::nn::cache_key<
			bc::utility::Name<'p','c'>, vec, is_recurrent>;

public:

	LSTM(int inputs, bc::size_t  outputs):
			parent_type(__func__, inputs, outputs),
			wf(outputs, inputs),
			wz(outputs, inputs),
			wi(outputs, inputs),
			wo(outputs, inputs),

			wf_gradients(outputs, inputs),
			wz_gradients(outputs, inputs),
			wi_gradients(outputs, inputs),
			wo_gradients(outputs, inputs),

			rf(outputs, outputs),
			rz(outputs, outputs),
			ri(outputs, outputs),
			ro(outputs, outputs),

			rf_gradients(outputs, outputs),
			rz_gradients(outputs, outputs),
			ri_gradients(outputs, outputs),
			ro_gradients(outputs, outputs),

			bf(outputs),
			bz(outputs),
			bi(outputs),
			bo(outputs),

			bf_gradients(outputs),
			bz_gradients(outputs),
			bi_gradients(outputs),
			bo_gradients(outputs),

			wf_opt(outputs, inputs),
			wz_opt(outputs, inputs),
			wi_opt(outputs, inputs),
			wo_opt(outputs, inputs),

			rf_opt(outputs, outputs),
			rz_opt(outputs, outputs),
			ri_opt(outputs, outputs),
			ro_opt(outputs, outputs),

			bf_opt(outputs),
			bz_opt(outputs),
			bi_opt(outputs),
			bo_opt(outputs)
	{
		randomize_weights();
		zero_gradients();
	}

	void randomize_weights()
	{
		wf.randomize(-.1, .1);
		wz.randomize(-.1, .1);
		wi.randomize(-.1, .1);
		wo.randomize(-.1, .1);

		rf.randomize(-.1, .1);
		rz.randomize(-.1, .1);
		ri.randomize(-.1, .1);
		ro.randomize(-.1, .1);

		bf.randomize(-.1, .1);
		bz.randomize(-.1, .1);
		bi.randomize(-.1, .1);
		bo.randomize(-.1, .1);
	}

	template<class X, class Y>
	auto forward_propagation(const X& x, const Y& y, Cache& cache)
	{
		mat& f = cache.store(forget_key(), f_g(wf * x + rf * y + bf));
		mat& z = cache.store(write_key(),  z_g(wz * x + rz * y + bz));
		mat& i = cache.store(input_key(),  i_g(wi * x + ri * y + bi));
		mat& o = cache.store(output_key(), o_g(wo * x + ro * y + bo));
		mat& c = cache.load(cell_key(), default_tensor_factory());
		c = c % f + z % i; //% element-wise multiplication

		mat& c_ = cache.store(cell_key(), c);
		return c_g(c_) % o;
	}

#ifndef _MSC_VER

	template<class X, class Y>
	auto predict(const X& x, const Y& y, Cache& cache)
	{
		mat f = f_g(wf * x + rf * y + bf);
		mat z = z_g(wz * x + rz * y + bz);
		mat i = i_g(wi * x + ri * y + bi);
		mat o = o_g(wo * x + ro * y + bo);
		mat& c = cache.load(cell_key(), default_tensor_factory());
		c = c % f + z % i; //%  element-wise multiplication

		mat& c_ = cache.store(cell_key(), c);
		return c_g(c_) % o;
	}

#endif

	template<class X, class Y>
	auto single_predict(const X& x, const Y& y, Cache& cache)
	{
		vec f = f_g(wf * x + rf * y + bf);
		vec z = z_g(wz * x + rz * y + bz);
		vec i = i_g(wi * x + ri * y + bi);
		vec o = o_g(wo * x + ro * y + bo);
		vec& c = cache.load(predict_cell_key(), default_predict_tensor_factory());

		c = c % f + z % i; //%  element-wise multiplication
		return c_g(c) % o;
	}

	template<class X, class Y, class Delta>
	auto back_propagation(const X& x, const Y& y,
			const Delta& delta_outputs, class Cache& cache)
	{
		//LSTM Backprop reference
		//Reference: https://arxiv.org/pdf/1503.04069.pdf

		if (cache.get_time_index() != 0) {
			rz_gradients -= dz * y.t();
			rf_gradients -= df * y.t();
			ri_gradients -= di * y.t();
			ro_gradients -= do_ * y.t();
		}

		auto& z = cache.load(write_key(), default_tensor_factory());
		auto& i = cache.load(input_key(), default_tensor_factory());
		auto& f = cache.load(forget_key(), default_tensor_factory());
		auto& o = cache.load(output_key(), default_tensor_factory());
		auto& cm1 = cache.load(cell_key(), -1, default_tensor_factory());
		auto& c = cache.load(cell_key(), default_tensor_factory());

		dy = delta_outputs +
				rz.t() * dz +
				ri.t() * di +
				rf.t() * df +
				ro.t() * do_;

		do_ = dy % c_g(c) % o_g.cached_dx(o);

		if (cache.get_time_index() != 0) {
			auto& fp1 = cache.load(forget_key(), 1, default_tensor_factory());
			dc = dy % o % c_g.dx(c) + dc % fp1;
		} else {
			dc = dy % o % c_g.dx(c);
		}

		df = dc % cm1  % f_g.cached_dx(f);
		di = dc % z % i_g.cached_dx(i);
		dz = dc % i % z_g.cached_dx(z);

		wz_gradients -= dz * x.t();
		wf_gradients -= df * x.t();
		wi_gradients -= di * x.t();
		wo_gradients -= do_ * x.t();

		bz_gradients -= dz;
		bf_gradients -= df;
		bi_gradients -= di;
		bo_gradients -= do_;

		return wz.t() * dz +
				wi.t() * dz +
				wf.t() * df +
				wo.t() * do_;
	}

	void update_weights()
	{
		wz_opt.update(wz, wz_gradients);
		wf_opt.update(wf, wf_gradients);
		wi_opt.update(wi, wi_gradients);
		wo_opt.update(wo, wo_gradients);

		rz_opt.update(rz, rz_gradients);
		rf_opt.update(rf, rf_gradients);
		ri_opt.update(ri, ri_gradients);
		ro_opt.update(ro, ro_gradients);

		bz_opt.update(bz, bz_gradients);
		bf_opt.update(bf, bf_gradients);
		bi_opt.update(bi, bi_gradients);
		bo_opt.update(bo, bo_gradients);

		zero_gradients();
	}

	void set_learning_rate(value_type lr)
	{
		parent_type::set_learning_rate(lr);
		value_type batched_lr = this->get_batched_learning_rate();

		auto optimizers = reference_list(
				wz_opt, wf_opt, wi_opt, wo_opt,
				rz_opt, rf_opt, ri_opt, ro_opt);

		auto bias_optimizers = reference_list(
				bf_opt, bz_opt, bi_opt, bo_opt);

		for (auto& optimizer : optimizers)
			optimizer.set_learning_rate(batched_lr);

		for (auto& optimizer : bias_optimizers)
			optimizer.set_learning_rate(batched_lr);
	}

	void set_batch_size(int bs)
	{
		parent_type::set_batch_size(bs);

		auto make_default =  [&](){
				mat m(this->output_size(), bs);
				m.zero();
				return m;
		};

		for (auto& tensor: reference_list(dc, df, dz, di, do_, dy)) {
			tensor = make_default();
		}
	}

	void zero_deltas()
	{
		for (auto& delta : reference_list(dc, df, di, dz, do_, dy)) {
			delta.zero();
		}
	}

	void zero_gradients()
	{
		for (auto& grad : reference_list(
				wf_gradients, wz_gradients, wi_gradients, wo_gradients,
				rf_gradients, rz_gradients, ri_gradients, ro_gradients)) {
			grad.zero();
		}

		for (auto& grad : reference_list(
				bf_gradients, bz_gradients, bi_gradients, bo_gradients)) {
			grad.zero();
		}
	}

	void clear_bp_storage(Cache& m_cache)
	{
		m_cache.clear_bp_storage(cell_key());
		m_cache.clear_bp_storage(write_key());
		m_cache.clear_bp_storage(input_key());
		m_cache.clear_bp_storage(forget_key());
		m_cache.clear_bp_storage(output_key());
	}

	void save(Layer_Loader& loader)
	{
		loader.save_variable(wf, "wf");
		loader.save_variable(rf, "rf");
		loader.save_variable(bf, "bf");

		loader.save_variable(wz, "wz");
		loader.save_variable(rz, "rz");
		loader.save_variable(bz, "bz");

		loader.save_variable(wi, "wi");
		loader.save_variable(ri, "ri");
		loader.save_variable(bi, "bi");

		loader.save_variable(wo, "wo");
		loader.save_variable(ro, "ro");
		loader.save_variable(bo, "bo");

		wf_opt.save(loader, "wf_opt");
		wz_opt.save(loader, "wz_opt");
		wi_opt.save(loader, "wi_opt");
		wo_opt.save(loader, "wo_opt");

		rf_opt.save(loader, "rf_opt");
		rz_opt.save(loader, "rz_opt");
		ri_opt.save(loader, "ri_opt");
		ro_opt.save(loader, "ro_opt");

		bf_opt.save(loader, "bf_opt");
		bz_opt.save(loader, "bz_opt");
		bi_opt.save(loader, "bi_opt");
		bo_opt.save(loader, "bo_opt");
	}

	void save_from_cache(Layer_Loader& loader, Cache& cache)
	{
		auto& z = cache.load(write_key(), default_tensor_factory());
		auto& i = cache.load(input_key(), default_tensor_factory());
		auto& f = cache.load(forget_key(), default_tensor_factory());
		auto& o = cache.load(output_key(), default_tensor_factory());
		auto& c = cache.load(cell_key(), default_tensor_factory());

		loader.save_variable(z, "write_gate_values");
		loader.save_variable(i, "input_gate_values");
		loader.save_variable(f, "forget_gate_values");
		loader.save_variable(o, "output_gate_values");
		loader.save_variable(c, "cellstate");

		if (cache.contains(predict_cell_key())) {
			auto& pc = cache.load(
					predict_cell_key(),
					default_predict_tensor_factory());
			loader.save_variable(pc, "predict_cellstate");
		}
	}

	void load(Layer_Loader& loader)
	{
		loader.load_variable(wf, "wf");
		loader.load_variable(rf, "rf");
		loader.load_variable(bf, "bf");

		loader.load_variable(wz, "wz");
		loader.load_variable(rz, "rz");
		loader.load_variable(bz, "bz");

		loader.load_variable(wi, "wi");
		loader.load_variable(ri, "ri");
		loader.load_variable(bi, "bi");

		loader.load_variable(wo, "wo");
		loader.load_variable(ro, "ro");
		loader.load_variable(bo, "bo");

		wf_opt.load(loader, "wf_opt");
		wz_opt.load(loader, "wz_opt");
		wi_opt.load(loader, "wi_opt");
		wo_opt.load(loader, "wo_opt");

		rf_opt.load(loader, "rf_opt");
		rz_opt.load(loader, "rz_opt");
		ri_opt.load(loader, "ri_opt");
		ro_opt.load(loader, "ro_opt");

		bf_opt.load(loader, "bf_opt");
		bz_opt.load(loader, "bz_opt");
		bi_opt.load(loader, "bi_opt");
		bo_opt.load(loader, "bo_opt");
	}

	void load_to_cache(Layer_Loader& loader, Cache& cache)
	{
		auto& z = cache.load(write_key(), default_tensor_factory());
		auto& i = cache.load(input_key(), default_tensor_factory());
		auto& f = cache.load(forget_key(), default_tensor_factory());
		auto& o = cache.load(output_key(), default_tensor_factory());
		auto& c = cache.load(cell_key(), default_tensor_factory());

		loader.load_variable(z, "write_gate_values");
		loader.load_variable(i, "input_gate_values");
		loader.load_variable(f, "forget_gate_values");
		loader.load_variable(o, "output_gate_values");
		loader.load_variable(c, "cellstate");

		if (loader.file_exists(1, "predict_cellstate")) {
			auto& pc = cache.load(
					predict_cell_key(),
					default_predict_tensor_factory());
			loader.load_variable(pc, "predict_cellstate");
		}
	}

	void copy_training_data_to_single_predict(Cache& cache, int batch_index)
	{
		auto& pc = cache.load(predict_cell_key(), default_predict_tensor_factory());
		auto& c = cache.load(cell_key(), default_tensor_factory());
		pc = c[batch_index];
	}

private:

	auto default_tensor_factory()
	{
		return [&]() {
			return mat(this->output_size(), this->batch_size()).zero();
		};
	}

	auto default_predict_tensor_factory()
	{
		 return [&]() {
			 return vec(this->output_size()).zero();
		 };
	}

};

template<class SystemTag, class Optimizer=nn_default_optimizer_type>
auto lstm(SystemTag system_tag, int inputs, int outputs, Optimizer=Optimizer()) {
	return LSTM<
			SystemTag,
			typename SystemTag::default_floating_point_type,
			Optimizer>(inputs, outputs);
}

template<class Optimizer=nn_default_optimizer_type>
auto lstm(int inputs, int outputs, Optimizer=Optimizer()) {
	return LSTM<
			BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type,
			Optimizer>(inputs, outputs);
}


}
}



#endif /* LSTM_H_ */
