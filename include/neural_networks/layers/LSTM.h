/*
 * LSTM.h
 *
 *  Created on: Aug 3, 2019
 *	  Author: joseph
 */

#ifndef BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_H_
#define BLACKCAT_TENSORS_NEURALNETWORKS_LSTM_H_

#include "../Layer_Cache.h"

namespace BC {
namespace nn {

using BC::algorithms::reference_list;

template<class SystemTag,
		class ValueType,
		class ForgetGateNonlinearity=BC::Logistic,
		class WriteGateNonlinearity=BC::Tanh,
		class InputGateNonlinearity=BC::Logistic,
		class OutputGateNonlinearity=BC::Logistic,
		class CellStateNonLinearity=BC::Tanh>
struct LSTM:
		public Layer_Base<LSTM<
				SystemTag,
				ValueType,
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
			ForgetGateNonlinearity,
			WriteGateNonlinearity,
			InputGateNonlinearity,
			OutputGateNonlinearity,
			CellStateNonLinearity>>;

	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;
	using forward_requires_outputs = std::true_type;
	using backward_requires_outputs = std::true_type;
	using requires_extra_cache = std::true_type;

#ifndef _MSC_VER
	using defines_predict = std::true_type;
#endif

	using defines_single_predict = std::true_type;

private:

	CellStateNonLinearity  c_g;
	ForgetGateNonlinearity f_g;
	WriteGateNonlinearity z_g;
	InputGateNonlinearity i_g;
	OutputGateNonlinearity o_g;

	mat wf, wz, wi, wo;
	mat wf_gradients, wz_gradients, wi_gradients, wo_gradients;

	mat rf, rz, ri, ro;
	mat rf_gradients, rz_gradients, ri_gradients, ro_gradients;

	vec bf, bz, bi, bo;
	vec bf_gradients, bz_gradients, bi_gradients, bo_gradients;

	mat dc, df, dz, di, do_, dy;


	using is_recurrent = std::true_type;

	template<char C>
	using key_type = BC::nn::cache_key<
			BC::utility::Name<C>, mat, is_recurrent>;

	using cell_key = key_type<'c'>;
	using forget_key = key_type<'f'>;
	using input_key = key_type<'i'>;
	using write_key = key_type<'z'>;
	using output_key = key_type<'o'>;

	using predict_cell_key = BC::nn::cache_key<
			BC::utility::Name<'p','c'>, vec, is_recurrent>;

public:

	LSTM(int inputs, BC::size_t  outputs):
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
			bo_gradients(outputs) {

		wf.randomize(-1, 0);
		wz.randomize(-.1, .1);
		wi.randomize(-.1, .1);
		wo.randomize(0, .5);

		rf.randomize(-1, 1);
		rz.randomize(-.1, .1);
		ri.randomize(-.1, .1);
		ro.randomize(0, .5);

		bf.randomize(-1, 1);
		bz.randomize(-.1, .1);
		bi.randomize(-.1, .1);
		bo.randomize(0, .5);

		zero_gradients();
	}

	template<class X, class Y>
	auto forward_propagation(const X& x, const Y& y, Cache& cache) {
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
	auto predict(const X& x, const Y& y, Cache& cache) {
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
	auto single_predict(const X& x, const Y& y, Cache& cache) {
		vec f = f_g(wf * x + rf * y + bf);
		vec z = z_g(wz * x + rz * y + bz);
		vec i = i_g(wi * x + ri * y + bi);
		vec o = o_g(wo * x + ro * y + bo);
		vec& c = cache.load(predict_cell_key(), default_predict_tensor_factory());

		c = c % f + z % i; //%  element-wise multiplication
		return c_g(c) % o;
	}

	template<class X, class Y, class Delta>
	auto back_propagation(const X& x, const Y& y, const Delta& delta_outputs, class Cache& cache) {
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

	void update_weights() {
		ValueType lr = this->get_batched_learning_rate();
		wz += wz_gradients * lr;
		wf += wf_gradients * lr;
		wi += wi_gradients * lr;
		wo += wo_gradients * lr;

		rz += rz_gradients * lr;
		rf += rf_gradients * lr;
		ri += ri_gradients * lr;
		ro += ro_gradients * lr;

		bz += bz_gradients * lr;
		bf += bf_gradients * lr;
		bi += bi_gradients * lr;
		bo += bo_gradients * lr;

		zero_gradients();
	}

	void set_batch_size(int bs) {
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

	void zero_deltas() {
		for (auto& delta : reference_list(dc, df, di, dz, do_, dy)) {
			delta.zero();
		}
	}

	void zero_gradients() {
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

	void clear_bp_storage(Cache& m_cache) {
		m_cache.clear_bp_storage(cell_key());
		m_cache.clear_bp_storage(write_key());
		m_cache.clear_bp_storage(input_key());
		m_cache.clear_bp_storage(forget_key());
		m_cache.clear_bp_storage(output_key());
	}

	void save(Layer_Loader& loader) {
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
	}

	void save_from_cache(Layer_Loader& loader, Cache& cache) {
		auto& c = cache.load(cell_key(), default_tensor_factory());
		loader.save_variable(c, "cellstate");


		if (cache.contains(predict_cell_key())) {
			auto& pc = cache.load(predict_cell_key());
			loader.save_variable(pc, "predict_celltate");
		}
	}

	void load(Layer_Loader& loader) {
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
	}

	void load_from_cache(Layer_Loader& loader, Cache& cache) {
		auto& c = cache.load(cell_key(), default_tensor_factory());
		loader.load_variable(c, "cellstate");

		if (loader.file_exists(1, "cellstate")) {
			auto& pc = cache.load(predict_cell_key());
			loader.load_variable(pc, "predict_celltate");
		}
	}

	void copy_training_data_to_single_predict(Cache& cache, int batch_index) {
		auto& pc = cache.load(predict_cell_key(), default_predict_tensor_factory());
		auto& c = cache.load(cell_key(), default_tensor_factory());
		pc = c[batch_index];
	}

private:

	auto default_tensor_factory() {
		return [&]() {
			mat m(this->output_size(), this->batch_size());
			m.zero();
			return m;
		};
	}

	auto default_predict_tensor_factory() {
		 return [&]() {
			 return vec(this->output_size()).zero();
		 };
	}

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



#endif /* LSTM_H_ */
