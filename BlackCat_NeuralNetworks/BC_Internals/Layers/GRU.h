/*
 * GRU.h
 *
 *  Created on: Sep 20, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_LAYERS_GRU_H_
#define BC_INTERNALS_LAYERS_GRU_H_

#include <forward_list>

#include "Recurrent_Layer_Base.h"
#include "Utility.h"
namespace BC {
namespace NN {

struct GRU : public Recurrent_Layer_Base {

	using Recurrent_Layer_Base::lr;	//the learning rate
	using Recurrent_Layer_Base::numb_inputs;
	using Recurrent_Layer_Base::numb_outputs;
	using Recurrent_Layer_Base::t; 	//current_time_stamp
	using Recurrent_Layer_Base::max_backprop_length;
	std::vector<mat_view> x = std::vector<mat_view>(this->get_max_bptt_length());					//inputs + outputs , batch_size (concatenated x and y)
	cube f, z, c;				 	//inputs, batch_size, maximum_BPTT
	mat dc, df, dz;
	mat wf, wz;                  	//weights
	vec bf, bz;                  	//biases

	mat wfd, wzd;					//weight deltas
	vec bfd, bzd; 					//bias deltas

	GRU(int inputs, int outputs) :
		Recurrent_Layer_Base(inputs, outputs),
			wf(outputs, outputs), wz(outputs, outputs),
			bf(outputs), bz(outputs),
			wfd(outputs, outputs), wzd(outputs, outputs),
			bfd(outputs), bzd(outputs)

	{
		wf.randomize(-1, 0);
		wz.randomize(-1, 1);
		bf.randomize(-1, 0);
		bz.randomize(-1, 1);
	}
	template<class T>
	void cache_inputs(const expr::mat<T>& x_) {
		int i = numb_inputs();
		int s = x[t].size();

		x[t][{0, i}] =  x_;						//x[t][{begin, end}] --> similar to python's  x[begin:end]
		x[t][{i, s}] = x[t-1][{0, numb_inputs()}];
	}

	template<class T>
	const auto forward_propagation(const expr::mat<T>& x_) {
		cache_inputs(x_);

		f[t] = g(wf * x[t] + bf);
		z[t] = g(wz * x[t] + bz);

		return c[t] = c[t] % f[t] + z[t];

	}
	template<class T>
	auto back_propagation(const expr::mat<T>& dy_) {
		dc = dy_;
		dz = dc * gd(z[t]);
		df = dc % c[t - 1] % gd(f[t]);

		return (wf.t() * dz + wz.t() * dz) % gd(x[t]);
	}
	void cache_gradients() {
		wzd -= dz * lr * z[t].t();
		wfd -= df * lr * f[t].t();

		bzd -= dz;
		bfd -= df;
	}

	void update_weights() {
		wz += wzd;
		wf += wfd;
		bz += bzd;
		bf += bfd;

		wzd.zero();
		wfd.zero();
		bzd.zero();
		bfd.zero();
	}

	void set_batch_size(int batch_sz) {
		Recurrent_Layer_Base::set_batch_size(batch_sz);
		f = cube(numb_outputs(), batch_sz, max_backprop_length);
		c = cube(numb_outputs(), batch_sz, max_backprop_length);
		z = cube(numb_outputs(), batch_sz, max_backprop_length);
	}
	void set_max_bptt_length(int len) {
		Recurrent_Layer_Base::set_max_bptt_length(len);
		x = std::vector<mat_view>(len);
		f = cube(this->numb_outputs(), this->batch_size(), len);
		c = cube(this->numb_outputs(), this->batch_size(), len);
		z = cube(this->numb_outputs(), this->batch_size(), len);
	}

};
}
}




#endif /* BC_INTERNALS_LAYERS_GRU_H_ */
