/*
 * GRU.h
 *
 *  Created on: Sep 20, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_LAYERS_GRU_H_
#define BC_INTERNALS_LAYERS_GRU_H_

#include "Layer_Base_Recurrent.h"
#include <forward_list>
#include "Utility.h"
namespace BC {
namespace NN {

struct GRU : public Layer_Base_Recurrent {

	using Layer_Base_Recurrent::lr;	//the learning rate
	using Layer_Base_Recurrent::inputs;
	using Layer_Base_Recurrent::outputs;
	using Layer_Base_Recurrent::t; 	//current_time_stamp
	using Layer_Base_Recurrent::max_backprop_length;
	cube x;							//inputs + outputs , batch_size (concatenated x and y)
	cube f, z, c;				 	//inputs, batch_size, maximum_BPTT
	mat dc, df, dz;
	mat wf, wz;                  	//weights
	vec bf, bz;                  	//biases

	mat wfd, wzd;					//weight deltas
	vec bfd, bzd; 					//bias deltas

	GRU(int inputs, int outputs) :
		Layer_Base_Recurrent(inputs, outputs),
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
		int i = this->inputs();
		int s = x[t].size();

		x[t][{0, i}] =  x_;						//x[t][{begin, end}] --> similar to python's  x[begin:end]
		x[t][{i, s}] = x[t-1][{0, inputs()}];
	}

	template<class T>
	const auto& forward_propagation(const expr::mat<T>& x_) {
		cache_inputs(x_);

		f[t] = g(wf * x[t] + bf);
		z[t] = g(wz * x[t] + bz);

		return c = c % f + z;

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
		x = cube(this->outputs() + this->inputs(), batch_sz, max_backprop_length);
		f = cube(this->outputs() + this->inputs(), batch_sz, max_backprop_length);
		c = cube(this->outputs() + this->outputs(), batch_sz, max_backprop_length);
		z = cube(this->outputs() + this->inputs(), batch_sz, max_backprop_length);
	}
//
//	auto& inputs()  { return x; }
//	auto& outputs() { return c; }
////	auto& deltas()  { return dy;}
//	auto& weights()	{ return w; }
//	auto& bias()	{ return b; }

	template<class tensor, class deltas> void set_activation(tensor& workspace, deltas& error_workspace) {

	}

};
}
}




#endif /* BC_INTERNALS_LAYERS_GRU_H_ */
