/*
 * Convolution.h
 *
 *  Created on: Aug 27, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_REFERENCE_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_REFERENCE_H_

#include "Layer_Base.h"


namespace BC {
namespace nn {
namespace deprecated {

template<
	class SystemTag,
	class ValueType>
struct Convolution:
		public Layer_Base<Convolution<SystemTag, ValueType>> {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;
	using parent_type = Layer_Base<Convolution<SystemTag, ValueType>>;
	using tensor4 = BC::Tensor<4, value_type, allocator_type>;
	using cube = BC::Cube<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;
	using input_tensor_dimension = BC::traits::Integer<3>;
	using output_tensor_dimension = BC::traits::Integer<3>;

private:

	tensor4 w;  //kernel
	tensor4 w_gradients;

public:

	Dim<3> get_input_shape() const { return m_input_shape; }
	Dim<3> get_output_shape() const { return m_output_shape; }

	BC::size_t rows, cols, depth;
	BC::size_t krnl_rows, krnl_cols, nkrnls;
	BC::size_t padding, stride;

	Dim<3> m_input_shape;
	Dim<3> m_output_shape;

	Convolution(
			BC::size_t rows,
			BC::size_t cols,
			BC::size_t depth,
			BC::size_t krnl_rows=3,
			BC::size_t krnl_cols=3,
			BC::size_t nkrnls=32,
			BC::size_t padding=0,
			BC::size_t stride=1) :
		parent_type(__func__,
				((rows+padding-krnl_rows+1)/stride) * ((cols+padding-krnl_cols+1)/stride) * depth,
				((rows+padding-krnl_rows+1)/stride) * ((cols+padding-krnl_cols+1)/stride) * nkrnls),
		rows(rows),
		cols(cols),
		depth(depth),
		krnl_rows(krnl_rows),
		krnl_cols(krnl_cols),
		nkrnls(nkrnls),
		padding(padding),
		stride(stride),
		w(krnl_rows, krnl_cols, depth, nkrnls),
		w_gradients(krnl_rows, krnl_cols, depth, nkrnls),
		m_input_shape{rows, cols, depth},
		m_output_shape{rows + padding*2 + 1 - krnl_rows,
				cols + padding*2 + 1 - krnl_cols, nkrnls}
	{
		w.randomize(-3, 3);
	}

	template<class X>
	auto forward_propagation(const X& x) {
		return w.multichannel_conv2d(x);
	}

	template<class X, class Delta>
	auto back_propagation(const X& x, const Delta& dy) {
		w_gradients -= x.multichannel_conv2d_kernel_backwards(dy);
		return w.multichannel_conv2d_data_backwards(dy);
	}

	void update_weights() {
		w += w_gradients * this->get_batched_learning_rate();
		w_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(w, "w");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(w, "w");
	}
};

}
}
}



#endif /* CONVOLUTION_H_ */
