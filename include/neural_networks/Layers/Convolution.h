/*
 * Convolution.h
 *
 *  Created on: Aug 27, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_CONVOLUTION_H_

#include "Layer_Base.h"


namespace BC {
namespace nn {

template<
	class SystemTag,
	class ValueType>
struct Convolution : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;

	using tensor4 = BC::Tensor<4, value_type, allocator_type>;
	using cube = BC::Cube<value_type, allocator_type>;

	using greedy_evaluate_delta = std::true_type;
	using input_tensor_dimension = BC::traits::Integer<3>;
	using output_tensor_dimension = BC::traits::Integer<3>;

private:

	ValueType lr = Layer_Base::default_learning_rate;

	tensor4 w;  //kernel
	tensor4 w_gradients;

public:

	BC::Shape<3> get_input_shape() const { return m_input_shape; }
	BC::Shape<3> get_output_shape() const { return m_output_shape; }
	BC::Shape<4> get_batched_input_shape() const { return m_batched_input_shape; }
	BC::Shape<4> get_batched_output_shape() const { return m_batched_output_shape; }

	BC::size_t rows, cols, depth;
	BC::size_t krnl_rows, krnl_cols, nkrnls;
	BC::size_t padding, stride;

	BC::Shape<3> m_input_shape;
	BC::Shape<4> m_batched_input_shape;
	BC::Shape<3> m_output_shape;
	BC::Shape<4> m_batched_output_shape;

	Convolution(
			BC::size_t rows,
			BC::size_t cols,
			BC::size_t depth,
			BC::size_t krnl_rows=3,
			BC::size_t krnl_cols=3,
			BC::size_t nkrnls=32,
			BC::size_t padding=0,
			BC::size_t stride=1) :
		Layer_Base(__func__,
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
		m_input_shape(rows, cols, depth),
		m_output_shape(rows + padding*2 + 1 - krnl_rows,
				cols + padding*2 + 1 - krnl_cols, nkrnls),
		m_batched_input_shape(rows, cols, depth, this->batch_size()),
		m_batched_output_shape(rows + padding*2 + 1 - krnl_rows,
				cols + padding*2 + 1 - krnl_cols, nkrnls, this->batch_size())
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
		ValueType lr = this->lr / this->batch_size();
		w += w_gradients * lr;
		w_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(w, "w");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(w, "w");
	}

	void set_batch_size(BC::size_t batch_sz) {
		Layer_Base::set_batch_size(batch_sz);

		m_batched_input_shape = BC::Shape<4>(
				m_input_shape.m_inner_shape[0],
				m_input_shape.m_inner_shape[1],
				m_input_shape.m_inner_shape[2],
				batch_sz);

		m_batched_output_shape = BC::Shape<4>(
				m_output_shape.m_inner_shape[0],
				m_output_shape.m_inner_shape[1],
				m_output_shape.m_inner_shape[2],
				batch_sz);
	}

	auto& get_weight() const { return w; }
	auto& get_weight() { return w; }
	auto get_learning_rate() const { return lr; }
};

}
}



#endif /* CONVOLUTION_H_ */
