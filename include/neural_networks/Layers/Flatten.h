/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *	  Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FEEDFORWARD_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYERS_FEEDFORWARD_H_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<
	class SystemTag,
	class ValueType,
	class InputTensorDimension=BC::traits::Integer<3>>
struct Flatten : public Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;

	using mat = BC::Matrix<value_type, allocator_type>;
	using vec = BC::Vector<value_type, allocator_type>;

	using input_tensor_dimension = InputTensorDimension;
	using output_tensor_dimension = BC::traits::Integer<1>;
	using greedy_evaluate_delta = std::true_type;

private:

	BC::Shape<input_tensor_dimension::value> m_input_shape;
	BC::Shape<input_tensor_dimension::value+1> m_batched_input_shape;

public:

	Flatten(BC::Shape<input_tensor_dimension::value> input_shape):
		Layer_Base(__func__, input_shape.size(), input_shape.size()) {}

	template<class Tensor3, class=std::enable_if_t<Tensor3::tensor_dimension==input_tensor_dimension::value>>
	auto forward_propagation(const Tensor3& x) {
		return BC::reshape(x, BC::shape(x.size()));
	}

	template<class Tensor3, class=std::enable_if_t<Tensor3::tensor_dimension==input_tensor_dimension::value+1>>
	auto forward_propagation(const Tensor3& x) {
		return BC::reshape(x, BC::shape(x.size(), this->batch_size()));
	}

	template<class X, class Delta, class=std::enable_if_t<Delta::tensor_dimension==1>>
	auto back_propagation(const X& x, const Delta& dy) {
		return BC::reshape(dy, m_input_shape);
	}

	template<class X, class Delta, class=std::enable_if_t<Delta::tensor_dimension==2>>
	auto back_propagation(const X& x, const Delta& dy) {
		return BC::reshape(dy, m_batched_input_shape);
	}

	void update_weights() {
		ValueType lr = this->lr / this->batch_size();
		w += w_gradients * lr;
		b += b_gradients * lr;
		w_gradients.zero();
		b_gradients.zero();
	}

	void save(Layer_Loader& loader) {
		loader.save_variable(w, "w");
		loader.save_variable(b, "b");
	}

	void load(Layer_Loader& loader) {
		loader.load_variable(w, "w");
		loader.load_variable(b, "b");
	}

	auto& get_weight() const { return w; }
	auto& get_weight() { return w; }
	auto& get_bias() const { return b; }
	auto& get_bias() { return b; }
	auto get_learning_rate() const { return lr; }
};

#ifndef BC_CLING_JIT
template<class ValueType, class SystemTag>
FeedForward<SystemTag, ValueType> feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, ValueType>(inputs, outputs);
}
#endif

template<class SystemTag>
auto feedforward(SystemTag system_tag, int inputs, int outputs) {
	return FeedForward<SystemTag, typename SystemTag::default_floating_point_type>(inputs, outputs);
}

auto feedforward(int inputs, int outputs) {
	return FeedForward<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs, outputs);
}


}
}

#endif /* FEEDFORWARD_CU_ */
