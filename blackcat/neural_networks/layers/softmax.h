/*
 * SoftMax.h
 *
 *  Created on: Jul 15, 2019
 *	  Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_SOFTMAX_H_
#define BLACKCAT_NEURALNETWORK_SOFTMAX_H_

#include "layer_base.h"

namespace bc {
namespace nn {

template<class SystemTag, class ValueType>
struct SoftMax:
		public Layer_Base<bc::traits::Integer<1>, ValueType, SystemTag>
{

	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<bc::traits::Integer<1>, ValueType, SystemTag>;

	using self_type = SoftMax<SystemTag, ValueType>;
	using allocator_type = nn_default_allocator_type<SystemTag, ValueType>;

	using greedy_evaluate_delta = std::true_type;
	using mat = bc::Matrix<ValueType, bc::Allocator<SystemTag, ValueType>>;
	using vec = bc::Vector<ValueType, bc::Allocator<SystemTag, ValueType>>;

private:
	using typename parent_type::batched_output_tensor_type;
	using typename parent_type::batched_input_tensor_type;

	mat y;

public:

	SoftMax(int inputs=0):
		parent_type(__func__) {
		this->m_input_shape[0] = inputs;
		this->m_output_shape[0] = inputs;
	}

	void init() override {
		if(this->m_input_shape.size() == 0) {
			this->m_input_shape = this->prev()->input_shape();
			this->m_output_shape = this->m_input_shape;
		}

		y = mat(this->batched_output_shape());
	}

	template<class Allocator>
	const auto& forward_propagation(const bc::Matrix<value_type, Allocator>& x) {
		//TODO -- convert this into an operation, need 'broadcasted' sum
		for (int i = 0; i < x.cols(); ++i) {
			y[i] = bc::exp(x[i]) / bc::tensors::sum(exp(x[i]));
		}

		return y;
	}
	template<class Allocator>
	auto forward_propagation(const bc::Vector<value_type, Allocator>& x) {
		return  bc::exp(x) / bc::tensors::sum(exp(x));
	}

	template<class X, class Matrix>
	auto back_propagation(const X& x, const Matrix& dy) {
		return dy;
	}


	virtual batched_output_tensor_type forward_propagation(
			const batched_input_tensor_type& x) override
	{
		//TODO -- convert this into an operation, need 'broadcasted' sum
		for (int i = 0; i < x.cols(); ++i) {
			y[i] = bc::exp(x[i]) / bc::tensors::sum(exp(x[i]));
		}

		return y;
	}

	virtual batched_input_tensor_type back_propagation(
			const batched_output_tensor_type& dy) override
	{
		return dy;

	}

	virtual void set_batch_size_hook(int batch_size) override
	{
		y = mat(this->input_shape().concat(batch_size));
	}

	virtual void save(Layer_Loader& loader) const override
	{
	}

	virtual void load(Layer_Loader& loader) override
	{
	}

};


template<class ValueType, class SystemTag>
SoftMax<SystemTag, ValueType> softmax(SystemTag system_tag, int inputs) {
	return SoftMax<SystemTag, ValueType>(inputs);
}
template<class SystemTag>
auto softmax(SystemTag system_tag, int inputs) {
	return SoftMax<SystemTag, typename SystemTag::default_floating_point_type>(inputs);
}

auto softmax(int inputs) {
	return SoftMax<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs);
}

}
}




#endif /* SOFTMAX_H_ */
