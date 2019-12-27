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
class SoftMax:
		public Layer_Base<SoftMax<SystemTag, ValueType>> {

public:

	using system_tag = SystemTag;
	using value_type = ValueType;
	using parent_type = Layer_Base<SoftMax<SystemTag, ValueType>>;

	using mat = bc::Matrix<ValueType, bc::Allocator<SystemTag, ValueType>>;
	using vec = bc::Vector<ValueType, bc::Allocator<SystemTag, ValueType>>;

private:

	mat y;

public:

	SoftMax(int inputs):
		parent_type(__func__, inputs, inputs) {}

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

	void update_weights() {}
	void set_batch_size(int bs) {
		parent_type::set_batch_size(bs);
		y = mat(this->output_size(), bs);
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
