/*
 * SoftMax.h
 *
 *  Created on: Jul 15, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORK_SOFTMAX_H_
#define BLACKCAT_NEURALNETWORK_SOFTMAX_H_

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
class SoftMax : public Layer_Base {

public:

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
    using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

    using mat_view = BC::Matrix_View<ValueType, BC::Allocator<SystemTag, ValueType>>;

private:
    mat y;
    mat_view x;

public:

    SoftMax(int inputs):
        Layer_Base(inputs, inputs) {}

    template<class Allocator>
    const auto& forward_propagation(const BC::Matrix<value_type, Allocator>& x_) {
    	using BC::exp;
    	using BC::sum;

    	//TODO -- convert this into an operation, need 'broadcasted' sum
    	mat x = mat(x_);
    	for (int i = 0; i < x.cols(); ++i) {
    		y[i] = exp(x[i]) / sum(exp(x[i]));
    	}

        return y;
    }
    template<class Allocator>
    auto forward_propagation(const BC::Vector<value_type, Allocator>& x) {
    	using BC::exp;
    	using BC::sum;

    	//TODO -- convert this into an operation, need 'broadcasted' sum
    		y[0] = exp(x) / sum(exp(x));

        return y[0];
    }


    template<class Matrix>
	auto back_propagation(const mat& x, const Matrix& dy) {
		return dy;
	}

    void update_weights() {}

    void set_batch_size(int x) {
        y = mat(this->output_size(), x);
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
