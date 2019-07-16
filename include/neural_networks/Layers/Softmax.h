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

    template<class Expression>
    const auto& forward_propagation(const BC::MatrixXpr<Expression>& x_) {
    	x = mat_view(x_);
        for (auto xyiters = std::make_pair(x.nd_begin(), y.nd_begin());
        		xyiters.first != x.nd_end() && xyiters.second != y.nd_end();
        		xyiters.first++, xyiters.second++) {
        	auto ycol = *xyiters.second;
        	auto xcol = *xyiters.first;

        	ycol = BC::exp(xcol) / BC::sum(BC::exp(xcol));
        }
        return y;
    }

//    template<class Expression>
//    const auto& forward_propagation(const BC::VectorXpr<Expression>& x_) {
//    	x = mat_view(x_);
//    	//Todo change to functor
//        return y = BC::exp(x) / BC::sum(BC::exp(x));
//    }

    template<class Matrix>
    auto back_propagation(const Matrix& dy) {
    	return dy;
    }
    void update_weights() {}

    void set_batch_size(int x) {
        y = mat(this->numb_outputs(), x);
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

}
}




#endif /* SOFTMAX_H_ */
