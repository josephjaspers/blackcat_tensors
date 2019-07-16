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
    	using BC::exp;
    	using BC::sum;

    	x = mat_view(x_);

//    	std::cout << " x is \n" <<  x[0].to_string() << std::endl;
//    	vec exp_ = BC::exp(x[0]);
//    	std::cout << " exp(x) is \n" << std::endl;
//    	exp_.print();
//    	std::cout << " sum(exp(x)) is \n" <<  BC::sum(exp_) << std::endl;


    	for (int i = 0; i < x.cols(); ++i) {
    		y[i] = exp(x[i]) / sum(exp(x[i]));
    	}


//    	std::cout << "waitign on int " << std::endl;
//    	int x;
//    	std::cin >> x;
        return y;
    }


    template<class Matrix>
    auto back_propagation(const Matrix& dy) {
    	return dy;
//    	return (y % (1-y)) % dy;
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
