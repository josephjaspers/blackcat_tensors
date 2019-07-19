/*
 * FeedForward.cu
 *
 *  Created on: Jan 28, 2018
 *      Author: joseph
 */

#ifndef OUTPUTas_CU
#define OUTPUTas_CU

#include "Layer_Base.h"

namespace BC {
namespace nn {

template<class SystemTag, class ValueType>
struct OutputLayer : Layer_Base {

	using system_tag = SystemTag;
	using value_type = ValueType;

	using mat = BC::Matrix<ValueType, BC::Allocator<SystemTag, ValueType>>;
    using vec = BC::Vector<ValueType, BC::Allocator<SystemTag, ValueType>>;

public:

    OutputLayer(int inputs) : Layer_Base(inputs, inputs) {}

    template <class Matrix>
    const auto& forward_propagation(const Matrix& x) {
    	return x;
    }

    template <class Matrix>
    auto back_propagation(const mat& x, const Matrix& exp) {
        return x - exp;
    }

    void update_weights() {}
    void clear_stored_gradients() {}
    void write(std::ofstream& is) {}
    void read(std::ifstream& os) {}

};

template<class ValueType, class SystemTag>
OutputLayer<SystemTag, ValueType> outputlayer(SystemTag system_tag, int inputs) {
	return OutputLayer<SystemTag, ValueType>(inputs);
}
template<class SystemTag>
auto outputlayer(SystemTag system_tag, int inputs) {
	return OutputLayer<SystemTag, typename SystemTag::default_floating_point_type>(inputs);
}
auto outputlayer(int inputs) {
	return OutputLayer<BLACKCAT_DEFAULT_SYSTEM_T,
			typename BLACKCAT_DEFAULT_SYSTEM_T::default_floating_point_type>(inputs);
}


}
}



#endif /* FEEDFORWARD_CU_ */
