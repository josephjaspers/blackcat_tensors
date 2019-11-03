/*
 * Max_Pooling.h
 *
 *  Created on: Oct 11, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_FUNCTIONS_MAX_POOLING_H_
#define BLACKCATTENSORS_NEURALNETWORKS_FUNCTIONS_MAX_POOLING_H_

namespace BC {
namespace nn {
namespace functions {

template<class SystemTag>
struct Max_Pooling;

template<>
struct Max_Pooling<BC::host_tag> {

	template<class IndexTensor, class MaxTensor, class InputTensor>
	void forward(
			IndexTensor index_tensor,
			MaxTensor max_tensor,
			InputTensor input_tensor,
			BC::size_t pool_range=3) {

		BC_ASSERT(index_tensor.inner_shape() == max_tensor.inner_shape(),
				"Index and Max tensor should have same dimensions");
		BC_ASSERT(index_tensor.rows() == input_tensor.rows() / pool_range,
				"dimension mismatch");
		BC_ASSERT(index_tensor.cols() == input_tensor.cols() / pool_range,
				"dimension mismatch");

#define MP_FOR(value)\
for (int d##value = 0; d##value < input_tensor.dimension(value); ++d##value)

		using value_type = typename InputTensor::value_type;

		MP_FOR(4) {
		MP_FOR(3) {
		MP_FOR(2) {
		MP_FOR(1) {
		MP_FOR(0) {
			BC::size_t max_index = input_tensor.dims_to_index(d2,d1*pool_range,d0*pool_range);
			value_type max = input_tensor[max_index];
			for (int j = 0; j < pool_range; ++j) {
				for (int k = 0; k < pool_range; ++k) {
					auto index = input_tensor.dims_to_index(d2,d1*pool_range+j, d0*pool_range+k);
					if (max < input_tensor[index]) {
						max = input_tensor[index];
						max_index = index;
					}
				}
			}
			max_tensor(d2,d1,d0) = max;
			index_tensor(d2,d1,d0) = max_index;
		}
		}
		}
		}
		}
	}

};

}
}
}




#endif /* MAX_POOLING_H_ */
