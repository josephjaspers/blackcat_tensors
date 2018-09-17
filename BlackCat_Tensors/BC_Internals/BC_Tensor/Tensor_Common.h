/*
 * Tensor_Common.h
 *
 *  Created on: Sep 9, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_

#define BC_ARRAY_ONLY(literal) static_assert(BC::is_array<internal_t>(), "BC Method: '" literal "' IS NOT SUPPORTED FOR EXPRESSIONS")

#include <type_traits>
namespace BC {


#define __BCfinline__ inline __attribute__((always_inline)) __attribute__((hot))

template<int> class DISABLED;
template<class> class Tensor_Base;

template<class internal_t>
auto make_tensor(internal_t internal) {
	return Tensor_Base<internal_t>(internal);
}

}

#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
