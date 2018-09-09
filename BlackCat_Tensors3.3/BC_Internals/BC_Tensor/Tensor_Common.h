/*
 * Tensor_Common.h
 *
 *  Created on: Sep 9, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_
#define BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_

#define BC_ARRAY_ONLY(literal) static_assert(true, " ");//static_assert(BC::is_array<functor_of<derived>>(), "BC Method: '" literal "' is only supported by Array_Base classes")

#include <type_traits>
namespace BC {
template<int> class DISABLED;

}

#endif /* BC_INTERNALS_BC_TENSOR_TENSOR_COMMON_H_ */
