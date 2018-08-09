/*
 * Tensor_Aliases.h
 *
 *  Created on: Aug 7, 2018
 *      Author: joseph
 */

#ifndef TENSOR_ALIASES_H_
#define TENSOR_ALIASES_H_
#include "Tensor_Base.h"


namespace BC {

template<class scalar_t, class allocator_t> using Scalar = Tensor_Base<internal::Array<0, scalar_t, allocator_t>>;
template<class scalar_t, class allocator_t> using Vector = Tensor_Base<internal::Array<1, scalar_t, allocator_t>>;
template<class scalar_t, class allocator_t> using Matrix = Tensor_Base<internal::Array<2, scalar_t, allocator_t>>;
template<class scalar_t, class allocator_t> using Cube   = Tensor_Base<internal::Array<3, scalar_t, allocator_t>>;

namespace expr {
template<class iterator_t, typename = std::enable_if_t<iterator_t::DIMS() == 0>> using scal = Tensor_Base<iterator_t>;
template<class iterator_t, typename = std::enable_if_t<iterator_t::DIMS() == 1>> using vec  = Tensor_Base<iterator_t>;
template<class iterator_t, typename = std::enable_if_t<iterator_t::DIMS() == 2>> using mat  = Tensor_Base<iterator_t>;
template<class iterator_t, typename = std::enable_if_t<iterator_t::DIMS() == 3>> using cube = Tensor_Base<iterator_t>;
}

}



#endif /* TENSOR_ALIASES_H_ */
