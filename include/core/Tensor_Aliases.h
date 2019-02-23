/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef BLACKCAT_TENSOR_ALIASES_H_
#define BLACKCAT_TENSOR_ALIASES_H_

#include "Tensor_Base.h"

namespace BC {

template<class T>
using Basic_Allocator = allocator::Host<T>;

#ifndef BC_GPU_DEFAULT

	template<class T>
	using alloc_t = allocator::Host<T>;
#else
	using alloc_t = allocator::Device;
#endif

#ifdef __CUDACC__
	template<class T>
	using Cuda_Allocator = allocator::Device<T>;
	template<class T>
	using Cuda_Managed = allocator::Device_Managed<T>;
#endif


template<int dimension, class value_type, class allocator_t=alloc_t<value_type>>
using Tensor = Tensor_Base<expression_template::Array<dimension, value_type, allocator_t>>;

#define BC_TENSOR_ALIAS_CORE_DEF(index, name)\
		template<class value_type, class allocator_t = alloc_t<value_type>>\
		using name = Tensor<index, value_type, allocator_t>;

BC_TENSOR_ALIAS_CORE_DEF(0, Scalar)
BC_TENSOR_ALIAS_CORE_DEF(1, Vector)
BC_TENSOR_ALIAS_CORE_DEF(2, Matrix)
BC_TENSOR_ALIAS_CORE_DEF(3, Cube)

#undef BC_TENSOR_ALIAS_CORE_DEF

template<int dimension, class value_type, class allocator_t=alloc_t<value_type>>
using Tensor_View = Tensor_Base<expression_template::Array_View<dimension, value_type, allocator_t>>;

template<class value_type, class allocator_t = alloc_t<value_type>> using Scalar_View = Tensor_View<0, value_type, allocator_t>;
template<class value_type, class allocator_t = alloc_t<value_type>> using Vector_View = Tensor_View<1, value_type, allocator_t>;
template<class value_type, class allocator_t = alloc_t<value_type>> using Matrix_View = Tensor_View<2, value_type, allocator_t>;
template<class value_type, class allocator_t = alloc_t<value_type>> using Cube_View   = Tensor_View<3, value_type, allocator_t>;

namespace expr {
template<int x, class iterator_t, typename = std::enable_if_t<iterator_t::DIMS == x>>
using tensor = Tensor_Base<iterator_t>;

template<class iterator_t> using scal = tensor<0, iterator_t>;
template<class iterator_t> using vec  = tensor<1, iterator_t>;
template<class iterator_t> using mat  = tensor<2, iterator_t>;
template<class iterator_t> using cube = tensor<3, iterator_t>;
}
}

#endif /* TENSOR_ALIASES_H_ */
