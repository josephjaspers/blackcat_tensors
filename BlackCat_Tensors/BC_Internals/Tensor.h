/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef TENSOR_ALIASES_H_
#define TENSOR_ALIASES_H_

#include <type_traits>
#include "Mathematics/CPU.h"
#include "Mathematics/GPU.cu"
#include "stl_style_Allocators/Basic_Allocator.h"
#include "stl_style_Allocators/CUDA_Allocator.h"
#include "stl_style_Allocators/CUDA_Managed_Allocator.h"
#include "Tensor_Base.h"
#include "Tensor_Algorithm.h"//needs to be included after allocators

namespace BC {

#ifndef BC_GPU_DEFAULT
//class CPU;
using alloc_t = module::stl::Basic_Allocator;
#else
class GPU;
using alloc_t = module::stl::CUDA_Allocator;
#endif

using Basic_Allocator = module::stl::Basic_Allocator;
#ifdef __CUDACC__
using CUDA_Allocator = module::stl::CUDA_Allocator;
using CUDA = CUDA_Allocator;
using CUDA_Managed = module::stl::CUDA_Managed_Allocator;
#endif
template<int dimension, class scalar_t, class allocator_t=alloc_t>
using Tensor = Tensor_Base<internal::Array<dimension, scalar_t, allocator_t>>;

template<class scalar_t, class allocator_t = alloc_t> using Scalar = Tensor<0, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Vector = Tensor<1, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Matrix = Tensor<2, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Cube   = Tensor<3, scalar_t, allocator_t>;

template<int dimension, class scalar_t, class allocator_t=alloc_t>
using Tensor_View = Tensor_Base<internal::Array_View<dimension, scalar_t, allocator_t>>;

template<class scalar_t, class allocator_t = alloc_t> using Scalar_View = Tensor_View<0, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Vector_View = Tensor_View<1, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Matrix_View = Tensor_View<2, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Cube_View   = Tensor_View<3, scalar_t, allocator_t>;

template<int dimension, class scalar_t, class allocator_t=alloc_t>
using Tensor_Shared = Tensor_Base<internal::Array_Shared<dimension, scalar_t, allocator_t>>;

template<class scalar_t, class allocator_t = alloc_t> using Scalar_Shared = Tensor_Shared<0, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Vector_Shared = Tensor_Shared<1, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Matrix_Shared = Tensor_Shared<2, scalar_t, allocator_t>;
template<class scalar_t, class allocator_t = alloc_t> using Cube_Shared   = Tensor_Shared<3, scalar_t, allocator_t>;

namespace expr {
template<int x, class iterator_t, typename = std::enable_if_t<iterator_t::DIMS() == x>>
using tensor = Tensor_Base<iterator_t>;

template<class iterator_t> using scal = tensor<0, iterator_t>;
template<class iterator_t> using vec  = tensor<1, iterator_t>;
template<class iterator_t> using mat  = tensor<2, iterator_t>;
template<class iterator_t> using cube  = tensor<3, iterator_t>;
}
}

#endif /* TENSOR_ALIASES_H_ */
