/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 18, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_SHAPE_H_
#define BC_TENSOR_SUPER_SHAPE_H_

#include "BC_Tensor_Super_Shape_impl.h"


template<int ... dimensions>
struct Shape {

	constexpr int size()  const { return BC_Shape_Identity::size <dimensions...>(); }
	constexpr int rows()   const { return BC_Shape_Identity::row  <dimensions...>(); }
	constexpr int cols()   const { return BC_Shape_Identity::col  <dimensions...>(); }
	constexpr int depth() const { return BC_Shape_Identity::depth<dimensions...>(); }
	constexpr int pages() const { return BC_Shape_Identity::pages<dimensions...>(); }
	constexpr int books() const { return BC_Shape_Identity::books<dimensions...>(); }
	constexpr int libraries() const { return BC_Shape_Identity::libraries<dimensions...>(); }

	template<int dim_index>
	constexpr int dimension() const { return BC_Shape_Identity::dimension<dim_index, dimensions...>(); }

	int dimension(int index) const { return BC_Shape_Identity::libraries<index, dimensions...>(); }
};

#endif /* BC_TENSOR_SUPER_SHAPE_H_ */
