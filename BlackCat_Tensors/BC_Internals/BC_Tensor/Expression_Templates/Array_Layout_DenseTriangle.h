/*
 * Layout_DenseTriangle.h
 *
 *  Created on: Oct 1, 2018
 *      Author: joseph
 */

#ifndef BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_LAYOUT_DENSETRIANGLE_H_
#define BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_LAYOUT_DENSETRIANGLE_H_

#include "Array.h"

namespace BC {
namespace internal {

enum upLo {
	up = 0,
	lo = 1,
};

template<class T, int layout, upLo uplo, class alloc_t>
struct Array_Layout_DenseTriangle : Array<2, T, alloc_t, Array_Layout_DenseTriangle<T, layout, uplo, alloc_t>> {

	static constexpr int DIMS() { return 2; }
	static constexpr int ITERATOR() { return 2; }

	using traits = BLAS_traits<dense, triangle<uplo>, alloc_t>;

	using self =  Array_Layout_DenseTriangle;
	using parent =  Array<2, T, alloc_t, self>;

	Array_Layout_DenseTriangle(int length) {

	}

};

}
}



#endif /* BC_INTERNALS_BC_TENSOR_EXPRESSION_TEMPLATES_ARRAY_LAYOUT_DENSETRIANGLE_H_ */
