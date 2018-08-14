/*
 * Array_View.h
 *
 *  Created on: Aug 10, 2018
 *      Author: joseph
 */

#ifndef ARRAY_VIEW_H_
#define ARRAY_VIEW_H_

namespace BC{
namespace internal {

template<int x, class scalar, class allocator_t>
struct Array_View : Array<x, scalar, allocator_t> {

	using scalar_t = scalar;
	using mathlib_t = allocator_t;
	using Array::Array;

	void destroy() {}
};
}
}

#endif /* ARRAY_VIEW_H_ */
