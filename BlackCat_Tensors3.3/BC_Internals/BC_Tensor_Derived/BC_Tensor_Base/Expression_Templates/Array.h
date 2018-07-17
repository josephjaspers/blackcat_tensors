/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "Expression_Base.h"

namespace BC {
namespace internal {
template<int dimension, class T, class mathlib>
struct Array : Tensor_Array_Base<Array<dimension, T, mathlib>, dimension>, public Shape<dimension> {

	using scalar_type = _scalar<T>;
	using math_lib = mathlib;

	__BCinline__ static constexpr int DIMS() { return dimension; }

	scalar_type* array = nullptr;

	Array(Shape<DIMS()> shape_, scalar_type* array_) : array(array_), Shape<DIMS()>(shape_) {}

	template<class U>
	Array(U param) : Shape<DIMS()>(param) { math_lib::initialize(array, this->size()); }

	template<class U>
	Array(U param, scalar_type* array_) : array(array_), Shape<DIMS()>(param) {}

	__BCinline__ const scalar_type* memptr() const { return array; }
	__BCinline__	   scalar_type* memptr()  	   { return array; }

	void destroy() {
		math_lib::destroy(array);
		array = nullptr;
	}

};
}
}

#endif /* SHAPE_H_ */
