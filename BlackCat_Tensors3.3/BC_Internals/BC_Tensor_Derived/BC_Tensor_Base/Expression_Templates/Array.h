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
struct Array : Tensor_Array_Base<Array<dimension, T, mathlib>, dimension>{

	using scalar_type = _scalar<T>;
	using math_lib = mathlib;

	__BCinline__ static constexpr int DIMS() { return dimension; }

	scalar_type* array = nullptr;
	Shape<DIMS()> shape;

	Array(Shape<DIMS()> shape_, scalar_type* array_) : array(array_), shape(shape_) {}

	template<class U>
	Array(U param) : shape(param) { math_lib::initialize(array, this->size()); }

	template<class U>
	Array(U param, scalar_type* array_) : array(array_), shape(param) {}


	__BCinline__ const auto inner_shape() const { return shape.inner_shape(); }
	__BCinline__ const auto outer_shape() const { return shape.outer_shape(); }
	__BCinline__ const scalar_type* memptr() const { return array; }
	__BCinline__	   scalar_type* memptr()  	    { return array; }


	template<class... integers> void resize(integers... ints)  {
		int sz = this->size();
		shape = Shape<DIMS()>(ints...);

		if (sz != this->size()) {
			math_lib::destroy(array);
			math_lib::initialize(array, this->size());
		}
	}
	void resize(Shape<DIMS()> new_shape)  {
		if (shape.size() == new_shape.size()) {
			shape = new_shape;
		} else {
			shape = new_shape;
			math_lib::destroy(array);
			math_lib::initialize(array, this->size());
		}
	}
	void destroy() {
		math_lib::destroy(array);
		array = nullptr;
	}

};
}
}

#endif /* SHAPE_H_ */
