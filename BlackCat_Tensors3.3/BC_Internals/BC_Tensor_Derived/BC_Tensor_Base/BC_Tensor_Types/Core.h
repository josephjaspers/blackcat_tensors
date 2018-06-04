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
template<class T>
struct Core : Tensor_Core_Base<Core<T>, _dimension_of<T>>{

	using scalar_type = _scalar<T>;
	using math_lib = _mathlib<T>;

	__BCinline__ static constexpr int DIMS() { return _dimension_of<T>; }

	scalar_type* array = nullptr;
	Shape<DIMS()> shape;

	Core(Shape<DIMS()> shape_, scalar_type* array_) : array(array_), shape(shape_) {}

	template<class U>
	Core(U param) : shape(param) { math_lib::initialize(array, this->size()); }

	template<class U>
	Core(U param, scalar_type* array_) : array(array_), shape(param) {}


	__BCinline__ const auto inner_shape() const { return shape.is(); }
	__BCinline__ const auto outer_shape() const { return shape.os(); }
	__BCinline__ const scalar_type* getIterator() const { return array; }
	__BCinline__	   scalar_type* getIterator()  	    { return array; }


	template<class... integers> void resetShape(integers... ints)  {
		int sz = this->size();
		shape = Shape<DIMS()>(ints...);

		if (sz != this->size()) {
			math_lib::destroy(array);
			math_lib::initialize(array, this->size());
		}
	}
	void resetShape(Shape<DIMS()> new_shape)  {
		if (shape.size() == new_shape.size()) {
			shape = new_shape;
		} else {
			shape = new_shape;
			math_lib::destroy(array);
			math_lib::initialize(array, this->size());
		}
	}

	__BCinline__ auto shift(int i ) {
		array += i;
		return *this;
	}

	void destroy() {
		math_lib::destroy(array);
		array = nullptr;
	}

};
}
}

#endif /* SHAPE_H_ */
