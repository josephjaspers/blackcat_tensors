/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "Expression_Base.h"
#include "Shape.h"
namespace BC {

template<class T>
struct Core : Core_Base<Core<T>, _dimension_of<T>>{

	using scalar_type = _scalar<T>;
	using math_lib = _mathlib<T>;

	__BCinline__ static constexpr int DIMS() { return _dimension_of<T>; }
	__BCinline__ static constexpr int PARENT_DIMS() { return _dimension_of<T>; }
	__BCinline__ static constexpr int last() { return DIMS() - 1;}

	scalar_type* array = nullptr;
	Shape<DIMS()> shape;

	template<class U>
	Core(const U& param) : shape(param) {
		static_assert(is_shape<U>, "NON_SHAPE DETECTED AS INITIALIZATION OF TENSOR SHAPE");
		math_lib::initialize(array, this->size());
	}

	Core() = default;


	__BCinline__ const auto innerShape() const { return shape.is(); }
	__BCinline__ const auto outerShape() const { return shape.os(); }
	__BCinline__ const scalar_type* getIterator() const { return array; }
	__BCinline__	   scalar_type* getIterator()  	    { return array; }

	template<class... integers, int dim = 0>
	void resetShape(integers... ints)  {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");

		int sz = this->size();
		shape = Shape<DIMS()>(ints...);

		if (sz != this->size()) {
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

#endif /* SHAPE_H_ */
