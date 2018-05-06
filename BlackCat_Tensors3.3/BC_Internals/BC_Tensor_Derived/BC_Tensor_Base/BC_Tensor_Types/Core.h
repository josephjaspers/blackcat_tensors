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

template<class T>
struct Core : Core_Base<Core<T>, _rankOf<T>>{

	__BCinline__ static constexpr int DIMS() { return _rankOf<T>; }
	__BCinline__ static constexpr int PARENT_DIMS() { return _rankOf<T>; }
	__BCinline__ static constexpr int LAST() { return DIMS() - 1;}

	using self = Core<T>;
	using scalar_type = _scalar<T>;
	using Mathlib = _mathlib<T>;
	using slice_type = Tensor_Slice<self>;

	scalar_type* array = nullptr;
	int* is = nullptr;
	int* os = nullptr;

	Core() = default;
	Core(const Core&) = default;
	Core(Core&&) = default;
	Core& operator = (const Core& ) = default;
	Core& operator = (	   Core&&) = default;

	template<class U>
	Core(const U& param) {
		static_assert(is_shape<U>, "NON_SHAPE DETECTED AS INITIALIZATION OF TENSOR SHAPE");
		Mathlib::unified_initialize(is, DIMS());
		Mathlib::unified_initialize(os, DIMS());

		if (DIMS() > 0) {

			is[0] = param[0];
			os[0] = is[0];
			for (int i = 1; i < DIMS(); ++i) {
				is[i] = param[i];
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, this->size());
	}

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	__BCinline__
	__BCinline__ const auto slice(int i) const { return slice_type(&array[slice_index(i)],*this); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&array[slice_index(i)],*this); }

	__BCinline__ const scalar_type* getIterator() const { return array; }
	__BCinline__	   scalar_type* getIterator()  	    { return array; }

	template<class... integers, int dim = 0>
	void resetShape(integers... ints)  {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		this->init<0>(ints...);
		Mathlib::destroy(array);
		Mathlib::initialize(array, this->size());
	}


	__BCinline__
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return outerShape()[LAST() - 1] * i;
	}


	template<int d> __BCinline__
	void init() {}

	template<int dim, class... integers> __BCinline__
	void init(int front, integers... ints) {

		//NVCC gives warning if you convert the static_assert into a one-liner
		static constexpr bool intList = MTF::is_integer_sequence<integers...>;
		static_assert(intList, "MUST BE INTEGER LIST");

		is[dim] = front;

		if (dim > 0)
			os[dim] = front * os[dim - 1];
		else
			os[0] = front;

		if (dim != DIMS() - 1) {
			init<(dim + 1 < DIMS() ? dim + 1 : DIMS())>(ints...);
		}
	}

	void destroy() {
		Mathlib::destroy(array);
		Mathlib::destroy(is);
		Mathlib::destroy(os);
		array = nullptr;
		is = nullptr;
		os = nullptr;
	}

};
}

#endif /* SHAPE_H_ */
