

/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "BC_Expressions/Expression_Base.h"
#include "Tensor_Core_Slice.h"
#include "Tensor_Core_Scalar.h"
#include "Tensor_Core_RowVector.h"
#include "Tensor_Core_Reshape.h"
#include "Tensor_Core_Interface.h"
#include "Determiners.h"
namespace BC {

template<class T>
struct Tensor_Core : Tensor_Core_Base<Tensor_Core<T>, _rankOf<T>>{

	__BCinline__ static constexpr int DIMS() { return _rankOf<T>; }
	__BCinline__ static constexpr int LAST() { return DIMS() - 1;}

	using self = Tensor_Core<T>;
	using scalar_type = _scalar<T>;
	using Mathlib = _mathlib<T>;
	using slice_type = Tensor_Slice<self>;

	scalar_type* array = nullptr;
	int* is = nullptr;
	int* os = nullptr;

	Tensor_Core() = default;

	Tensor_Core(const Tensor_Core& cpy) {
		Mathlib::unified_initialize(is, DIMS());
		Mathlib::unified_initialize(os, DIMS());

		for (int i  = 0; i < DIMS(); ++i) {
			is[i] = cpy.is[i];
			os[i] = cpy.os[i];
		}

		Mathlib::initialize(array, this->size());
		Mathlib::copy(array, cpy.array, this->size());
	}
	Tensor_Core(Tensor_Core&& cpy) {
		is = cpy.is;
		os = cpy.os;
		array = cpy.array;

		cpy.is = nullptr;
		cpy.os = nullptr;
		cpy.array = nullptr;
	}

	Tensor_Core(std::vector<int> param) {
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
	Tensor_Core(const int* param) {
		Mathlib::unified_initialize(is, DIMS());
		Mathlib::unified_initialize(os, DIMS());

		if (DIMS() > 0) {
			Mathlib::HostToDevice(is, &param[0], DIMS());

			os[0] = is[0];
			for (int i = 1; i < DIMS(); ++i) {
				os[i] = os[i - 1] * is[i];
			}
		}
		Mathlib::initialize(array, this->size());
	}

	__BCinline__ const auto innerShape() const { return is; }
	__BCinline__ const auto outerShape() const { return os; }

	//this needs to be overridden in tensor core (tensor_core can't return itself, doesn't have a "PARENT" memeber)
	__BCinline__ const auto slice(int i) const { return slice_type(&array[slice_index(i)],*this); }
	__BCinline__	   auto slice(int i) 	   { return slice_type(&array[slice_index(i)],*this); }

	__BCinline__ const scalar_type* getIterator() const { return array; }
	__BCinline__ 	   scalar_type* getIterator()  	    { return array; }




	template<class... integers, int dim = 0>
	void resetShape(integers... ints)  {
		this->init<0>(ints...);
		Mathlib::destroy(array);
		Mathlib::initialize(array, this->size());
	}

	//----------------------------------implementation for slice------------------------------------//
	int slice_index(int i) const {
		if (DIMS() == 0)
			return 0;
		else if (DIMS() == 1)
			return i;
		else
			return outerShape()[LAST() - 1] * i;
	}


	//---------------------------------implementation for reset shape-------------------------------------//
	template<int d>
	void init() {/*I AM FRIEND I HELP COMPILE DONT DELETE ME */}

	template<int dim, class... integers> __BCinline__
	void init(int front, integers... ints) {
		is[dim] = front;
		if (dim > 0)
			os[dim] = front * os[dim - 1];
		else
			os[0] = front;

		if (dim != DIMS() - 1) {
			init<(dim + 1 < DIMS() ? dim + 1 : DIMS())>(ints...);
		}
	}



};
}

#endif /* SHAPE_H_ */
