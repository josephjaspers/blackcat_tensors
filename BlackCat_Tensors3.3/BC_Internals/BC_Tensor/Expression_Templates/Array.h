/*
 * Shape.h
 *
 *  Created on: Jan 18, 2018
 *      Author: joseph
 */

#ifndef SHAPE_H_
#define SHAPE_H_

#include "Expression_Interface.h"

namespace BC {
namespace internal {
template<int dimension, class T, class mathlib>
struct Array : Array_Base<Array<dimension, T, mathlib>, dimension>, public Shape<dimension> {

	using scalar_t = T;
	using mathlib_t = mathlib;


	__BCinline__ static constexpr int DIMS() { return dimension; }

	scalar_t* array = nullptr;
	Array() = default;

	Array(Shape<DIMS()> shape_, scalar_t* array_) : array(array_), Shape<DIMS()>(shape_) {}

	template<class U,typename = std::enable_if_t<not std::is_base_of<BC_internal_interface<U>, U>::value>>
	Array(U param) : Shape<DIMS()>(param) { mathlib_t::initialize(array, this->size()); }

	template<class... integers>//CAUSES FAILURE WITH NVCC 9.2, typename = std::enable_if_t<MTF::is_integer_sequence<integers...>>>
	Array(integers... ints) : Shape<DIMS()>(ints...) {
		static_assert(MTF::is_integer_sequence<integers...>,"PARAMETER LIST MUST BE INTEGER_SEQUNCE");
		mathlib_t::initialize(array, this->size()); }

	template<class deriv_expr, typename = std::enable_if_t<std::is_base_of<BC_internal_interface<deriv_expr>, deriv_expr>::value>>
	Array(const deriv_expr& expr) : Shape<DIMS()>(static_cast<const deriv_expr&>(expr).inner_shape()) {
		mathlib_t::initialize(array, this->size());
		auto eval = binary_expression<Array<dimension, T, mathlib_t>, deriv_expr, oper::assign>(*this, static_cast<const deriv_expr&>(expr));
		BC::Evaluator<mathlib_t>::evaluate(eval);
	}

protected:
	template<class U> Array(U param, scalar_t* array_) : array(array_), Shape<DIMS()>(param) {}
	Array(scalar_t* array_) : array(array_) {}

public:
	__BCinline__ const scalar_t* memptr() const { return array; }
	__BCinline__	   scalar_t* memptr()  	    { return array; }

	void swap_array(Array& param) {
		std::swap(this->array, param.array);
	}

	void destroy() {
		mathlib_t::destroy(array);
		array = nullptr;
	}
};


//specialization for scalar --------------------------------------------------------------------------------------------------------
template<class T, class mathlib>
struct Array<0, T, mathlib> : Array_Base<Array<0, T, mathlib>, 0>, public Shape<0> {

	using scalar_t = T;
	using mathlib_t = mathlib;

	__BCinline__ static constexpr int DIMS() { return 0; }


	operator T () const {
		T value;
		mathlib::DeviceToHost(&value, array);
		return value;
	}

	scalar_t* array = nullptr;
	Array() {
		mathlib_t::initialize(array, this->size());
	}
	Array(Shape<DIMS()> shape_, scalar_t* array_) : array(array_), Shape<0>(shape_) {}

	template<class U>
	Array(U param) : Shape<DIMS()>(param) { mathlib_t::initialize(array, this->size()); }

	template<class U>
	Array(U param, scalar_t* array_) : array(array_), Shape<DIMS()>(param) {}

	__BCinline__ const auto& operator [] (int index) const { return array[0]; }
	__BCinline__ 	   auto& operator [] (int index) 	   { return array[0]; }

	template<class... integers> __BCinline__ 	   auto& operator () (integers... ints) {
		return array[0];
	}
	template<class... integers> __BCinline__ const auto& operator () (integers... ints) const {
		return array[0];
	}

	__BCinline__ const scalar_t* memptr() const { return array; }
	__BCinline__	   scalar_t* memptr()  	   { return array; }

	void destroy() {
		mathlib_t::destroy(array);
		array = nullptr;
	}

};
}
template<class T> struct isPrimaryArray { static constexpr bool conditional = false; };
template<int d, class T, class ml> struct isPrimaryArray<internal::Array<d,T,ml>> { static constexpr bool conditional = true; };
template<class T> static constexpr bool is_array_core() { return isPrimaryArray<T>::conditional; }


template<class T>
struct BC_array_move_assignable_overrider<T, std::enable_if_t<is_array_core<T>()>> {
	static constexpr bool boolean = true;
};
//template<class T>
//struct BC_array_copy_assignable_overrider<T, std::enable_if_t<is_array_core<T>()>> {
//	static constexpr bool boolean = true;
//};
template<class T>
struct BC_array_move_constructible_overrider<T, std::enable_if_t<is_array_core<T>()>> {
	static constexpr bool boolean = true;
};
template<class T>
struct BC_array_copy_constructible_overrider<T, std::enable_if_t<is_array_core<T>()>> {
	static constexpr bool boolean = true;
};


}

#endif /* SHAPE_H_ */
