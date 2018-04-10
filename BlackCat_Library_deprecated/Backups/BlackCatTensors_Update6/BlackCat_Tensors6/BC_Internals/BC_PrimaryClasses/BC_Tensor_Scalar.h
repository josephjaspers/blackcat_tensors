/*
 * BC_Tensor_Scalar.h
 *
 *  Created on: Dec 18, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SCALAR_H_
#define BC_TENSOR_SCALAR_H_

#include "../BlackCat_Internal_GlobalUnifier.h"

namespace BC {
template<class T, class lib>
class Scalar : public Tensor_Mathematics_Head<T, Scalar<T, lib>, lib, Static_Inner_Shape<1>, Static_Outer_Shape<0>> {

	using functor_type = typename Tensor_FunctorType<T>::type;
	using parent_class = Tensor_Mathematics_Head<T, Scalar<T, lib>, lib, Static_Inner_Shape<1>, Static_Outer_Shape<0>>;
	using grandparent_class = typename parent_class::grandparent_class;
	using this_type = Scalar<T, lib>;
	static constexpr Tensor_Shape RANK = SCALAR;

public:
	Scalar() = delete;
	Scalar(T* t) : parent_class(t) {}

	template<class U> Scalar<T, lib>& operator =(const Scalar<U, lib>& v) {

		static_assert(grandparent_class::ASSIGNABLE, "Scalar<T, lib> of type T is non assignable (use Eval() to evaluate expression-tensors)");
		lib::set_heap(this->data(), v.data());
		return *this;
	}

	template<class U> Scalar<T, lib>& operator =(U scalar) {

		static_assert(grandparent_class::ASSIGNABLE, "Scalar<T, lib> of type T is non assignable (use Eval() to evaluate expression-tensors)");
		lib::set_stack(this->data(), scalar);
		return *this;
	}

	//specializations for Scalar by Scalar
	template<typename U> T operator + (const Scalar<U, lib>& scalar) { return this->array[0] + scalar.array[0]; }
	template<typename U> T operator - (const Scalar<U, lib>& scalar) { return this->array[0] - scalar.array[0]; }
	template<typename U> T operator / (const Scalar<U, lib>& scalar) { return this->array[0] / scalar.array[0]; }
	template<typename U> T operator % (const Scalar<U, lib>& scalar) { return this->array[0] * scalar.array[0]; }
	template<typename U> T operator * (const Scalar<U, lib>& scalar) { return this->array[0] * scalar.array[0]; }

	//This works because scalars cannot be binary expression
	T operator + (const T scalar) { return this->array[0] + scalar; }
	T operator - (const T scalar) { return this->array[0] - scalar; }
	T operator / (const T scalar) { return this->array[0] / scalar; }
	T operator % (const T scalar) { return this->array[0] * scalar; }
	T operator * (const T scalar) { return this->array[0] * scalar; }

	//Specializations for Scalar by Tensor
	template<typename U, class deriv,  class is, class os, class ld>
	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::add, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>::type  //return object
	operator + (const Tensor_Mathematics_Head<U, deriv, lib, is, os>& tensor) {
		return 	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::add, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>(this->array, tensor.array);  //return object
	}
	template<typename U, class deriv,  class is, class os, class ld>
	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::sub, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>::type  //return object
	operator - (const Tensor_Mathematics_Head<U, deriv, lib, is, os>& tensor) {
		return 	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::sub, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>(this->array, tensor.array);  //return object
	}

	template<typename U, class deriv,  class is, class os, class ld>
	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::div, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>::type  //return object
	operator / (const Tensor_Mathematics_Head<U, deriv, lib, is, os>& tensor) {
		return 	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::div, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>(this->array, tensor.array);  //return object
	}

	template<typename U, class deriv,  class is, class os, class ld>
	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::mul, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>::type  //return object
	operator % (const Tensor_Mathematics_Head<U, deriv, lib, is, os>& tensor) {
		return 	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::mul, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>(this->array, tensor.array);  //return object
	}

	///Alternate
	template<typename U, class deriv,  class is, class os, class ld>
	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::mul, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>::type  //return object
	operator * (const Tensor_Mathematics_Head<U, deriv, lib, is, os>& tensor) {
		return 	typename BC_Substitute_Type::Identity<binary_expression_scalar_L<T, BC::mul, functor_type, typename Tensor_Mathematics_Head<U, deriv, lib, is, ld>::functor_type>, this_type>(this->array, tensor.array);  //return object
	}

};
//
//
//template<class T, class lib>
//class Scalar : public Tensor_Mathematics_Head<T, Scalar<T, lib>, lib, Inner_Shape<1>, Outer_Shape<0>> {
//
//	using functor_type = typename Tensor_FunctorType<T>::type;
//	using parent_class = Tensor_Mathematics_Head<T, Scalar<T, lib>, lib, Inner_Shape<1>, Outer_Shape<0>>;
//	using grandparent_class = typename parent_class::grandparent_class;
//	using this_type = Scalar<T, lib>;
//
//public:
//	Scalar() = delete;
//	Scalar(T* t) : parent_class(t) {}
//
//	template<class U> Scalar<T, lib>& operator =(const Scalar<U, lib>& v) {
//
//		static_assert(grandparent_class::ASSIGNABLE, "Scalar<T, lib> of type T is non assignable (use Eval() to evaluate expression-tensors)");
//		lib::set_heap(this->data(), v.data());
//		return *this;
//	}
//
//	template<class U> Scalar<T, lib>& operator =(U scalar) {
//
//		static_assert(grandparent_class::ASSIGNABLE, "Scalar<T, lib> of type T is non assignable (use Eval() to evaluate expression-tensors)");
//		lib::set_stack(this->data(), scalar);
//		return *this;
//	}
//
//};
}
#endif /* BC_TENSOR_SCALAR_H_ */
