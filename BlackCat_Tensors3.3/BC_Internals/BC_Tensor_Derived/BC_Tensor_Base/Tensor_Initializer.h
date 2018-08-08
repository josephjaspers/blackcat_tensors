///*
// * Tensor_Initializer.h
// *
// *  Created on: Mar 3, 2018
// *      Author: joseph
// */
//
//#ifndef TENSOR_INITIALIZER_H_
//#define TENSOR_INITIALIZER_H_
//
//#include "Expression_Templates/Array.h"
//#include "Expression_Templates/Array_Scalar.h"
//#include "Expression_Templates/Array_Slice.h"
//#include "Expression_Templates/Array_Slice_Complex.h"
//
//#include "Expression_Templates/Array_Chunk.h"
//#include "Expression_Templates/Array_Reshape.h"
//
////-------------------------------------SPECIALIZATION FOR EXPRESSION TENSORS OR TENSORS OF NON_OWNERSHIP/CREATION-------------------------------------//
//
//namespace BC {
//
//template<class T> class Tensor_Base;
//
//namespace Base {
//template<class derived>
//class Tensor_Initializer :  public _functor<derived> {
//
//	using self 			= Tensor_Initializer<derived>;
//	using parent 		= _functor<derived>;
//
//public:
//
//	 Tensor_Initializer(const  derived&  tensor) : parent(		    tensor.internal()) {}
//	 Tensor_Initializer(	   derived&& tensor) : parent(std::move(tensor.internal())){}
//
//	template<class... params> explicit Tensor_Initializer(const params&...  p) : parent(p...) {}
//	template<class... params> explicit Tensor_Initializer(		params&&... p) : parent(p...) {}
//
//	const parent& internal() const { return static_cast<const parent&>(*this); }
//	 	  parent& internal() 	   { return static_cast<	  parent&>(*this); }
//
//};
////-------------------------------------SPECIALIZATION FOR TENSORS THAT CONTROL / DELETE THEIR ARRAY-------------------------------------//
//
//template<int x, class T, class ml>
//struct Tensor_Initializer<Tensor_Base<internal::Array<x, T, ml>>> : public  internal::Array<x, T, ml> {
//
//	using self 		= Tensor_Initializer<Tensor_Base<internal::Array<x, T, ml>>>;
//	using derived 	= Tensor_Base<internal::Array<x,T,ml>>;
//	using parent 		= internal::Array<x,T,ml>;
//
//private:
//
//	const auto& as_derived() const { return static_cast<const derived&>(*this); }
//		  auto& as_derived() 	   { return static_cast<	  derived&>(*this); }
//
//public:
//
//	 const parent& internal() const { return static_cast<const parent&>(*this); }
//	 	   parent& internal() 	  	{ return static_cast<	   parent&>(*this); }
//
//	Tensor_Initializer(const derived& tensor)  : parent(tensor.inner_shape()) { this->as_derived() = tensor; }
//	Tensor_Initializer(		 derived&& tensor) : parent(std::move(tensor.internal())) {}
//	template<class... params> explicit Tensor_Initializer(const params&...  p) : parent(p...) {}
//	template<class... params> explicit Tensor_Initializer(		params&&... p) : parent(p...) {}
//
//	Tensor_Initializer() = default;
//
//	template<class U>
//	Tensor_Initializer(const Tensor_Initializer<U>&  tensor) : parent(tensor.inner_shape()) {
//		this->as_derived() = tensor;
//	}
//
//	template<class U>
//	Tensor_Initializer(const Tensor_Base<U>&  tensor) : parent(tensor.inner_shape()) {
//		this->as_derived() = tensor;
//	}
//};
//}
//}
//
//
//
//#endif /* TENSOR_INITIALIZER_H_ */
