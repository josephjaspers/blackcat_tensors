/*
 * Tensor_Shaping.h
 *
 *  Created on: May 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SHAPING_H_
#define TENSOR_SHAPING_H_

#include "Tensor_Shaping_Static.h"

namespace BC {

template<class> class Tensor_Base;

namespace Base {
template<class derived>
struct Tensor_Shaping {

	__BCinline__ static constexpr int DIMS() { return derived::DIMS(); }

	using operations  	= Tensor_Operations<derived>;
//	using initializer 	= Tensor_Initializer<derived>;
	using utility		= Tensor_Utility<derived>;

	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

private:

	const auto& as_derived() const { return static_cast<const derived&>(*this); }
		  auto& as_derived()       { return static_cast<	  derived&>(*this); }

	const auto transpose_impl() const { return internal::unary_expression<functor_type, oper::transpose<mathlib_type>>(as_derived().internal()); }
	 	  auto transpose_impl() 	  { return internal::unary_expression<functor_type, oper::transpose<mathlib_type>>(as_derived().internal()); }

public:

	const auto t() const { return Tensor_Base<decltype(transpose_impl())>(transpose_impl()); }
		  auto t() 		 { return Tensor_Base<decltype(transpose_impl())>(transpose_impl()); }

	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	const auto scalar(int i) const { return Tensor_Base<internal::Array_Scalar<functor_type>>(as_derived()._scalar(i)); }
		  auto scalar(int i) 	   { return Tensor_Base<internal::Array_Scalar<functor_type>>(as_derived()._scalar(i)); }

	const auto slice(int i) const  { return Tensor_Base<decltype(as_derived()._slice(0))>(as_derived()._slice(i)); }
		  auto slice(int i) 	   { return Tensor_Base<decltype(as_derived()._slice(0))>(as_derived()._slice(i)); }

	const auto col(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	auto col(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}

	const auto operator() (int i) const { return scalar(i); }
		  auto operator() (int i) 	    { return scalar(i); }

	const auto& operator() () const { return *this; }
		  auto& operator() () 	    { return *this; }

	template<class... integers> const auto operator() (int i, integers... ints) const  {
		static_assert(MTF::is_integer_sequence<integers...>, "operator()(integers...) -> PARAMS MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

	template<class... integers> 	  auto operator() (int i, integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "operator()(integers...) -> PARAMS MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}


	//THIS IS THE CURRIED CHUNK LAMBDA, WE MUST USE AN ACTUAL CLASS TO ACT AS A LAMDA AS CUDA COMPILER IS IFFY WITH LAMBDA

	struct CHUNK {

		const derived& tensor;
		int location;
		CHUNK(const derived& tensor_, int index_) : tensor(tensor_), location(index_) {}

		template<class... integers>
		const auto operator () (integers... shape_dimensions) const {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename internal::Array_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = Tensor_Base<chunk_type>;

			return type(tensor.internal()._chunk(location, shape_dimensions...));
		}
		template<class... integers>
		auto operator () (integers... shape_dimensions) {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename internal::Array_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = Tensor_Base<chunk_type>;

			return type(tensor.internal()._chunk(location, shape_dimensions...));
		}
	};
};

}	//END OF BASE NAMESPACE
//-----------------------------------------------THESE ARE THE "CURRIED FUNCTIONS"-------------------------------------------------"

}

#endif /* TENSOR_SHAPING_H_ */
