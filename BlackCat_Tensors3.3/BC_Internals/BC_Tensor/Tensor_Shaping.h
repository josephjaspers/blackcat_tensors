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

namespace module {

template<class derived>
struct Tensor_Shaping {

	__BCinline__ static constexpr int DIMS() { return derived::DIMS(); }

	using operations  	= Tensor_Operations<derived>;
	using utility		= Tensor_Utility<derived>;

	using functor_type 	= functor_of<derived>;
	using scalar_type	= scalar_of<derived>;
	using mathlib_type 	= mathlib_of<derived>;

private:

	const auto& as_derived() const { return static_cast<const derived&>(*this); }
		  auto& as_derived()       { return static_cast<	  derived&>(*this); }

	const auto transpose_impl() const { return internal::unary_expression<functor_type, oper::transpose<mathlib_type>>(as_derived().internal()); }
	 	  auto transpose_impl() 	  { return internal::unary_expression<functor_type, oper::transpose<mathlib_type>>(as_derived().internal()); }

	 template<class internal_t>
	 auto tensor(internal_t internal) {
		 return Tensor_Base<internal_t>(internal);
	 }

public:

	const auto t() const { return tensor(transpose_impl()); }
		  auto t() 		 { return tensor(transpose_impl()); }

	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }


	struct range { int from, to; };

	const auto operator [] (range r) const { return slice(r.from, r.to); }
		  auto operator [] (range r) 		{ return slice(r.from, r.to); }

	const auto scalar(int i) const { return tensor(as_derived()._scalar(i)); }
		  auto scalar(int i) 	   { return tensor(as_derived()._scalar(i)); }

	const auto slice(int i) const  { return tensor(as_derived()._slice(i)); }
		  auto slice(int i) 	   { return tensor(as_derived()._slice(i)); }

	const auto slice(int from, int to) const  { return tensor(as_derived()._slice_range(from, to)); }
		  auto slice(int from, int to) 	   	  { return tensor(as_derived()._slice_range(from, to)); }


	const auto col(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	auto col(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return slice(i);
	}
	const auto row(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return tensor(as_derived()._row(i));
	}
	auto row(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return tensor(as_derived()._row(i));
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
