/*
 * Tensor_Shaping.h
 *
 *  Created on: May 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SHAPING_H_
#define TENSOR_SHAPING_H_

#include "Tensor_Functions/Shaping.h"

namespace BC {
namespace Base {
template<class derived>
struct Tensor_Shaping {

	__BCinline__ static constexpr int DIMS() { return derived::DIMS(); }

	using operations  	= Tensor_Operations<derived>;
	using initializer 	= Tensor_Initializer<derived>;
	using utility		= Tensor_Utility<derived>;

	using functor_type 	= _functor<derived>;
	using scalar_type	= _scalar<derived>;
	using mathlib_type 	= _mathlib<derived>;

private:

	const auto& as_derived() const { return static_cast<const derived&>(*this); }
		  auto& as_derived()       { return static_cast<	  derived&>(*this); }

public:

	//-------const reshape (using int list)
	template<class... integers>
	const auto reshape_impl(integers... ints) const {
		using internal = decltype(std::declval<derived>().internal().reshape(ints...));
		static constexpr int tensor_dim =  sizeof...(integers);
		using type = typename tensor_of<tensor_dim>::template type<internal, mathlib_type>;
		return type(this->internal().reshape(ints...));
	}
	//-------const reshape (using shape object)
	template<int dims>
	const auto reshape_impl(Shape<dims> shape) const {
		using internal = decltype(std::declval<derived>().internal().reshape(shape));
		using type = typename tensor_of<dims>::template type<internal, mathlib_type>;
		return type(this->internal().reshape(shape));
	}
	//-------non-const reshape (using int list)
	template<class... integers>
	 auto reshape_impl(integers... ints)  {
		using internal = decltype(std::declval<derived>().internal().reshape(ints...));
		static constexpr int tensor_dim =  sizeof...(integers);
		using type = typename tensor_of<tensor_dim>::template type<internal, mathlib_type>;
		return type(as_derived().internal().reshape(ints...));
	}
	//-------non-const reshape (using shape object)
	template<int dims>
	 auto reshape_impl(Shape<dims> shape)  {
		using internal = decltype(std::declval<derived>().internal().reshape(shape));
		using type = typename tensor_of<dims>::template type<internal, mathlib_type>;
		return type(as_derived().internal().reshape(shape));
	}
	template<class... integers>
	auto chunk_impl(integers... ints) {
		int location = as_derived().internal().dims_to_index_reverse(ints...);
		return CHUNK(as_derived(), location);
	}
	template<class... integers>
	const auto chunk_impl(integers... ints) const {
		int location = as_derived().internal().dims_to_index_reverse(ints...);
		return CHUNK(as_derived(), location);
	}


private:
	const auto slice_impl(int i) const { return as_derived().internal().slice(i); }
		  auto slice_impl(int i) 	  { return as_derived().internal().slice(i);  }

	const auto scalar_impl(int i) const { return as_derived().internal().scalar(i); }
		  auto scalar_impl(int i)	    { return as_derived().internal().scalar(i);  }

	const auto row_impl(int i) const { return as_derived().internal().row(i); }
		  auto row_impl(int i)	     { return as_derived().internal().row(i); }

	const auto transpose_impl() const { return internal::unary_expression<functor_type, oper::transpose<mathlib_type>>(as_derived().internal()); }
	 	  auto transpose_impl() 	  { return internal::unary_expression<functor_type, oper::transpose<mathlib_type>>(as_derived().internal()); }

public:
	const auto t() const { return BC::tensor_of_t<DIMS(), decltype(transpose_impl()), mathlib_type>(transpose_impl()); }
		  auto t() 		 { return BC::tensor_of_t<DIMS(), decltype(transpose_impl()), mathlib_type>(transpose_impl()); }

	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	const auto scalar(int i) const { return tensor_of<0>::type<internal::Array_Scalar<functor_type>, mathlib_type>(scalar_impl(i)); }
		  auto scalar(int i) 	   { return tensor_of<0>::type<internal::Array_Scalar<functor_type>, mathlib_type>(scalar_impl(i)); }

	const auto slice(int i) const {
		static_assert(DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename tensor_of<DIMS()>::template slice<decltype(slice_impl(0)), mathlib_type>(slice_impl(i)); }

		  auto slice(int i) 	  {
		static_assert(derived::DIMS() > 0, "SCALAR SLICE IS NOT DEFINED");
		return typename tensor_of<DIMS()>::template slice<decltype(slice_impl(0)), mathlib_type>(slice_impl(i)); }

	const auto row(int i) const {
		static_assert(DIMS() == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename tensor_of<1>::template type<decltype(row_impl(0)), mathlib_type>(row_impl(i));
	}
		  auto row(int i) 		{
		static_assert(DIMS() == 2, "MATRIX ROW ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return typename tensor_of<1>::template type<decltype(row_impl(0)), mathlib_type>(row_impl(i));
	}
	const auto col(int i) const {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}
		 auto col(int i) {
		static_assert(DIMS() == 2, "MATRIX COL ONLY AVAILABLE TO MATRICES OF ORDER 2");
		return (*this)[i];
	}


	const auto operator() (int i) const { return scalar(i); }
		  auto operator() (int i) 	    { return scalar(i); }

	const auto& operator() () const { return *this; }
		  auto& operator() () 	    { return *this; }

	template<class... integers> const auto operator() (int i, integers... ints) const  {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

	template<class... integers> 	  auto operator() (int i, integers... ints) {
		static_assert(MTF::is_integer_sequence<integers...>, "MUST BE INTEGER LIST");
		return (*this)[i](ints...);
	}

//	template<class... integers>
//	void resize(integers... ints) {
//		as_derived().internal().resize(ints...);
//	}
	//THIS IS THE CURRIED CHUNK LAMBDA, WE MUST USE AN ACTUAL CLASS TO ACT AS A LAMDA AS CUDA COMPILER IS IFFY WITH LAMBDA
	struct CHUNK {

		const derived& tensor;
		int location;
		CHUNK(const derived& tensor_, int index_) : tensor(tensor_), location(index_) {}

		template<class... integers>
		const auto operator () (integers... shape_dimensions) const {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename internal::Array_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = tensor_of_t<tensor_dimension, chunk_type, mathlib_type>;

			return type(tensor.internal().chunk(location, shape_dimensions...));
		}
		template<class... integers>
		auto operator () (integers... shape_dimensions) {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename internal::Array_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = tensor_of_t<tensor_dimension, chunk_type, mathlib_type>;

			return type(tensor.internal().chunk(location, shape_dimensions...));
		}
	};
};

}	//END OF BASE NAMESPACE
//-----------------------------------------------THESE ARE THE "CURRIED FUNCTIONS"-------------------------------------------------"

}

#endif /* TENSOR_SHAPING_H_ */
