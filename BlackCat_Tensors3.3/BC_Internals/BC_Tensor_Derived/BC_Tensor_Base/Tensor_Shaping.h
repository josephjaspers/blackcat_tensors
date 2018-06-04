/*
 * Tensor_Shaping.h
 *
 *  Created on: May 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SHAPING_H_
#define TENSOR_SHAPING_H_

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

	const auto& as_derived() const { return static_cast<derived&>(*this); }
	auto& as_derived() { return static_cast<derived&>(*this); }

public:

	//-------const reshape (using int list)
	template<class... integers>
	const auto lazy_reshape(integers... ints) const {
		using internal = decltype(std::declval<derived>().data().reshape(ints...));
		static constexpr int tensor_dim =  sizeof...(integers);
		using type = typename tensor_of<tensor_dim>::template type<internal, mathlib_type>;
		return type(this->data().reshape(ints...));
	}
	//-------const reshape (using shape object)
	template<int dims>
	const auto lazy_reshape(Shape<dims> shape) const {
		using internal = decltype(std::declval<derived>().data().reshape(shape));
		using type = typename tensor_of<dims>::template type<internal, mathlib_type>;
		return type(this->data().reshape(shape));
	}
	//-------non-const reshape (using int list)
	template<class... integers>
	 auto lazy_reshape(integers... ints)  {
		using internal = decltype(std::declval<derived>().data().reshape(ints...));
		static constexpr int tensor_dim =  sizeof...(integers);
		using type = typename tensor_of<tensor_dim>::template type<internal, mathlib_type>;
		return type(this->as_derived().data().reshape(ints...));
	}
	//-------non-const reshape (using shape object)
	template<int dims>
	 auto lazy_reshape(Shape<dims> shape)  {
		using internal = decltype(std::declval<derived>().data().reshape(shape));
		using type = typename tensor_of<dims>::template type<internal, mathlib_type>;
		return type(this->as_derived().data().reshape(shape));
	}
	template<class... integers>
	auto lazy_chunk(integers... ints) {
		int location = this->as_derived().data().dims_to_index_reverse(ints...);
		return CHUNK(this->as_derived(), location);
	}
	template<class... integers>
	const auto lazy_chunk(integers... ints) const {
		int location = this->as_derived().data().dims_to_index_reverse(ints...);
		return CHUNK(this->as_derived(), location);
	}


private:
	const auto slice_impl(int i) const { return this->black_cat_array.slice(i); }
		  auto slice_impl(int i) 	  { return this->black_cat_array.slice(i);  }

	const auto scalar_impl(int i) const { return this->as_derived().black_cat_array.scalar(i); }
		  auto scalar_impl(int i)	    { return this->as_derived().black_cat_array.scalar(i);  }

	const auto row_impl(int i) const { return this->black_cat_array.row(i); }
		  auto row_impl(int i)	     { return this->black_cat_array.row(i); }
public:
	const auto operator [] (int i) const { return slice(i); }
		  auto operator [] (int i) 		 { return slice(i); }

	const auto scalar(int i) const { return tensor_of<0>::type<internal::Tensor_Scalar<functor_type>, mathlib_type>(scalar_impl(i)); }
		  auto scalar(int i) 	   { return tensor_of<0>::type<internal::Tensor_Scalar<functor_type>, mathlib_type>(scalar_impl(i)); }

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

	template<class... integers>
	void resize(integers... ints) {
		this->as_derived().black_cat_array.resetShape(ints...);
	}
	//THIS IS THE CURRIED CHUNK LAMBDA, WE MUST USE AN ACTUAL CLASS TO ACT AS A LAMDA AS CUDA COMPILER IS IFFY WITH LAMBDA
	struct CHUNK {

		const derived& tensor;
		int location;
		CHUNK(const derived& tensor_, int index_) : tensor(tensor_), location(index_) {}

		template<class... integers>
		const auto operator () (integers... shape_dimensions) const {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename internal::Tensor_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = tensor_of_t<tensor_dimension, chunk_type, mathlib_type>;

			return type(tensor.data().chunk(location, shape_dimensions...));
		}
		template<class... integers>
		auto operator () (integers... shape_dimensions) {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename internal::Tensor_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = tensor_of_t<tensor_dimension, chunk_type, mathlib_type>;

			return type(tensor.data().chunk(location, shape_dimensions...));
		}
	};
};

}	//END OF BASE NAMESPACE
//-----------------------------------------------THESE ARE THE "CURRIED FUNCTIONS"-------------------------------------------------"
	template<class> class Tensor_Base;

	template<class T> __BC_host_inline__
	const auto reshape(const Tensor_Base<T>& tensor) {
		return [&](auto... integers) { return tensor.lazy_reshape(integers...); };
	}
	template<class T> __BC_host_inline__
	auto reshape(Tensor_Base<T>& tensor) {
		return [&](auto... integers) { return tensor.lazy_reshape(integers...); };
	}

	template<class T> __BC_host_inline__
	auto chunk(Tensor_Base<T>& tensor) {
		return [&](auto... integers) {
			return tensor.lazy_chunk(integers...);
		};
	}
	template<class T> __BC_host_inline__
	const auto chunk(const Tensor_Base<T>& tensor) {
		return [&](auto... integers) {
			return tensor.lazy_chunk(integers...);
		};
	}

	template<class T> __BC_host_inline__
	const auto reshape(const Tensor_Base<T>&& tensor) {
		return [&](auto... integers) { return tensor.lazy_reshape(integers...); };
	}
	template<class T> __BC_host_inline__
	auto reshape(Tensor_Base<T>&& tensor) {
		return [&](auto... integers) { return tensor.lazy_reshape(integers...); };
	}
	template<class T> __BC_host_inline__
	auto chunk(Tensor_Base<T>&& tensor) {
		return [&](auto... integers) {
			return tensor.lazy_chunk(integers...);
		};
	}
	template<class T> __BC_host_inline__
	const auto chunk(const Tensor_Base<T>&& tensor) {
		return [&](auto... integers) {
			return tensor.lazy_chunk(integers...);
		};
	}
}

#endif /* TENSOR_SHAPING_H_ */
