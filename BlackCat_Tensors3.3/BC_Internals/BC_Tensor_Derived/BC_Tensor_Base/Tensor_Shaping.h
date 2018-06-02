/*
 * Tensor_Shaping.h
 *
 *  Created on: May 31, 2018
 *      Author: joseph
 */

#ifndef TENSOR_SHAPING_H_
#define TENSOR_SHAPING_H_

namespace BC {
template<class derived>
struct Tensor_Shaping {

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

	//THIS IS THE CURRIED CHUNK LAMBDA, WE MUST USE AN ACTUAL CLASS TO ACT AS A LAMDA AS CUDA COMPILER IS IFFY WITH LAMBDA
	struct CHUNK {

		const derived& tensor;
		int location;
		CHUNK(const derived& tensor_, int index_) : tensor(tensor_), location(index_) {}

		template<class... integers>
		const auto operator () (integers... shape_dimensions) const {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename Tensor_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = tensor_of_t<tensor_dimension, chunk_type, mathlib_type>;

			return type(tensor.data().chunk(location, shape_dimensions...));
		}
		template<class... integers>
		auto operator () (integers... shape_dimensions) {
			static constexpr int tensor_dimension = sizeof...(shape_dimensions);
			using chunk_type = typename Tensor_Chunk<tensor_dimension>::template implementation<functor_type>;
			using type = tensor_of_t<tensor_dimension, chunk_type, mathlib_type>;

			return type(tensor.data().chunk(location, shape_dimensions...));
		}
	};
};

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
