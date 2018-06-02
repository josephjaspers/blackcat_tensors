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
	using mathlib_type 	= _mathlib<derived>;private:

private:

	const auto& as_derived() const { return static_cast<derived&>(*this); }
	auto& as_derived() { return static_cast<derived&>(*this); }

public:


	//-------const reshape (using int list)
	template<class... integers>
	const auto self_reshape(integers... ints) const {
		using internal = decltype(std::declval<derived>().data().reshape(ints...));
		static constexpr int tensor_dim =  sizeof...(integers);
		using type = typename tensor_of<tensor_dim>::template type<internal, mathlib_type>;
		return type(this->data().reshape(ints...));

	}
	//-------const reshape (using shape object)
	template<int dims>
	const auto self_reshape(Shape<dims> shape) const {
		using internal = decltype(std::declval<derived>().data().reshape(shape));
		using type = typename tensor_of<dims>::template type<internal, mathlib_type>;
		return type(this->data().reshape(shape));
	}
	//-------non-const reshape (using int list)
	template<class... integers>
	 auto self_reshape(integers... ints)  {
		using internal = decltype(std::declval<derived>().data().reshape(ints...));
		static constexpr int tensor_dim =  sizeof...(integers);
		using type = typename tensor_of<tensor_dim>::template type<internal, mathlib_type>;
		return type(this->as_derived().data().reshape(ints...));

	}
	//-------non-const reshape (using shape object)
	template<int dims>
	 auto self_reshape(Shape<dims> shape)  {
		using internal = decltype(std::declval<derived>().data().reshape(shape));
		using type = typename tensor_of<dims>::template type<internal, mathlib_type>;
		return type(this->data().reshape(shape));
	}



	template<class... integers>
	auto self_chunk(integers... ints) {
		auto internal_location = this->as_derived().data().chunk(ints...);

		return [&](auto... shape_dimension) {
			auto internal_type = internal_location(shape_dimension...);
			constexpr int tensor_dimension = decltype(internal_type)::DIMS();
			using tensor_t = typename tensor_of<tensor_dimension>::template type<decltype(internal_type), mathlib_type>;
			return tensor_t(internal_type);
		};
	}
	template<class... integers>
	const auto self_chunk(integers... ints) const {
		auto internal_location = this->as_derived().data().chunk(ints...);

		return [&](auto... shape_dimension) {
			auto internal_type = internal_location(shape_dimension...);
			constexpr int tensor_dimension = decltype(internal_type)::DIMS();
			using tensor_t = typename tensor_of<tensor_dimension>::template type<decltype(internal_type), mathlib_type>;
			return tensor_t(internal_type);
		};
	}


};
	template<class> class Tensor_Base;

	template<class T> __BC_host_inline__
	auto reshape(const Tensor_Base<T>& tensor) {
		return [&](auto... integers) { return tensor.self_reshape(integers...); };
	}
	template<class T> __BC_host_inline__
	auto reshape(Tensor_Base<T>& tensor) {
		return [&](auto... integers) { return tensor.self_reshape(integers...); };
	}
	template<class T> __BC_host_inline__
	const auto chunk(const Tensor_Base<T>& tensor) {
		return [&](auto... location_indices) {
			return [&] (auto... chunk_dimension) {
				auto internal = tensor.data().chunk(location_indices...)(chunk_dimension...);
				return tensor_of_t<decltype(internal)::DIMS(), decltype(internal), _mathlib<T>>(internal);
			};
		};
	}
	template<class T> __BC_host_inline__
	auto chunk(Tensor_Base<T>& tensor) {
		return [&](auto... location_indices) {
			return [&] (auto... chunk_dimension) {
				auto internal = tensor.data().chunk(location_indices...)(chunk_dimension...);
				return tensor_of_t<decltype(internal)::DIMS(), decltype(internal), _mathlib<T>>(internal);
			};
		};
	}


}

#endif /* TENSOR_SHAPING_H_ */
