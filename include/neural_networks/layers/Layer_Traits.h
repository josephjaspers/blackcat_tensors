/*
 * Layer_Traits.h
 *
 *  Created on: Jul 20, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TENSORS_LAYER_TRAITS_H_
#define BLACKCAT_TENSORS_LAYER_TRAITS_H_

namespace BC {
namespace nn {
namespace detail {

template<class T> using query_forward_requires_inputs  = typename T::forward_requires_inputs;
template<class T> using query_forward_requires_outputs = typename T::forward_requires_outputs;
template<class T> using query_forward_requires_extra_cache = typename T::forward_requires_extra_cache;
template<class T> using query_backward_requires_inputs  = typename T::backward_requires_inputs;
template<class T> using query_backward_requires_outputs = typename T::backward_requires_outputs;
template<class T> using query_backward_requires_extra_cache = typename T::backward_requires_extra_cache;
template<class T> using query_input_tensor_dimension  = typename T::input_tensor_dimension;
template<class T> using query_output_tensor_dimension = typename T::output_tensor_dimension;
template<class T> using query_greedy_evaluate_delta = typename T::greedy_evaluate_delta;

//If true we cache the delta- into a matrix/vector. This is not stored in recurrent layers.
//It is used for things like feedforward backprop which require using the 'deltaY' error multiple times
//This tag is used to prevent recalculating the same error values multiple times (and saving on reallocations)
template<class T> using query_backwards_delta_should_be_cached = typename T::query_backwards_delta_should_be_cached;

template<class T> using query_requires_extra_cache = typename T::requires_extra_cache;
template<class T> using query_extra_batched_cache_args = typename T::extra_batched_cache_args;

template<class T>
using query_defines_single_predict = typename T::defines_single_predict;

template<class T>
using query_defines_predict = typename T::defines_predict;

} // ns detail

template<class T>
struct layer_traits: BC::traits::common_traits<T> {
	/**
	 *  Layers have the function: backward_propagate(Args...);
	 *  -- The arguments supplied are based upon these traits.
	 *
	 *  If forwards_requires_inputs==std::true_type, inputs will be supplied in forward prop
	 *  If forwards_requires_outputs==std::true_type, outputs will be supplied in forward prop
	 *
	 *  If backwards_requires_inputs==std::true_type, inputs will be supplied in backward prop
	 *  If backwards_requires_outputs==std::true_type, outputs will be supplied in backward prop
	 *
	 */

	using system_tag = typename T::system_tag;

	using value_type =
			BC::traits::conditional_detected_t<
					BC::traits::query_value_type, T,
					typename system_tag::default_floating_point_type>;

	using allocator_type =
			BC::traits::conditional_detected_t<
					BC::traits::query_allocator_type, T,
					BC::Allocator<system_tag, value_type>>;

	using requires_extra_cache = BC::traits::conditional_detected_t<
			detail::query_requires_extra_cache, T, std::false_type>;

	using input_tensor_dimension = BC::traits::conditional_detected_t<
			detail::query_input_tensor_dimension, T, BC::traits::Integer<1>>;

	using output_tensor_dimension = BC::traits::conditional_detected_t<
			detail::query_output_tensor_dimension, T, input_tensor_dimension>;

	using forward_requires_inputs = BC::traits::conditional_detected_t<
			detail::query_forward_requires_inputs, T, std::true_type>;

	using forward_requires_outputs = BC::traits::conditional_detected_t<
			detail::query_forward_requires_outputs, T, std::false_type>;

	using forward_requires_extra_cache = BC::traits::conditional_detected_t<
			detail::query_forward_requires_extra_cache, T, std::false_type>;

	using backward_requires_inputs = BC::traits::conditional_detected_t<
			detail::query_backward_requires_inputs, T, std::true_type>;

	using backward_requires_outputs = BC::traits::conditional_detected_t<
			detail::query_backward_requires_outputs, T, std::false_type>;

	using backward_delta_should_be_cached = BC::traits::conditional_detected_t<
			detail::query_backward_requires_outputs, T, std::false_type>;

	using backward_requires_extra_cache = BC::traits::conditional_detected_t<
			detail::query_backward_requires_extra_cache, T, std::false_type>;

	using greedy_evaluate_delta = BC::traits::conditional_detected_t<
			detail::query_greedy_evaluate_delta, T, std::false_type>;


	template<class... Args>
	static auto select_on_predict(T& layer, Args&&... args) {
		using detected =
				BC::traits::truth_type<
						BC::traits::is_detected_v<
								detail::query_defines_predict, T>>;
		return select_on_predict(detected(), layer, std::forward<Args>(args)...);
	}

	template<class... Args>
	static auto select_on_single_predict(T& layer, Args&&... args) {
		using detected =
				BC::traits::truth_type<
						BC::traits::is_detected_v<
								detail::query_defines_single_predict, T>>;
		return select_on_single_predict(detected(), layer, std::forward<Args>(args)...);
	}


private:

	template<class... Args>
	static auto select_on_predict(std::true_type, T& layer, Args&&... args) {
		return layer.predict(args...);
	}

	template<class... Args>
	static auto select_on_predict(std::false_type, T& layer, Args&&... args) {
		return layer.forward_propagation(args...);
	}

	template<class... Args>
	static auto select_on_single_predict(std::true_type, T& layer, Args&&... args) {
		return layer.single_predict(args...);
	}

	template<class... Args>
	static auto select_on_single_predict(std::false_type, T& layer, Args&&... args) {
		return select_on_predict(layer, args...);
	}


};
}  // namespace nn
}  // namespace BC



#endif /* LAYER_TRAITS_H_ */
