/*
 * Layer_Traits.h
 *
 *  Created on: Jul 20, 2019
 *      Author: joseph
 */

#ifndef LAYER_TRAITS_H_
#define LAYER_TRAITS_H_

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

template<class T> using query_extra_cache_args = typename T::extra_cache_args;
template<class T> using query_extra_batched_cache_args = typename T::extra_batched_cache_args;

template<class T, class Delta> using detect_back_propagation_type =
		decltype(std::declval<T>().back_propagation(std::declval<T>(), std::declval<Delta>()));

}

template<class T>
struct layer_traits {
	/**
	 *  Layers have the function: backward_propagate(Args...);
	 *  -- The arguments supplied are based upon these traits.
	 *
	 *  If forwards_requires_inputs==std::true_type, inputs will be supplied in forward prop
	 *  If forwards_requires_outputs==std::true_type, outputs will be supplied in forward prop
	 *
	 *
	 *  If backwards_requires_inputs==std::true_type, inputs will be supplied in backward prop
	 *  If backwards_requires_outputs==std::true_type, outputs will be supplied in backward prop
	 *
	 */

	template<class Delta, class=void>
	auto select_on_backpropagation(T& expression, const Delta& delta);

	using system_tag = typename T::system_tag;
	using value_type
			= BC::traits::conditional_detected_t<BC::traits::query_value_type, T,
				typename system_tag::default_floating_point_type>;

	using requires_extra_cache = BC::traits::truth_type<BC::traits::is_detected_v<detail::query_extra_cache_args, T>>;

	using input_tensor_dimension
			= BC::traits::conditional_detected_t<detail::query_input_tensor_dimension, T, BC::traits::Integer<1>>;
	using output_tensor_dimension
			= BC::traits::conditional_detected_t<detail::query_output_tensor_dimension, T, input_tensor_dimension>;

	using forward_requires_inputs
			= BC::traits::conditional_detected_t<detail::query_forward_requires_inputs, T, std::true_type>;
	using forward_requires_outputs
			= BC::traits::conditional_detected_t<detail::query_forward_requires_outputs, T, std::false_type>;
	using forward_requires_extra_cache
			= BC::traits::conditional_detected_t<detail::query_forward_requires_extra_cache, T, std::false_type>;

	using backward_requires_inputs
			= BC::traits::conditional_detected_t<detail::query_backward_requires_inputs, T, std::true_type>;
	using backward_requires_outputs
			= BC::traits::conditional_detected_t<detail::query_backward_requires_outputs, T, std::false_type>;
	using backward_delta_should_be_cached
			= BC::traits::conditional_detected_t<detail::query_backward_requires_outputs, T, std::false_type>;
	using backward_requires_extra_cache
			= BC::traits::conditional_detected_t<detail::query_backward_requires_extra_cache, T, std::false_type>;

	using greedy_evaluate_delta
			= BC::traits::conditional_detected_t<detail::query_greedy_evaluate_delta, T, std::false_type>;

	using extra_cache_args
			= BC::traits::conditional_detected_t<detail::query_extra_cache_args, T, std::tuple<>>;
	using extra_batched_cache_args
			= BC::traits::conditional_detected_t<detail::query_extra_batched_cache_args, T, std::tuple<>>;
};

template<class T, class Delta>
auto layer_traits<T>::select_on_backpropagation<Delta, std::enable_if_t<BC::traits::is_detected_v<detect_back_propgation_type<T>>>>
(T& expression, const Delta& delta) {

}

}  // namespace nn
}  // namespace BC



#endif /* LAYER_TRAITS_H_ */
