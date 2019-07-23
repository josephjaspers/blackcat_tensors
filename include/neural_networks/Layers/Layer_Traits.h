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
namespace impl {

using namespace BC::traits;

template<class T> using query_forward_requires_inputs  = typename T::forward_requires_inputs;
template<class T> using query_forward_requires_outputs = typename T::forward_requires_outputs;
template<class T> using query_backwards_requires_inputs  = typename T::backwards_requires_inputs;
template<class T> using query_backwards_requires_outputs = typename T::backwards_requires_outputs;

template<class T> using query_input_tensor_dimension  = typename T::input_tensor_dimension;
template<class T> using query_output_tensor_dimension = typename T::output_tensor_dimension;


//If true we cache the delta- into a matrix/vector. This is not stored in recurrent layers.
//It is used for things like feedforward backprop which require using the 'deltaY' error multiple times
//This tag is used to prevent recalculating the same error values multiple times (and saving on reallocations)
template<class T> using query_backwards_delta_should_be_cached = typename T::query_backwards_delta_should_be_cached;

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
	using system_tag = typename T::system_tag;
	using value_type
			= conditional_detected_t<BC::traits::query_value_type, T,
				typename system_tag::default_floating_point_type>;
	using input_tensor_dimension
			= conditional_detected_t<query_input_tensor_dimension, T, BC::traits::Integer<1>>;
	using output_tensor_dimension
			= conditional_detected_t<query_output_tensor_dimension, T, BC::traits::Integer<1>>;


	using forward_requires_inputs
			= conditional_detected_t<query_forward_requires_inputs, T, std::true_type>;
	using forward_requires_outputs
			= conditional_detected_t<query_forward_requires_outputs, T, std::false_type>;


	using backwards_requires_inputs
			= conditional_detected_t<query_backwards_requires_inputs, T, std::true_type>;
	using backwards_requires_outputs
			= conditional_detected_t<query_backwards_requires_outputs, T, std::false_type>;
	using backwards_delta_should_be_cached
			= conditional_detected_t<query_backwards_requires_outputs, T, std::false_type>;
};

 } //namespace impl

using impl::layer_traits;

}  // namespace nn
}  // namespace BC



#endif /* LAYER_TRAITS_H_ */
