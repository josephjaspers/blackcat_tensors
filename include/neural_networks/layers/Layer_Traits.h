/*
 * Layer_Traits.h
 *
 *  Created on: Jul 20, 2019
 *      Author: joseph
 */

#include "../common.h"

#ifndef BLACKCAT_TENSORS_LAYER_TRAITS_H_
#define BLACKCAT_TENSORS_LAYER_TRAITS_H_

namespace BC {
namespace nn {
namespace detail {

#define BC_QUERY_TAG(tagname)\
	template<class T> using query_##tagname = typename T::tagname;

BC_QUERY_TAG(forward_requires_inputs)
BC_QUERY_TAG(forward_requires_outputs)
BC_QUERY_TAG(backward_requires_inputs)
BC_QUERY_TAG(backward_requires_outputs)
BC_QUERY_TAG(input_tensor_dimension)
BC_QUERY_TAG(output_tensor_dimension)
BC_QUERY_TAG(greedy_evaluate_delta)
BC_QUERY_TAG(requires_extra_cache)
BC_QUERY_TAG(defines_single_predict)
BC_QUERY_TAG(defines_predict)

#undef BC_QUERY_TAG
} // ns detail

template<class T>
struct layer_traits: BC::traits::common_traits<T> {
	/**
	 * Layers have the function: backward_propagate(Args...);
	 * -- The arguments supplied are based upon these traits.
	 *
	 * If forwards_requires_inputs==std::true_type,
	 *     inputs will be supplied in forward prop
	 *
	 * If forwards_requires_outputs==std::true_type,
	 *     outputs will be supplied in forward prop
	 *
	 * If backwards_requires_inputs==std::true_type, inputs will be supplied in backward prop
	 * If backwards_requires_outputs==std::true_type, outputs will be supplied in backward prop
	 */
#define BC_LAYER_TRAITS_TAG(tagname, default_type)\
	using tagname = BC::traits::conditional_detected_t<\
			detail::query_##tagname, T, default_type>;

	using system_tag = BC::traits::conditional_detected_t<
			BC::traits::query_system_tag, T, nn_default_system_tag>;

	using value_type = BC::traits::conditional_detected_t<
			BC::traits::query_value_type, T,
			typename system_tag::default_floating_point_type>;

	using allocator_type = BC::traits::conditional_detected_t<
			BC::traits::query_allocator_type, T,
			nn_default_allocator_type<system_tag, value_type>>;

	BC_LAYER_TRAITS_TAG(input_tensor_dimension, BC::traits::Integer<1>)
	BC_LAYER_TRAITS_TAG(output_tensor_dimension, input_tensor_dimension)

	BC_LAYER_TRAITS_TAG(forward_requires_inputs, std::true_type)
	BC_LAYER_TRAITS_TAG(forward_requires_outputs, std::false_type)

	BC_LAYER_TRAITS_TAG(backward_requires_inputs, std::true_type)
	BC_LAYER_TRAITS_TAG(backward_requires_outputs, std::false_type)

	BC_LAYER_TRAITS_TAG(requires_extra_cache, std::false_type)
	BC_LAYER_TRAITS_TAG(greedy_evaluate_delta, std::false_type)

	BC_LAYER_TRAITS_TAG(defines_predict, std::false_type)
	BC_LAYER_TRAITS_TAG(defines_single_predict, std::false_type)


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
