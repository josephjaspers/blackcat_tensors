/*
 * Recurrent_Layer_Manager.h
 *
 *  Created on: Jul 25, 2019
 *      Author: joseph
 */

#ifndef RECURRENT_LAYER_MANAGER_H_
#define RECURRENT_LAYER_MANAGER_H_

namespace BC {
namespace nn {
//namespace incomplete {


//TODO
template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<typename layer_traits<Layer>::system_tag, typename layer_traits<Layer>::value_type>>
struct Recurrent_Layer_Manager: Layer {

	template<class... Args>
	Recurrent_Layer_Manager(Args... args):
		Layer(args...),
		inputs(Layer::input_size()) {} 	//TODO must change once we support more dimension for Neural Nets

//	using Allocator = BC::Allocator<typename layer_traits<Layer>::system_tag, typename layer_traits<Layer>::value_type>;

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using Input_Tensor_Type = BC::Tensor<input_tensor_dimension::value, value_type, Allocator>;
	using Batched_Input_Tensor_Type = BC::Tensor<input_tensor_dimension::value+1, value_type, Allocator>;

	std::vector<Input_Tensor_Type> inputs = std::vector<Input_Tensor_Type>(0);
	std::vector<Batched_Input_Tensor_Type> batched_inputs = std::vector<Batched_Input_Tensor_Type>(0);


	auto& get_cache(std::false_type is_batched=std::false_type()) {
		return inputs;
	}
	auto& get_cache(std::true_type is_batched) {
		return batched_inputs;
	}
	auto& get_cache(std::false_type is_batched=std::false_type()) const {
		return inputs;
	}
	auto& get_cache(std::true_type is_batched) const {
		return batched_inputs;
	}

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);
	}

	template<class T>
	auto forward_propagation(const T& expression) {
		constexpr bool is_batched = T::tensor_dimension == input_tensor_dimension::value + 1;
		return forward_propagation_cache(expression, BC::traits::truth_type<is_batched>());
	}
	template<class T>
	auto back_propagation(const T& dy) {
		constexpr bool is_batched = T::tensor_dimension == output_tensor_dimension::value + 1;
		return back_propagation_maybe_supply_previous_outputs(
				dy,
				typename layer_traits<Layer>::backwards_requires_outputs(),
				BC::traits::truth_type<is_batched>());
	}

private:

	/** Casts this to the derived LayerChain class
	 *  (so we can access the previous/next LayerManager's cache) */
	const Derived& as_derived() const { return static_cast<const Derived&>(*this); }
	Derived& as_derived() { return static_cast<Derived&>(*this); }

	template<class T, class TruthType>
	auto forward_propagation_maybe_supply_previous_outputs(
			const T& expression, std::true_type requires_outputs, TruthType is_batched) {

		auto& outputs_container = as_derived().next().layer().get_cache(is_batched);
		using tensor_t = std::conditional_t<TruthType::value, Batched_Input_Tensor_Type, Input_Tensor_Type>;

		if (outputs_container.empty()) {
			return tensor_t(Layer::forward_propagation(get_cache(is_batched).back()));
		} else {
			return tensor_t(Layer::forward_propagation(get_cache(is_batched).back(), outputs_container.back()));
		}
	}

	template<class T, class TruthType>
	auto forward_propagation_maybe_supply_previous_outputs(
			const T& expression, std::false_type requires_outputs, TruthType is_batched) {
		return Layer::forward_propagation(get_cache(is_batched).back());
	}

	template<class T, class TruthType>
	auto forward_propagation_cache(const T& expression, TruthType is_batched) {
		this->get_cache(is_batched).push_back(expression);
		return forward_propagation_maybe_supply_previous_outputs(this->get_cache(is_batched).back(),
				typename layer_traits<Layer>::forward_requires_outputs(), is_batched);
	}


	template<class T, class TruthType>
	auto back_propagation_maybe_supply_previous_outputs(
			const T& dy, std::true_type requires_outputs, TruthType is_batched) {
		return Layer::back_propagation(this->get_cache(is_batched).back(),
				this->as_derived().next().layer().get_cache(is_batched).back(), dy);
	}
	template<class T, class TruthType>
	auto back_propagation_maybe_supply_previous_outputs(
			const T& dy, std::false_type requires_outputs, TruthType is_batched) {
		return Layer::back_propagation(this->get_cache(is_batched).back(), dy);
	}

};

//}
}
}


#endif /* RECURRENT_LAYER_MANAGER_H_ */
