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


//TODO
template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<typename layer_traits<Layer>::system_tag, typename layer_traits<Layer>::value_type>>
struct Recurrent_Layer_Manager: Layer {

	using value_type = typename layer_traits<Layer>::value_type;
	using allocator_type = Allocator;

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using Input_Tensor_Type = BC::Tensor<input_tensor_dimension::value, value_type, Allocator>;
	using Batched_Input_Tensor_Type = BC::Tensor<input_tensor_dimension::value+1, value_type, Allocator>;

	using Output_Tensor_Type = BC::Tensor<output_tensor_dimension::value, value_type, Allocator>;
	using Batched_Output_Tensor_Type = BC::Tensor<output_tensor_dimension::value+1, value_type, Allocator>;

private:

	unsigned time_minus_index = 0;

	Output_Tensor_Type delta_cache;
	Batched_Output_Tensor_Type batched_delta_cache;

	std::vector<Input_Tensor_Type> inputs;
	std::vector<Batched_Input_Tensor_Type> batched_inputs;

public:

	template<class... Args>
	Recurrent_Layer_Manager(Args... args):
		Layer(args...),
		inputs(0),
		batched_inputs(0) {}

	int batched_cache_index(int tminus_idx=0) const { return batched_inputs.size() - 1 - tminus_idx; }
	int cache_index(int tminus_idx=0) const { return inputs.size() - 1 - tminus_idx; }

	auto& get_cache(std::false_type is_batched) const { return inputs; }
	auto& get_cache(std::true_type  is_batched) const { return batched_inputs; }
	auto& get_cache(std::false_type is_batched) { return inputs; }
	auto& get_cache(std::true_type  is_batched) { return batched_inputs; }
	auto& get_cache(std::false_type is_batched, int tminus_idx) const { return inputs[cache_index(tminus_idx)]; }
	auto& get_cache(std::true_type  is_batched, int tminus_idx) const { return batched_inputs[batched_cache_index(tminus_idx)]; }
	auto& get_cache(std::false_type is_batched, int tminus_idx) { return inputs[cache_index(tminus_idx)]; }
	auto& get_cache(std::true_type  is_batched, int tminus_idx) { return batched_inputs[batched_cache_index(tminus_idx)]; }

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			batched_delta_cache = Batched_Output_Tensor_Type(Layer::output_size(), batch_sz);
		}
	}

	template<class T>
	auto forward_propagation(const T& expression) {
		time_minus_index = 0;
		constexpr bool is_batched = T::tensor_dimension == input_tensor_dimension::value + 1;
		return forward_propagation_cache(expression, BC::traits::truth_type<is_batched>());
	}

	template<class T>
	auto back_propagation(const T& dy_) {
		auto& dy = bp_cache_delta(dy_, typename layer_traits<Layer>::greedy_evaluate_delta());
		constexpr bool is_batched = T::tensor_dimension == output_tensor_dimension::value + 1;
		return back_propagation_maybe_supply_previous_outputs(
				dy,
				typename layer_traits<Layer>::backward_requires_outputs(),
				BC::traits::truth_type<is_batched>());
	}

private:

	template<class T>
	const auto& bp_cache_delta(const T& dy, std::true_type is_batched) {
		auto& cache = get_delta_cache(is_batched);
		return cache = dy;
	}
	template<class T>
	const auto& bp_cache_delta(const T& dy, std::false_type is_batched) {
		return dy;
	}

	auto& get_delta_cache(std::false_type is_batched) const { return delta_cache; }
	auto& get_delta_cache(std::true_type  is_batched) const { return batched_delta_cache; }

	auto& get_delta_cache(std::false_type is_batched) { return delta_cache; }
	auto& get_delta_cache(std::true_type  is_batched) { return batched_delta_cache; }

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
			return tensor_t(Layer::forward_propagation(get_cache(is_batched, time_minus_index)));
		} else {
			auto& last_output = as_derived().next().layer().get_cache(is_batched, time_minus_index);
			return tensor_t(Layer::forward_propagation(get_cache(is_batched, time_minus_index), last_output));
		}
	}

	template<class T, class TruthType>
	auto forward_propagation_maybe_supply_previous_outputs(
			const T& expression, std::false_type requires_outputs, TruthType is_batched) {
		return Layer::forward_propagation(get_cache(is_batched, time_minus_index));
	}

	template<class T, class TruthType>
	auto forward_propagation_cache(const T& expression, TruthType is_batched) {
		this->get_cache(is_batched).push_back(expression);
		return forward_propagation_maybe_supply_previous_outputs(this->get_cache(is_batched, time_minus_index),
				typename layer_traits<Layer>::forward_requires_outputs(), is_batched);
	}


	template<class T, class TruthType>
	auto back_propagation_maybe_supply_previous_outputs(
			const T& dy, std::true_type requires_outputs, TruthType is_batched) {

		return Layer::back_propagation(this->get_cache(is_batched, time_minus_index),
				this->as_derived().next().layer().get_cache(is_batched, time_minus_index++), dy); //note: post inc (not pre)
	}

	template<class T, class TruthType>
	auto back_propagation_maybe_supply_previous_outputs(
			const T& dy, std::false_type requires_outputs, TruthType is_batched) {
		return Layer::back_propagation(this->get_cache(is_batched, time_minus_index++), dy); //note: post inc (not pre)
	}

public:
	void update_weights() {
		Layer::update_weights();

		//Clear all but the last input, this ensures we maintain the internal state inbetween layers
		if (!inputs.empty()) {
			auto last = std::move(inputs.back());
			inputs.clear();
			inputs.push_back(std::move(last));
		}
		if (!batched_inputs.empty()) {
			auto last = std::move(batched_inputs.back());
			batched_inputs.clear();
			batched_inputs.push_back(std::move(last));
		}
	}
};

}
}


#endif /* RECURRENT_LAYER_MANAGER_H_ */
