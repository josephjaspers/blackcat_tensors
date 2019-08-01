/*
 * Layer_Manager.h
 *
 *  Created on: Jul 23, 2019
 *      Author: joseph
 */

#ifndef LAYER_MANAGER_H_
#define LAYER_MANAGER_H_

#include "Recurrent_Layer_Manager.h"

namespace BC {
namespace nn {

/*
 * These Layer Manager objects handle the memory management of each layer.
 * They will cache the inputs and outputs (NOT COMPLETED YET) during forward_prop and backward prop.
 *
 */

//Non-recurrent layer_manager
template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<
		typename layer_traits<Layer>::system_tag,
		typename layer_traits<Layer>::value_type>>
struct Layer_Manager: Layer {

	template<class... Args>
	Layer_Manager(Args... args):
		Layer(args...),
		inputs(Layer::input_size()),
		outputs(layer_traits<Layer>::greedy_evaluate_delta::value ? Layer::output_size() : 0) {
	} 	//TODO must change once we support more dimension for Neural Nets

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using Output_Tensor_Type = BC::Tensor<output_tensor_dimension::value, value_type, Allocator>;
	using Batched_Output_Tensor_Type = BC::Tensor<output_tensor_dimension::value+1, value_type, Allocator>;

	using Input_Tensor_Type = BC::Tensor<input_tensor_dimension::value, value_type, Allocator>;
	using Batched_Input_Tensor_Type = BC::Tensor<input_tensor_dimension::value+1, value_type, Allocator>;

	Input_Tensor_Type inputs;
	Output_Tensor_Type outputs;
	Batched_Output_Tensor_Type batched_outputs;
	Batched_Input_Tensor_Type batched_inputs;

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);
		batched_inputs = Batched_Input_Tensor_Type(Layer::input_size(), batch_sz);

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			batched_outputs = Batched_Output_Tensor_Type(Layer::output_size(), batch_sz);
		}
	}
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

	template<class T>
	auto forward_propagation(const T& expression) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == input_tensor_dimension::value + 1)>;
		return Layer::forward_propagation(this->get_cache(is_batched()) = expression);
	}
	template<class T>
	auto back_propagation(const T& dy) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == input_tensor_dimension::value + 1)>;
		return back_propagation_maybe_greedy_evaluate_delta(
				dy,
				is_batched(),
				typename layer_traits<Layer>::greedy_evaluate_delta());
	}
private:

	template<class T, class TruthType>
	auto back_propagation_maybe_greedy_evaluate_delta(const T& dy, TruthType is_batched, std::false_type greedy_eval) {
		return Layer::back_propagation(this->get_cache(is_batched), dy);
	}
	template<class T, class TruthType>
	auto back_propagation_maybe_greedy_evaluate_delta(const T& dy, TruthType is_batched, std::true_type greedy_eval) {
		get_delta_cache(is_batched) = dy;
		return Layer::back_propagation(this->get_cache(is_batched), get_delta_cache(is_batched));
	}

	auto& get_delta_cache(std::false_type is_batched=std::false_type()) {
		return outputs;
	}
	auto& get_delta_cache(std::true_type is_batched) {
		return batched_outputs;
	}
	auto& get_delta_cache(std::false_type is_batched=std::false_type()) const {
		return outputs;
	}
	auto& get_delta_cache(std::true_type is_batched) const {
		return batched_outputs;
	}
};


//TODO MERGE INPUT_LAYER_MANAGER and LAYER_MANAGER  to the same type,
//There code is mostly copy-and paste
template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<typename layer_traits<Layer>::system_tag, typename layer_traits<Layer>::value_type>>
struct Input_Layer_Manager: Layer {

	template<class... Args>
	Input_Layer_Manager(Args... args):
		Layer(args...),
		outputs(layer_traits<Layer>::greedy_evaluate_delta::value ? Layer::output_size() : 0) {
	} 	//TODO must change once we support more dimension for Neural Nets

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using Output_Tensor_Type = BC::Tensor<output_tensor_dimension::value, value_type, Allocator>;
	using Batched_Output_Tensor_Type = BC::Tensor<output_tensor_dimension::value+1, value_type, Allocator>;

	using Input_Tensor_Type = BC::Tensor_View<input_tensor_dimension::value, value_type, Allocator>;
	using Batched_Input_Tensor_Type = BC::Tensor_View<input_tensor_dimension::value+1, value_type, Allocator>;

	Input_Tensor_Type inputs;
	Output_Tensor_Type outputs;
	Batched_Output_Tensor_Type batched_outputs;
	Batched_Input_Tensor_Type batched_inputs;

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			batched_outputs = Batched_Output_Tensor_Type(Layer::output_size(), batch_sz);
		}
	}
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

	template<class T>
	auto forward_propagation(const T& expression) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == input_tensor_dimension::value + 1)>;
		this->get_cache(is_batched()) = std::decay_t<decltype(this->get_cache(is_batched()))>(expression);
		return Layer::forward_propagation(this->get_cache(is_batched()));
	}
	template<class T>
	auto back_propagation(const T& dy) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == input_tensor_dimension::value + 1)>;
		return back_propagation_maybe_greedy_evaluate_delta(
				dy,
				is_batched(),
				typename layer_traits<Layer>::greedy_evaluate_delta());
	}
private:

	template<class T, class TruthType>
	auto back_propagation_maybe_greedy_evaluate_delta(const T& dy, TruthType is_batched, std::false_type greedy_eval) {
		return Layer::back_propagation(this->get_cache(is_batched), dy);
	}
	template<class T, class TruthType>
	auto back_propagation_maybe_greedy_evaluate_delta(const T& dy, TruthType is_batched, std::true_type greedy_eval) {
		get_delta_cache(is_batched) = dy;
		return Layer::back_propagation(this->get_cache(is_batched), get_delta_cache(is_batched));
	}

	auto& get_delta_cache(std::false_type is_batched=std::false_type()) {
		return outputs;
	}
	auto& get_delta_cache(std::true_type is_batched) {
		return batched_outputs;
	}
	auto& get_delta_cache(std::false_type is_batched=std::false_type()) const {
		return outputs;
	}
	auto& get_delta_cache(std::true_type is_batched) const {
		return batched_outputs;
	}
};



}  // namespace nn
}  // namespace BC



#endif /* LAYER_MANAGER_H_ */
