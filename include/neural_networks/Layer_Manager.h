/*
 * Layer_Manager.h
 *
 *  Created on: Jul 23, 2019
 *      Author: joseph
 */

#ifndef LAYER_MANAGER_H_
#define LAYER_MANAGER_H_

namespace BC {
namespace nn {

/*
 * These Layer Manager objects handle the memory management of each layer.
 * They will cache the inputs and outputs (NOT COMPLETED YET) during forward_prop and backward prop.
 *
 */

//TODO
template<class Layer>
struct Recurrent_Layer_Manager;

//Non-recurrent layer_manager
template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<typename layer_traits<Layer>::system_tag, typename layer_traits<Layer>::value_type>>
struct Layer_Manager: Layer {

	template<class... Args>
	Layer_Manager(Args... args):
		Layer(args...),
		inputs(Layer::input_size()) {} 	//TODO must change once we support more dimension for Neural Nets

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using TensorX = BC::Tensor<input_tensor_dimension::value, value_type, Allocator>;
	using Batched_TensorX = BC::Tensor<input_tensor_dimension::value+1, value_type, Allocator>;

	TensorX inputs;
	Batched_TensorX batched_inputs;

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);
		batched_inputs = Batched_TensorX(Layer::input_size(), batch_sz);
	}

	template<class T>
	auto forward_propagation(const T& expression) {
		constexpr bool is_batched = T::tensor_dimension == input_tensor_dimension::value + 1;
		return forward_propagation(expression, BC::traits::truth_type<is_batched>());
	}
	template<class T>
	auto back_propagation(const T& dy) {
		constexpr bool is_batched = T::tensor_dimension == output_tensor_dimension::value + 1;
		return back_propagation(dy, BC::traits::truth_type<is_batched>());
	}

private:
	template<class T>
	auto forward_propagation(const T& expression, std::true_type is_batched) {
		batched_inputs = expression;
		return Layer::forward_propagation(batched_inputs);
	}
	template<class T>
	auto forward_propagation(const T& expression, std::false_type is_batched) {
		inputs = expression;
		return Layer::forward_propagation(inputs);
	}

	template<class T>
	auto back_propagation(const T& dy, std::true_type is_batched) {
		return Layer::back_propagation(batched_inputs, dy);
	}

	template<class T>
	auto back_propagation(const T& dy, std::false_type is_batched) {
		return Layer::back_propagation(inputs, dy);
	}
};


//Same as forward_layer manager but uses 'Tensor_Views' to avoid doing copy operations on input data
template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<typename layer_traits<Layer>::system_tag, typename layer_traits<Layer>::value_type>>
struct Input_Layer_Manager: Layer {

	template<class... Args>
	Input_Layer_Manager(Args... args):
		Layer(args...) {} 	//TODO must change once we support more dimension for Neural Nets

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using TensorX = BC::Tensor_View<input_tensor_dimension::value, value_type, Allocator>;
	using Batched_TensorX = BC::Tensor_View<input_tensor_dimension::value+1, value_type, Allocator>;

	TensorX inputs;
	Batched_TensorX batched_inputs;

	template<class T>
	auto forward_propagation(const T& expression) {
		constexpr bool is_batched = T::tensor_dimension == input_tensor_dimension::value + 1;
		return forward_propagation(expression, BC::traits::truth_type<is_batched>());
	}
	template<class T>
	auto back_propagation(const T& dy) {
		constexpr bool is_batched = T::tensor_dimension == output_tensor_dimension::value + 1;
		return back_propagation(dy, BC::traits::truth_type<is_batched>());
	}

private:
	template<class T>
	auto forward_propagation(const T& expression, std::true_type is_batched) {
		batched_inputs = Batched_TensorX(expression);
		return Layer::forward_propagation(batched_inputs);
	}
	template<class T>
	auto forward_propagation(const T& expression, std::false_type is_batched) {
		inputs = TensorX(expression);
		return Layer::forward_propagation(inputs);
	}

	template<class T>
	auto back_propagation(const T& dy, std::true_type is_batched) {
		return Layer::back_propagation(batched_inputs, dy);
	}

	template<class T>
	auto back_propagation(const T& dy, std::false_type is_batched) {
		return Layer::back_propagation(inputs, dy);
	}
};

}  // namespace nn
}  // namespace BC



#endif /* LAYER_MANAGER_H_ */
