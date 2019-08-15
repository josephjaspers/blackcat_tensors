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

template<class Type, class BatchedType>
struct Tensor_Cache {

	Type tensor;
	BatchedType batched_tensor;

	void init_tensor(Type init) {
		this->tensor = std::move(init);
	}
	void init_batched(BatchedType init) {
		this->batched_tensor = std::move(init);
	}
	template<class... Args>
	void init_tensor(const Args&... init) {
		this->tensor = Type(init...);
	}
	template<class... Args>
	void init_batched(const Args&... init) {
		this->batched_tensor = BatchedType(init...);
	}

	auto& load(std::false_type is_batched=std::false_type()) { return tensor; }
	auto& load(std::true_type is_batched) { return batched_tensor; }
	auto& load(std::false_type is_batched=std::false_type()) const { return tensor; }
	auto& load(std::true_type is_batched) const { return batched_tensor; }

	template<class T>
	auto& store(const T& expression) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == BatchedType::tensor_dimension)>;
		return store(expression, is_batched());
	}

	template<class T>
	auto& store(const T& expression, std::true_type is_batched) {
		return this->batched_tensor = expression;
	}

	template<class T>
	auto& store(const T& expression, std::false_type is_batched) {
		return this->tensor = expression;
	}
};

template<
	class Derived, //The LayerChain base
	class Layer,
	class Allocator=BC::Allocator<
		typename layer_traits<Layer>::system_tag,
		typename layer_traits<Layer>::value_type>>
struct Layer_Manager: Layer {

	template<class... Args>
	Layer_Manager(Args... args):
		Layer(args...) {
		m_input_cache.init_tensor(Layer::input_size());

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			m_output_cache.init_tensor(Layer::output_size());
		}
	} 	//TODO must change once we support more dimension for Neural Nets

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using output_tensor_type = BC::Tensor<output_tensor_dimension::value, value_type, Allocator>;
	using batched_output_tensor_type = BC::Tensor<output_tensor_dimension::value+1, value_type, Allocator>;

	using input_tensor_type = BC::Tensor<input_tensor_dimension::value, value_type, Allocator>;
	using batched_input_tensor_type = BC::Tensor<input_tensor_dimension::value+1, value_type, Allocator>;

	Tensor_Cache<output_tensor_type, batched_output_tensor_type> m_output_cache;
	Tensor_Cache<input_tensor_type, batched_input_tensor_type> m_input_cache;

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);
		m_input_cache.init_batched(batched_input_tensor_type(Layer::input_size(), batch_sz));

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			m_output_cache.init_batched(batched_output_tensor_type(Layer::output_size(), batch_sz));
		}
	}

	template<class T>
	auto forward_propagation(const T& expression)
		-> decltype(Layer::forward_propagation(this->m_input_cache.store(expression))) {
		return Layer::forward_propagation(this->m_input_cache.store(expression));
	}
	template<class T>
	auto back_propagation(const T& dy) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == input_tensor_dimension::value + 1)>;
		return Layer::back_propagation(
				m_input_cache.load(is_batched()),
				maybe_cache_delta(dy));
	}
private:

	template<class T>
	auto maybe_cache_delta(const T& dy) {
		return maybe_cache_delta_impl(dy, typename layer_traits<Layer>::greedy_evaluate_delta());
	}
	template<class T>
	auto& maybe_cache_delta_impl(const T& dy, std::true_type cache_delta) {
		return m_output_cache.store(dy);
	}
	template<class T>
	auto maybe_cache_delta_impl(const T& dy, std::false_type cache_delta) {
		return dy;
	}
};
}  // namespace nn
}  // namespace BC



#endif /* LAYER_MANAGER_H_ */
