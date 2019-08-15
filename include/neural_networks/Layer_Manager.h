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

template<class Type, class BatchedType>
struct Forward_Tensor_Cache {

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

	void clear_bp_storage() {}
	void increment_time_index() {}
	void decrement_time_index() {}
	void zero_time_index() {}

};

template<class Type, class BatchedType>
struct Recurrent_Tensor_Cache {

	//time index refers to 'the past index' IE time_index==3 means '3 timestamps in the past'
	int time_index = 0;

	std::vector<Type> tensor;
	std::vector<BatchedType> batched_tensor;

	void increment_time_index() { time_index++; }
	void decrement_time_index() { time_index--; }
	void zero_time_index() { time_index = 0; }

	void init_tensor(Type init) {
		tensor.push_back(std::move(init));
		tensor.back().zero();
	}
	void init_batched(BatchedType init) {
		batched_tensor.push_back(std::move(init));
		batched_tensor.back().zero();
	}
	template<class... Args>
	void init_tensor(const Args&... init) {
		tensor.push_back(Type(init...));
		tensor.back().zero();
	}
	template<class... Args>
	void init_batched(const Args&... init) {
		batched_tensor.push_back(BatchedType(init...));
		batched_tensor.back().zero();
	}

	Type& load(std::false_type is_batched = std::false_type()) {
		return tensor[tensor.size() - 1 - time_index];
	}
	BatchedType& load(std::true_type is_batched) {
		return batched_tensor[batched_tensor.size() - 1 - time_index];
	}
	const Type& load(std::false_type is_batched = std::false_type()) const {
		return tensor[tensor.size() - 1 - time_index];
	}
	const BatchedType& load(std::true_type is_batched) const {
		return batched_tensor[batched_tensor.size() - 1 - time_index];
	}

	template<class T>
	auto& store(const T& expression) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == BatchedType::tensor_dimension)>;
		return store(expression, is_batched());
	}

	template<class T>
	auto& store(const T& expression, std::true_type is_batched) {
		this->batched_tensor.push_back(expression);
		return this->batched_tensor.back();
	}

	template<class T>
	auto& store(const T& expression, std::false_type is_batched) {
		this->tensor.push_back(expression);
		return this->tensor.back();
	}

	void clear_bp_storage() {
		if (tensor.size() > 1) {
			auto last = std::move(tensor.back());
			tensor.clear();
			tensor.push_back(std::move(last));
		}
		if (batched_tensor.size() > 1) {
			auto last = std::move(batched_tensor.back());
			batched_tensor.clear();
			batched_tensor.push_back(std::move(last));
		}
	}
};


template<
	class Derived, //The LayerChain base
	class Layer,
	class Recurrent=std::false_type, //must be std::true_type or false_type or have "value"
	class Allocator=BC::Allocator<
		typename layer_traits<Layer>::system_tag,
		typename layer_traits<Layer>::value_type>>
struct Layer_Manager: Layer {

	template<class... Args>
	Layer_Manager(Args... args):
		Layer(args...) {
		m_input_cache.init_tensor(Layer::input_size());

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			m_delta_cache.init_tensor(Layer::output_size());
		}
	} 	//TODO must change once we support more dimension for Neural Nets

	using input_tensor_dimension = typename layer_traits<Layer>::input_tensor_dimension;
	using output_tensor_dimension = typename layer_traits<Layer>::output_tensor_dimension;

	using value_type = typename layer_traits<Layer>::value_type;

	using output_tensor_type = BC::Tensor<output_tensor_dimension::value, value_type, Allocator>;
	using batched_output_tensor_type = BC::Tensor<output_tensor_dimension::value+1, value_type, Allocator>;

	using input_tensor_type = BC::Tensor<input_tensor_dimension::value, value_type, Allocator>;
	using batched_input_tensor_type = BC::Tensor<input_tensor_dimension::value+1, value_type, Allocator>;

	using requires_extra_cache = typename layer_traits<Layer>::requires_extra_cache;
	using extra_cache_type = typename layer_traits<Layer>::extra_cache_args;
	using extra_batched_cache_type = typename layer_traits<Layer>::extra_batched_cache_args;

	template<class Tensor, class BatchedTensor>
	using tensor_cache_type = std::conditional_t<Recurrent::value,
			Recurrent_Tensor_Cache<Tensor, BatchedTensor>, Forward_Tensor_Cache<Tensor, BatchedTensor>>;

	///delta cache is always a 'forward' cache -> It is only when delta is expected to be reused in the layer
	Forward_Tensor_Cache<output_tensor_type, batched_output_tensor_type> m_delta_cache;
	tensor_cache_type<input_tensor_type, batched_input_tensor_type> m_input_cache;
	tensor_cache_type<extra_cache_type, extra_batched_cache_type> m_extra_cache_args;

	void zero_bp_index() {
		m_input_cache.zero_time_index();
		m_delta_cache.zero_time_index();
	}
	void inc_bp_index() {
		m_input_cache.increment_time_index();
		m_delta_cache.increment_time_index();
	}

	void set_batch_size(BC::size_t batch_sz) {
		Layer::set_batch_size(batch_sz);
		m_input_cache.init_batched(batched_input_tensor_type(Layer::input_size(), batch_sz));

		if (layer_traits<Layer>::greedy_evaluate_delta::value) {
			m_delta_cache.init_batched(batched_output_tensor_type(Layer::output_size(), batch_sz));
		}
	}

	template<class T>
	auto forward_propagation(const T& expression) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == input_tensor_dimension::value + 1)>;
		return forward_supply_outputs(typename layer_traits<Layer>::forward_requires_outputs(), this->m_input_cache.store(expression));
	}
	template<class T>
	auto back_propagation(const T& dy) {
		using is_batched = BC::traits::truth_type<(T::tensor_dimension == output_tensor_dimension::value + 1)>;
		return  backward_supply_outputs(typename layer_traits<Layer>::backward_requires_outputs(), m_input_cache.load(is_batched()), dy);
	}

	void update_weights() {
		Layer::update_weights();
		m_input_cache.clear_bp_storage();
		m_extra_cache_args.clear_bp_storage();
	}

private:
	//TODO add suportfor 'extra' args

	const auto& as_derived() const { return static_cast<const Derived&>(*this); }
	auto& as_derived() { return static_cast<Derived&>(*this); }

//	template<class... T>
//	auto forward_supply_extra(std::false_type, const T&... args) {
//		return Layer::forward_propagation(args...);
//	}
//
//	template<class Input, class... T>
//	auto forward_supply_extra(std::true_type, const Input& inputs, const T&... args){
//		using is_batched = BC::traits::truth_type<Input::tensor_dimension==input_tensor_dimension::value+1>;
//		auto& extra = m_extra_cache_args.load(is_batched());
//		return Layer::forward_propagation(inputs, args..., extra);
//	}

	template<class X>
	auto forward_supply_outputs(std::false_type,  const X& inputs) {
		return Layer::forward_propagation(inputs);
//		return forward_supply_extra(layer_traits<Layer>::forward_requires_extra_cache(), input);
	}
	template<class Input>
	auto forward_supply_outputs(std::true_type,  const Input& inputs) {
		using is_batched = BC::traits::truth_type<Input::tensor_dimension==input_tensor_dimension::value+1>;
		auto& outputs = BC::traits::derived_cast(*this).next().layer().m_input_cache.load(is_batched());
		return Layer::forward_propagation(inputs, outputs);
//		return forward_supply_extra(typename layer_traits<Layer>::forward_requires_extra_cache(), inputs, outputs);
	}


	template<class X, class... T>
	auto backward_supply_outputs(std::false_type,  const X& x, const T&... args) {
//		return backward_supply_extra(layer_traits<Layer>::backward_requires_extra_cache(), args...);
		return Layer::back_propagation(x, args...);
	}
//	template<class... T>
//	auto backward_supply_extra(std::false_type, const T&... args) {
//		return Layer::back_propagation(args...);
//	}
	template<class Input, class Dy>
	auto backward_supply_outputs(std::true_type,  const Input& inputs, const Dy& delta) {
		using is_batched = BC::traits::truth_type<Input::tensor_dimension==input_tensor_dimension::value+1>;
		auto& outputs = BC::traits::derived_cast(*this).next().layer().m_input_cache.load(is_batched());
		return Layer::back_propagation(inputs, outputs, delta);
//		return  Layer::back_propagation(layer_traits<Layer>::backward_requires_extra_cache(), inputs, outputs, delta);
	}
//	template<class Input, class... T>
//	auto backward_supply_extra(std::true_type, const Input& inputs, const T&... args){
//		using is_batched = BC::traits::truth_type<Input::tensor_dimension==input_tensor_dimension::value+1>;
//		auto& outputs = BC::traits::derived_cast(*this).next().layer().m_input_cache.load(is_batched());
//		return Layer::back_propagation(inputs, args...);
//	}
//

	template<class T>
	auto maybe_cache_delta(const T& dy) {
		return maybe_cache_delta_impl(dy, typename layer_traits<Layer>::greedy_evaluate_delta());
	}
	template<class T>
	auto& maybe_cache_delta_impl(const T& dy, std::true_type cache_delta) {
		return m_delta_cache.store(dy);
	}
	template<class T>
	auto maybe_cache_delta_impl(const T& dy, std::false_type cache_delta) {
		return dy;
	}
};
}  // namespace nn
}  // namespace BC



#endif /* LAYER_MANAGER_H_ */
