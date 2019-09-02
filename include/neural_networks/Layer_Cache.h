/*
 * Layer_Cache.h
 *
 *  Created on: Aug 31, 2019
 *      Author: joseph
 */

#ifndef BLACKCATTENSORS_NEURALNETWORKS_LAYER_CACHE_H_
#define BLACKCATTENSORS_NEURALNETWORKS_LAYER_CACHE_H_

#include <vector>
#include <type_traits>

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
		this->batched_tensor = expression;
		return this->batched_tensor;
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

}
}


#endif /* LAYER_CACHE_H_ */
