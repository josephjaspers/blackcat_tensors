/*
 * LayerCache.h
 *
 *  Created on: Jul 18, 2019
 *      Author: joseph
 */

#ifndef LAYERCACHE_H_
#define LAYERCACHE_H_

#include <vector>

namespace BC {
namespace nn {

template<
	int Dimension,
	class ValueType,
	class SystemTag,
//	class Allocator=BC::allocators::fancy::Polymorphic_Allocator<SystemTag, ValueType>,

	template<int,class,class>
	class Tensor=BC::Tensor>
struct LayerCache {

	static constexpr int tensor_dimension = Dimension;
	using value_type = ValueType;
	using allocator_type = BC::Allocator<SystemTag, ValueType>;

	using cache_value_type = Tensor<Dimension, ValueType, allocator_type>;
	using cache_type = std::vector<cache_value_type>;

	using batched_cache_value_type = Tensor<Dimension + 1, ValueType, allocator_type>;
	using batched_cache_type = std::vector<batched_cache_value_type>;

//	allocator_type m_allocator;
	cache_type m_cache;
	batched_cache_type m_batched_cache;

//	LayerCache(allocator_type allocator=allocator_type()):
//		m_allocator(allocator) {}


private:

	template<class Expression>
	cache_value_type& cache_impl(const Tensor_Base<Expression>& expression, BC::traits::Integer<Dimension>) {
		m_cache.push_back(expression);
		return m_cache.back();
	}
	template<class Expression>
	batched_cache_value_type& cache_impl(const Tensor_Base<Expression>& expression, BC::traits::Integer<Dimension+1>) {
		m_batched_cache.push_back(expression);
		return m_batched_cache.back();

	}



public:

	template<class Expression>
	auto& cache(const Tensor_Base<Expression>& expression) {
		static_assert(
				Expression::tensor_dimension == Dimension ||
				Expression::tensor_dimension == Dimension + 1,
				"Valid cache arguments must have the same Dimension or the Batched Dimension (Dimension+1)"
			);
		return cache_impl(expression, BC::traits::Integer<Expression::tensor_dimension>());
	}

	const auto& get_last(BC::traits::Integer<Dimension>) const {
		return m_cache.back();
	}

	const auto& get_last(BC::traits::Integer<Dimension+1>) const {
		return m_batched_cache.back();
	}

	auto& get_last(BC::traits::Integer<Dimension>) {
		return m_cache.back();
	}

	auto& get_last(BC::traits::Integer<Dimension+1>) {
		return m_batched_cache.back();
	}


	void clear() {
		clear_batched_cache();
		clear_single_cache();
	}
	void clear_batched_cache() {
		m_batched_cache.clear();
	}
	void clear_single_cache() {
		m_cache.clear();
	}
};

/**
 * The Input layer will use 'TensorView' as its container to avoid copying the first set of data passed in via forwardpropagation.
 *
 * TODO add the Tag option to disable this feature
 */
//template<
//	int Dimension,
//	class ValueType,
//	class SystemTag//,
////	class Allocator=BC::allocators::fancy::Polymorphic_Allocator<SystemTag, ValueType>>
//	>
//using InputLayerCache = LayerCache<Dimension, ValueType, SystemTag,
////		Allocator,
//		BC::Tensor_View>;
//

}
}




#endif /* LAYERCACHE_H_ */
