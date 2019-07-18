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

	template<class Expression>
	void cache(const Tensor_Base<Expression>& expression) {
		static_assert(
				Expression::tensor_dimension == Dimension ||
				Expression::tensor_dimension == Dimension + 1,
				"Valid cache arguments must have the same Dimension or the Batched Dimension (Dimension+1)"
			);

		BC::traits::constexpr_ternary<(Expression::tensor_dimension == Dimension)>(
			BC::traits::bind([&](auto& m_cache_, auto& expression_){
				m_cache_.push_back(cache_value_type(expression_));//, m_allocator));
			}, m_cache, expression),

			BC::traits::bind([&](auto& m_batched_cache_, auto& expression_){
				m_batched_cache_.push_back(batched_cache_value_type(expression_));//, m_allocator));
			}, m_batched_cache, expression));

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
template<
	int Dimension,
	class ValueType,
	class SystemTag//,
//	class Allocator=BC::allocators::fancy::Polymorphic_Allocator<SystemTag, ValueType>>
	>
using InputLayerCache = LayerCache<Dimension, ValueType, SystemTag,
//		Allocator,
		BC::Tensor_View>;


}
}




#endif /* LAYERCACHE_H_ */
