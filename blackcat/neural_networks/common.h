/*
 * common.h
 *
 *  Created on: Jun 8, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORKS_COMMON_H_
#define BLACKCAT_NEURALNETWORKS_COMMON_H_


namespace bc {
namespace nn {

template<class ValueType, class SystemTag, class... AltAllocator>
using nn_default_allocator_type =
		bc::allocators::Recycle_Allocator<ValueType, SystemTag, AltAllocator...>;

template<
	class ValueType,
	class SystemTag,
	class NumDimension,
	class AllocatorType=bc::allocators::Polymorphic_Allocator<ValueType, SystemTag>>
struct Tensor_Descriptor
{
	using value_type = ValueType;
	using tensor_dim = NumDimension;
	using allocator_type = AllocatorType;
	using system_tag = SystemTag;

	using type = bc::Tensor<tensor_dim::value, value_type, allocator_type>;
	using batched_type = bc::Tensor<tensor_dim::value+1, value_type, allocator_type>;
};

using bc::traits::Integer;

struct Layer_Loader;
struct Momentum;

using nn_default_system_tag = bc::host_tag;


using nn_default_optimizer_type = Momentum;


static constexpr double default_learning_rate = 0.003;

#ifndef BLACKCAT_DEFAULT_SYSTEM
#define BLACKCAT_DEFAULT_SYSTEM_T bc::host_tag
#else
#define BLACKCAT_DEFAULT_SYSTEM_T bc::BLACKCAT_DEFAULT_SYSTEM##_tag
#endif

}
}

//required to be below
#include "layers/layer_traits.h"

#endif /* COMMON_H_ */
