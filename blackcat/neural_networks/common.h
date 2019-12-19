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

struct Layer_Loader;
struct Momentum;

using nn_default_system_tag = bc::host_tag;

template<class SystemTag, class ValueType, class... AltAllocator>
using nn_default_allocator_type =
		bc::allocators::Recycle_Allocator<SystemTag, ValueType, AltAllocator...>;

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
