/*
 * common.h
 *
 *  Created on: Jun 8, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_NEURALNETWORKS_COMMON_H_
#define BLACKCAT_NEURALNETWORKS_COMMON_H_

#include "layers/Layer_Traits.h"

namespace BC {
namespace nn {

struct Layer_Loader;
struct Momentum;

template<class SystemTag, class ValueType, class... AltAllocator>
using nn_default_allocator_type =
		BC::allocators::Recycle_Allocator<SystemTag, ValueType, AltAllocator...>;

using nn_default_optimizer_type = Momentum;

#ifndef BLACKCAT_DEFAULT_SYSTEM
#define BLACKCAT_DEFAULT_SYSTEM_T BC::host_tag
#else
#define BLACKCAT_DEFAULT_SYSTEM_T BC::BLACKCAT_DEFAULT_SYSTEM##_tag
#endif

}
}



#endif /* COMMON_H_ */
