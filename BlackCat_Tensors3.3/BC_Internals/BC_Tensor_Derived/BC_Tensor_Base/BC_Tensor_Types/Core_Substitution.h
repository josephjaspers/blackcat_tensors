/*
 * Core_Sub.h
 *
 *  Created on: Jun 11, 2018
 *      Author: joseph
 */

#ifndef CORE_SUB_H_
#define CORE_SUB_H_

#include "Core.h"

namespace BC {
namespace internal {


template<class tensor>
struct Tensor_Substitution : Core<tensor>  {

	using Core<tensor>::Core;

	__BCinline__ void temporary_destroy() {
		Core<tensor>::destroy();
	}
};
}
}



#endif /* CORE_SUB_H_ */
