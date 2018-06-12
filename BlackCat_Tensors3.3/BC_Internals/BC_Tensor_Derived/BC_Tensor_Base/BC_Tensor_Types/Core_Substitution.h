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
//	using scalar = _scalar<tensor>;
//	operator Core<tensor>& () { return static_cast<Core<tensor>&>(*this); }
//	operator scalar* 		() {  return static_cast<Core<tensor>&>(*this);}

	__BCinline__ void temporary_destroy() {
		Core<tensor>::destroy();
	}
};
}
}



#endif /* CORE_SUB_H_ */
