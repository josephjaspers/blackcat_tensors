/*
 * Blas_tools.h
 *
 *  Created on: Apr 24, 2019
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_BLAS_TOOLS_H_
#define BC_EXPRESSION_BLAS_TOOLS_H_

#include "../Expression_Template_Traits.h"

namespace BC {

class host_tag;
class device_tag;

namespace tensors {
namespace exprs {
namespace blas_expression_parser {

	template<class SystemTag>
	class Blas_Expression_Parser;

	template<class SystemTag>
	using implementation = Blas_Expression_Parser<SystemTag>;

}
}
}
}

#include "Host.h"
#include "Device.h"

#endif /* BLAS_TOOLS_H_ */
