/*
 * Blas_tools.h
 *
 *  Created on: Apr 24, 2019
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_BLAS_TOOLS_H_
#define BC_EXPRESSION_BLAS_TOOLS_H_

#include "../expression_template_traits.h"

namespace bc {

struct host_tag;
struct device_tag;

namespace tensors {
namespace exprs {
namespace blas_expression_parser {

	template<class SystemTag>
	struct Blas_Expression_Parser;

	template<class SystemTag>
	using implementation = Blas_Expression_Parser<SystemTag>;

}
}
}
}

#include "host.h"
#include "device.h"

#endif /* BLAS_TOOLS_H_ */
