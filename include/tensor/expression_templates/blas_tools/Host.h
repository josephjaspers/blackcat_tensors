/*
 * Host.h
 *
 *  Created on: Apr 24, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_EXPRESSION_TEMPLATES_BLAS_TOOLS_HOST_H_
#define BLACKCAT_EXPRESSION_TEMPLATES_BLAS_TOOLS_HOST_H_

#include "Host_Impl.h"
#include "Common.h"
namespace BC {
namespace exprs {
namespace blas_tools {

struct Host : Common_Tools<Host> {

	template<class Stream, class OutputScalar, class... Scalars>
	static void scalar_multiply(Stream, OutputScalar& eval, Scalars... scalars) {
		eval[0] = BC::exprs::blas_tools::host_impl::calculate_alpha(scalars...);
	}
	template<class Stream, class OutputScalar, class... Scalars>
	static void scalar_multiply(Stream, OutputScalar* eval, Scalars... scalars) {
		eval[0] = BC::exprs::blas_tools::host_impl::calculate_alpha(scalars...);
	}

	template<class value_type, int value>
	static auto scalar_constant() {
		return make_scalar_constant<BC::host_tag, value_type>(value);
	}

};


}
}
}



#endif /* HOST_H_ */
