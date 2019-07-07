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
namespace tensors {
namespace exprs { 
namespace blas_tools {

template<>
struct BLAS_Tools<host_tag> : Common_Tools<BLAS_Tools<host_tag>> {

	template<class Stream, class OutputScalar, class... Scalars>
	static void scalar_multiply(Stream, OutputScalar& eval, Scalars... scalars) {
		eval[0] = BC::tensors::exprs::blas_tools::host_impl::calculate_alpha(scalars...);
	}
	template<class Stream, class OutputScalar, class... Scalars>
	static void scalar_multiply(Stream, OutputScalar* eval, Scalars... scalars) {
		eval[0] = BC::tensors::exprs::blas_tools::host_impl::calculate_alpha(scalars...);
	}

	template<class value_type, int value>
	static auto scalar_constant() {
		return make_scalar_constant<BC::host_tag, value_type>(value);
	}

};


}
}
}
}


#endif /* HOST_H_ */
