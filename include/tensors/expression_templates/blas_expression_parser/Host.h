/*
 * Host.h
 *
 *  Created on: Apr 24, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_EXPRESSION_TEMPLATES_BLAS_TOOLS_HOST_H_
#define BLACKCAT_EXPRESSION_TEMPLATES_BLAS_TOOLS_HOST_H_

#include "Common.h"

namespace BC {
namespace tensors {
namespace exprs { 
namespace blas_expression_parser {

namespace host_detail {

template<class T, class enabler = void>
struct get_value_impl {
	static T& impl(T& scalar) { return scalar; }
	static const T& impl(const T& scalar) { return scalar; }
};
template<class T>
struct get_value_impl<T, std::enable_if_t<T::tensor_dimension==0>>  {
	static auto impl(T scalar) -> decltype(scalar[0]) {
		return scalar[0];
	}
};
template<class T>
struct get_value_impl<T*, void>  {
	static auto impl(const T* scalar) -> decltype(scalar[0])  {
		static constexpr T one = 1;
        return scalar == nullptr ? one : scalar[0];
	}
};

template<class T>
auto get_value(T value) {
	return get_value_impl<T>::impl(value);
}

template<class Scalar>
static auto calculate_alpha(Scalar head) {
	return get_value(head);
}

template<class Scalar, class... Scalars>
static auto calculate_alpha(Scalar head, Scalars... tails) {
	return get_value(head) * calculate_alpha(tails...);
}
}


template<>
struct Blas_Expression_Parser<host_tag> : Common_Tools<Blas_Expression_Parser<host_tag>> {

	template<class Stream, class OutputScalar, class... Scalars>
	static void scalar_multiply(Stream, OutputScalar& eval, Scalars... scalars) {
		eval[0] = host_detail::calculate_alpha(scalars...);
	}

	template<class Stream, class OutputScalar, class... Scalars>
	static void scalar_multiply(Stream, OutputScalar* eval, Scalars... scalars) {
		eval[0] = host_detail::calculate_alpha(scalars...);
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
