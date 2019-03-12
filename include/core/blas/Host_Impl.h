/*
 * Host_Impl.h
 *
 *  Created on: Mar 10, 2019
 *      Author: joseph
 */

#ifndef BC_CORE_BLAS_HOST_IMPL_H_
#define BC_CORE_BLAS_HOST_IMPL_H_

namespace BC {
namespace blas {
namespace host_impl {

template<class T, class enabler = void>
struct get_value_impl {
    static T& impl(T& scalar) {
        return scalar;
    }
	static const T& impl(const T& scalar) {
		return scalar;
	}
};
template<class T>
struct get_value_impl<T, std::enable_if_t<T::ITERATOR==0>>  {
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
}
}



#endif /* HOST_IMPL_H_ */
