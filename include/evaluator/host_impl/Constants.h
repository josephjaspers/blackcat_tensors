/*
 * CPU_Constants.h
 *
 *  Created on: Jun 10, 2018
 *      Author: joseph
 */

#ifndef CPU_CONSTANTS_H_
#define CPU_CONSTANTS_H_
namespace BC {
namespace evaluator {
namespace host_impl {

template<class T, class enabler = void>
struct get_value_impl {
    static auto impl(T scalar) {
        return scalar;
    }
};
template<class T>
struct get_value_impl<T, std::enable_if_t<!std::is_same<decltype(std::declval<T&>()[0]), void>::value>>  {
    static auto impl(T scalar) {
        return scalar[0];
    }
};


template<class core_lib>
struct Constants {

    template<class T>
    static auto get_value(T scalar) {
        return get_value_impl<T>::impl(scalar);
    }

    template<class U, class T, class V>
    static void scalar_mul(U& eval, T a, V b) {
        eval = get_value(a) * get_value(b);
    }


    template<class T>
    static T static_allocate(T value) {
        return T(value);
    }
};
}
}
}


#endif /* CPU_CONSTANTS_H_ */
