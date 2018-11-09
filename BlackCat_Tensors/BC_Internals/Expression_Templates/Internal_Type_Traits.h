/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BLACKCAT_INTERNAL_FORWARD_DECLS_H_
#define BLACKCAT_INTERNAL_FORWARD_DECLS_H_
#include <type_traits>
namespace BC {
namespace internal {

template<class,class,class> class Binary_Expression;
template<class,class>        class Unary_Expression;

template<class T> using allocator_of = std::decay_t<typename T::allocator_t>;
template<class T> using scalar_of  = std::decay_t<typename T::scalar_t>;


template<class T, class enabler=void>
struct BC_array_move_constructible_overrider {
    static constexpr bool boolean = false;
};
template<class T, class enabler=void>
struct BC_array_copy_constructible_overrider {
    static constexpr bool boolean = false;
};
template<class T, class enabler=void>
struct BC_array_move_assignable_overrider {
    static constexpr bool boolean = false;
};
template<class T, class enabler=void>
struct BC_array_copy_assignable_overrider {
    static constexpr bool boolean = false;
};

//all expressions and 'views' (slices/reshape/chunk etc) are 'rvalues'
//anything that owns the internal memptr is an 'lvalue'.
template<class T, class enabler=void>
struct BC_lvalue_type_overrider {
    static constexpr bool boolean = false;
};

template<class T, class enabler=void>
struct BC_iterable_overrider : std::false_type {};

template<class T> constexpr bool BC_array_move_constructible() {
    return BC_array_move_constructible_overrider<std::decay_t<T>>::boolean;
}
template<class T> constexpr bool BC_array_copy_constructible() {
    return BC_array_copy_constructible_overrider<std::decay_t<T>>::boolean;
}
template<class T> constexpr bool BC_array_move_assignable() {
    return BC_array_move_assignable_overrider<std::decay_t<T>>::boolean;
}
template<class T> constexpr bool BC_array_copy_assignable()    {
    return BC_array_copy_assignable_overrider<std::decay_t<T>>::boolean;
}
template<class T> constexpr bool BC_lvalue_type()    {
    return BC_lvalue_type_overrider<std::decay_t<T>>::boolean;
}
template<class T> constexpr bool BC_rvalue_type()    {
    return !BC_lvalue_type<T>();
}

template<class T> constexpr bool BC_iterable() {
    return BC_iterable_overrider<T>::boolean;
}
}

}
#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
