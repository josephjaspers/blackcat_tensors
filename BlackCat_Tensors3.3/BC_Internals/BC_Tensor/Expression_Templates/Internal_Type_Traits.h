/*
 * BlackCat_Internal_Forward_Decls.h
 *
 *  Created on: Jun 12, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_INTERNAL_FORWARD_DECLS_H_
#define BLACKCAT_INTERNAL_FORWARD_DECLS_H_

namespace BC {
namespace internal {

template<class,class,class> class binary_expression;
template<class,class>		class unary_expression;


template<class T> using mathlib_of = std::decay_t<typename T::mathlib_t>;
template<class T> using scalar_of  = std::decay_t<typename T::scalar_t>;

}

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


template<class T> constexpr bool BC_array_move_constructible() {
	return BC_array_move_constructible_overrider<T>::boolean;
}
template<class T> constexpr bool BC_array_copy_constructible() {
	return BC_array_copy_constructible_overrider<T>::boolean;
}
template<class T> constexpr bool BC_array_move_assignable() {
	return BC_array_move_assignable_overrider<T>::boolean;
}
template<class T> constexpr bool BC_array_copy_assignable()	{
	return BC_array_copy_assignable_overrider<T>::boolean;
}


}
#endif /* BLACKCAT_INTERNAL_FORWARD_DECLS_H_ */
