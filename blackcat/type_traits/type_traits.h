/*

 * TypeTraits.h
 *
 *  Created on: Jun 30, 2019
 *	  Author: joseph
 */

#ifndef BLACKCAT_TYPETRAITS_H_
#define BLACKCAT_TYPETRAITS_H_

#include "common.h"
#include "constexpr_if.h"
#include "bind.h"

namespace bc {
namespace traits {

using namespace bc::traits::common;

namespace detail { template<class> struct DISABLE; }

template<bool x,class T>
using only_if = conditional_t<x, T, detail::DISABLE<T>>;

template<int x> struct Integer { static constexpr int value = x; };

template<class...> using void_t = void;
template<class...> static constexpr bool true_v  = true;
template<class...> static constexpr bool false_v = false;

template<class T>
struct using_type {
	using type = T;
};

/**
 * true_call and false_call are CUDA-supported alternatives for true_v and false_v
 * -- CUDA does not support using 'host_variables' on the device (including static-constexpr values)
 */
template<class T>
BCINLINE static constexpr bool true_call() { return true; }

template<class T>
BCINLINE static constexpr bool false_call() { return false; }



template<class...> using void_t = void;
template<class...> using true_t  = true_type;
template<class...> using false_t = false_type;

template<bool cond>
using truth_type = conditional_t<cond, true_type, false_type>;

template<bool cond>
using not_type = conditional_t<cond, false_type, true_type>;

//--------------------------------
template<class From, class To>
struct propagate_const: using_type<To> {};

template<class From, class To>
struct propagate_const<const From, To>: using_type<const To> {};

template<class From, class To>
using propagate_const_t = typename propagate_const<From, To>::type;

//----------------------------------

template<template<class> class func, class T, class voider=void>
struct is_detected: false_type {};

template<template<class> class func, class T>
struct is_detected<func, T, enable_if_t<true_v<func<T>>>>: true_type {};

template<template<class> class func, class T>
static constexpr bool is_detected_v = is_detected<func, T>::value;

//----------------------------------

template<
	template<class> class func,
	class TestType,
	class DefaultType=void,
	class enabler=void>
struct conditional_detected: using_type<DefaultType> {};

template<
	template<class> class func,
	class TestType,
	class DefaultType>
struct conditional_detected<
		func,
		TestType,
		DefaultType,
		enable_if_t<is_detected_v<func,TestType>>>:
	using_type<func<TestType>> {};

template<template<class> class func, class TestType, class DefaultType>
using conditional_detected_t =
		typename conditional_detected<func, TestType, DefaultType>::type;

//----------------------------------


template<class T>
BCINLINE static constexpr bc::size_t  max(const T& x) { return x; }

template<class T>
BCINLINE static constexpr bc::size_t  min(const T& x) { return x; }

template<class T, class... Ts> BCINLINE
static constexpr size_t max(const T& x, const Ts&... xs) {
	return x > max(xs...) ? x : max(xs...);
}

template<class T, class... Ts> BCINLINE
static constexpr size_t min(const T& x, const Ts&... xs) {
	return x < min(xs...) ? x : min(xs...);
}

//----------------------------------

template<template<class> class Function, class... Ts>
struct all: true_type {};

template<template<class> class Function, class T, class... Ts>
struct all<Function, T, Ts...>:
	conditional_t<Function<T>::value, all<Function, Ts...>, false_type> {};

template<template<class> class Function, class... Ts>
static constexpr bool all_v = all<Function, Ts...>::value;

template<template<class> class Function, class... Ts>
struct any: false_type {};

template<template<class> class Function, class T, class... Ts>
struct any<Function, T, Ts...>:
	conditional_t<Function<T>::value, true_type, any<Function, Ts...>> {};

template<template<class> class Function, class... Ts>
static constexpr bool any_v = any<Function, Ts...>::value;

template<template<class> class Function, class... Ts>
struct none: truth_type<any<Function, Ts...>::value> {};

template<template<class> class Function, class... Ts>
static constexpr bool none_v = none<Function, Ts...>::value;

// ---------------------

template<class T, class... Ts>
class sequence_of {
	template<class U> using is_same_ = std::is_same<U, T>;
public:
	static constexpr bool value = all_v<is_same_, Ts...>;
};

template<class... Ts>
static constexpr bool sequence_of_v = sequence_of<Ts...>::value;


template<class T, class... Ts>
class sequence_contains {
	template<class U> using is_same_ = std::is_same<U, T>;
public:
	static constexpr bool value = any_v<is_same_, Ts...>;
};

template<class... Ts>
static constexpr bool sequence_contains_v = sequence_contains<Ts...>::value;


template<class T, class... Ts>
class sequence_first: using_type<T> {};

template<class T, class... Ts>
class sequence_last: using_type<typename sequence_last<Ts...>::type> {};

template<class T>
class sequence_last<T>: using_type<T> {};


// -------- new addition, other code will use static_cast
template<class T>
using query_derived_type = typename T::derived_type;

template<template<class...> class Type, class Derived, class... Ts>
auto& derived_cast(const Type<Derived, Ts...>& t) {
	using derived_type = conditional_detected_t<
			query_derived_type, Type<Derived,Ts...>, Derived>;

	return static_cast<const derived_type&>(t);
}

template<template<class...> class Type, class Derived, class... Ts>
auto& derived_cast(Type<Derived, Ts...>& t) {
	using derived_type = conditional_detected_t<
			query_derived_type, Type<Derived,Ts...>, Derived>;

	return static_cast<derived_type&>(t);
}

template<
	class Type,
	class=std::enable_if_t<is_detected_v<query_derived_type, Type>>>
const auto& derived_cast(const Type& t) {
	return static_cast<const typename Type::derived_type&>(t);
}

template<
	class Type,
	class=std::enable_if_t<is_detected_v<query_derived_type, Type>>>
auto& derived_cast(Type& t) {
	return static_cast<typename Type::derived_type&>(t);
}

template<class T> BCINLINE
T& auto_remove_const(const T& param) {
	return const_cast<T&>(param);
}

template<class T> BCINLINE
T&& auto_remove_const(const T&& param) {
	return const_cast<T&&>(param);
}

template<class T> BCINLINE
T* auto_remove_const(const T* param) {
	return const_cast<T*>(param);
}

template<class T>
using apply_const_t = conditional_t<is_const<T>::value, T, const T>;


template<class T> BCINLINE
apply_const_t<T>& auto_apply_const(T& param) { return param; }
template<class T> BCINLINE
apply_const_t<T>&& auto_apply_const(T&& param) { return param; }
template<class T> BCINLINE
apply_const_t<T>* auto_apply_const(T* param) { return param; }


template<class T> using query_value_type = typename T::value_type;
template<class T> using query_allocator_type = typename T::allocator_type;
template<class T> using query_system_tag = typename T::system_tag;
template<class T> using query_get_stream = decltype(std::declval<T>().get_stream);
template<class T> using query_get_allocator = decltype(std::declval<T>().get_allocator);

class None {};

template<class T>
struct common_traits {

	using defines_value_type =
			truth_type<is_detected_v<query_value_type, T>>;

	using defines_allocator_type =
			truth_type<is_detected_v<query_allocator_type, T>>;

	using defines_system_tag =
			truth_type<is_detected_v<query_system_tag, T>>;

	using defines_get_stream =
			truth_type<is_detected_v<query_get_stream, T>>;

	using defines_get_allocator =
			truth_type<is_detected_v<query_get_allocator, T>>;

	using type = T;

	using value_type =
			traits::conditional_detected_t<query_value_type, T, None>;
	using allocator_type =
			traits::conditional_detected_t<query_allocator_type, T, None>;
	using system_tag =
			traits::conditional_detected_t<query_system_tag, T, host_tag>;

	static auto select_on_get_allocator(const T& type)
	{
		static_assert(
				std::is_same<system_tag, host_tag>::value ||
						std::is_same<system_tag, device_tag>::value,
				"SystemTag Mismatch");

		return traits::constexpr_ternary<defines_get_allocator::value>(
				bc::traits::bind([](const auto& type) {
						return type.get_allocator();
				}, type),
				[]() {
					return allocator_type();
				}
		);
	}
};

template<class T>
struct Type : common_traits<T> {};


}

using traits::common_traits; //import common_traits into BC namespace

}


#endif /* TYPETRAITS_H_ */
