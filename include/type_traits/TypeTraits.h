/*
 * TypeTraits.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_TYPETRAITS_H_
#define BLACKCAT_TYPETRAITS_H_

#include "Common.h"
#include "ConstexprIf.h"
#include "Bind.h"

namespace BC {
namespace meta {

using namespace BC::meta::common_traits;

	namespace { template<class> struct DISABLE; }
	template<bool x,class T> using only_if = conditional_t<x, T, DISABLE<T>>;

	template<int x> struct Integer { static constexpr int value = x; };
	template<class T> struct Type { using type = T; };
	template<class T, class... Ts> using front_t = T;

	template<class...> using void_t = void;
	template<class...> static constexpr bool true_v  = true;
	template<class...> static constexpr bool false_v = false;

	template<bool cond> using truth_type = conditional_t<cond, true_type, false_type>;
	template<bool cond> using not_truth_type = conditional_t<cond, false_type, true_type>;

	template<class> struct not_type;
	template<> struct not_type<false_type>: true_type {};
	template<> struct not_type<true_type>: false_type {};

	//---------------------------------
	template<class T> using query_allocator_type = typename T::allocator_type;
	template<class T> using query_value_type  	 = typename T::value_type;

	//----------------------------------

	template<template<class> class func, class T, class voider=void>
	struct is_detected : false_type {};

	template<template<class> class func, class T>
	struct is_detected<func, T, enable_if_t<true_v<func<T>>>> : true_type {};

	template<template<class> class func, class T>
	static constexpr bool is_detected_v = is_detected<func, T>::value;

	//----------------------------------

	template<template<class> class func, class TestType, class DefaultType, class enabler=void>
	struct conditional_detected {
		using type = DefaultType;
	};
	template<template<class> class func, class TestType, class DefaultType>
	struct conditional_detected<func, TestType, DefaultType, enable_if_t<is_detected_v<func,TestType>>> {
		using type = func<TestType>;
	};

	template<template<class> class func, class TestType, class DefaultType>
	using conditional_detected_t = typename conditional_detected<func, TestType, DefaultType>::type;

	//----------------------------------

	template<class Function, class voider=void>
	struct is_compileable : false_type {};

	template<class Function>
	struct is_compileable<Function,
			enable_if_t<
					true_v<
						decltype(declval<Function>()())>
				>
			> : true_type {};

	template<class Function>
	static constexpr bool compileable(Function&&) {
		return is_compileable<Function>::value;
	}
	template<class Function>
	static constexpr bool compileable() {
		return is_compileable<Function>::value;
	}

	//----------------------------------


    BCINLINE static constexpr BC::size_t  max(int x) { return x; }
    BCINLINE static constexpr BC::size_t  min(int x) { return x; }

    template<class T, class... Ts> BCINLINE
    static constexpr size_t max(T&& x, Ts&&... xs) {
    	return x > max(xs...) ? x : max(xs...);
    }

    template<class T, class... Ts> BCINLINE
    static constexpr size_t min(T&& x, Ts&&... xs) {
    	return x < min(xs...) ? x : min(xs...);
    }

	//----------------------------------

    template<template<class> class Function, class... Ts>
    struct all: true_type {};

    template<template<class> class Function, class T, class... Ts>
    struct all<Function, T, Ts...>: conditional_t<Function<T>::value, all<Function, Ts...>, false_type> {};

    template<template<class> class Function, class... Ts>
    static constexpr bool all_v = all<Function, Ts...>::value;

    template<template<class> class Function, class... Ts>
    struct any: false_type {};

    template<template<class> class Function, class T, class... Ts>
    struct any<Function, T, Ts...>: conditional_t<Function<T>::value, true_type, any<Function, Ts...>> {};

    template<template<class> class Function, class... Ts>
    static constexpr bool any_v = any<Function, Ts...>::value;

    template<template<class> class Function, class... Ts>
    struct none: conditional_t<any<Function, Ts...>::value, false_type, true_type> {};

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


    //---------------------

    //Only required for NVCC (can't pass std::pair to cuda kernels)
	template<class T, class U>
	struct Pair {
		T first;
		U second;
	};
	template<class T, class U> BCINLINE
	Pair<T, U> make_pair(T t, U u) { return Pair<T, U>{t, u}; }

    //---------------------

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

	template<class T> //for removing GCC warning "const modifier doesn't do anything"
	using apply_const_t = conditional_t<is_const<T>::value, T, const T>;


	template<class T> BCINLINE
	apply_const_t<T>& auto_apply_const(T& param) { return param; }
	template<class T> BCINLINE
	apply_const_t<T>&& auto_apply_const(T&& param) { return param; }
	template<class T> BCINLINE
	apply_const_t<T>* auto_apply_const(T* param) { return param; }

}
}


#endif /* TYPETRAITS_H_ */
