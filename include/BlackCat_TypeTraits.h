/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_CORE_META_H_
#define BC_CORE_META_H_

#include "BlackCat_Common.h"
#include <type_traits>
#include <tuple>

namespace BC {
namespace meta {
namespace {
	using std::enable_if_t;
	using std::true_type;
	using std::false_type;
	using std::is_same;
	using std::declval;
	using std::is_const;
	using std::conditional_t;
	using std::tuple;
	using std::forward;
}
	template<int x>
	struct Integer { static constexpr int value = x; };

	template<class T, class... Ts>
	using front_t = T;

	template<class... Ts>
	using void_t = void;


	template<class T, class U>
	struct Pair {
		T left;
		U right;
	};

	template<class T, class U>
	Pair<T, U> make_pair(T&& t, U&& u) {
		return {t, u};
	}

	template<class...> static constexpr bool true_v  = true;
	template<class...> static constexpr bool false_v = false;
	template<class...> using void_t = void;

	//----------------------------------

	template<template<class> class func, class T, class voider=void>
	struct is_detected : false_type { };

	template<template<class> class func, class T>
	struct is_detected<func, T, enable_if_t<true_v<func<T>>>> : true_type { };

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

    template<class... integers>
    BCINLINE static constexpr BC::size_t  max(int x, integers... ints) { return x > max (ints...) ? x : max(ints...); }

    template<class... integers>
    BCINLINE static constexpr BC::size_t  min(int x, integers... ints) { return x < min (ints...) ? x : min(ints...); }

	//----------------------------------

    template<class T, class... Ts>
    struct seq_contains_impl {
        static constexpr bool value= false;
    };
    template<class T, class U, class... Ts>
	struct seq_contains_impl<T,U,Ts...> {
		static constexpr bool value= is_same<T,U>::value || seq_contains_impl<T,Ts...>::value;
	};

    template<class T, class... Ts>
    struct seq_of_impl : false_type{};

    template<class T, class U>
    struct seq_of_impl<T,U> {
        static constexpr bool value = is_same<T,U>::value;
    };
    template<class T, class U, class... Ts>
    struct seq_of_impl<T,U,Ts...> {
        static constexpr bool value = is_same<T,U>::value || seq_of_impl<T,Ts...>::value;
    };

    template<template<class> class function, class... Ts>
    struct seq_is : true_type {};

    template<template<class> class function, class T, class... Ts>
	struct seq_is<function, T, Ts...> :
        conditional_t<function<T>::value && seq_is<function, Ts...>::value, true_type, false_type> {};

    template<class T, class U, class... Ts>
    static constexpr bool seq_of = seq_of_impl<T,U,Ts...>::value;

    template<class... Ts> static constexpr bool
    seq_contains = seq_contains_impl<Ts...>::value;


    template<int index, class... Ts>
    struct get_impl;

    template<int index>
    struct get_impl<index>{
        template<class T, class... Ts>
        static auto impl(T head, Ts... params) {
             return get_impl<index - 1>::impl(params...);
        }
    };
    template<>
    struct get_impl<0>{
        template<class T, class... Ts>
        static auto impl(T head, Ts... params) {
             return head;
        }
    };
    template<int index, class... Ts>
    auto get(Ts... params) {
        return get_impl<index>::impl(params...);
    }


    template<bool>
    struct constexpr_ternary_impl {
        template<class f1, class f2>
        static auto impl(f1 true_path, f2 false_path) {
            return true_path();
        }
    };

    template<>
    struct constexpr_ternary_impl<false> {
        template<class f1, class f2>
        static auto impl(f1 true_path, f2 false_path) {
            return false_path();
        }
    };

    template<bool cond, class f1, class f2>
    auto constexpr_ternary(f1 true_path, f2 false_path) {
        return constexpr_ternary_impl<cond>::impl(true_path, false_path);
    }

    template<bool>
    struct constexpr_if_impl {
        template<class f1>
        static auto impl(f1 path) {
            return path();
        }
    };
    template<>
    struct constexpr_if_impl<false> {
        template<class f1>
        static void impl(f1 path) {}
    };
    template<bool b,class f>
    auto constexpr_if(f path) {
        return constexpr_if_impl<b>::impl(path);
    }
    template<bool cond, class f1, class f2>
    auto constexpr_if(f1 true_path, f2 false_path) {
        return constexpr_ternary_impl<cond>::impl(true_path, false_path);
    }

    template<bool, class...> struct Constexpr_Else_If;

    template<class f1, class f2>
    struct Constexpr_Else_If<true, f1, f2> {
    	mutable f1 f1_;
    	f2 f2_;

    	template<int ADL=0>
    	auto operator () () {
    		return f1_();
    	}
    	template<int ADL=0>
    	auto operator () () const {
    		return f1_();
    	}
    };
    template<class f1, class f2>
    struct Constexpr_Else_If<false, f1, f2> {
    	f1 f1_;
    	f2 f2_;

    	template<int ADL=0>
    	auto operator () () {
    		return f2_();
    	}
    	template<int ADL=0>
    	auto operator () () const {
    		return f2_();
    	}
    };
    template<class f1>
    struct Constexpr_Else_If<true, f1> {
    	f1 f1_;

    	template<int ADL=0>
    	auto operator () () {
    		return f1_();
    	}
    	template<int ADL=0>
    	auto operator () () const {
    		return f1_();
    	}
    };
    template<class f1>
    struct Constexpr_Else_If<false, f1> {
    	f1 f1_;

    	template<int ADL=0>
    	void operator () () {
    	}
    	template<int ADL=0>
    	void operator () () const {
    	}
    };

    template<bool cond, class f1>
    auto constexpr_else_if(f1 f1_) {
    	return Constexpr_Else_If<cond, f1>{f1_};
    }
    template<bool cond, class f1, class f2>
    auto constexpr_else_if(f1 f1_, f2 f2_) {
    	return Constexpr_Else_If<cond, f1, f2>{f1_, f2_};
    }

    template<class f1>
    struct Else_ {
    	f1 f1_;

    	template<int adl=0>
    	auto operator () () {
    		return f1_();
    	}
    	template<int adl=0>
    	auto operator () () const {
    		return f1_();
    	}
    };

    template<class f1>
    Else_<f1> else_(f1 f1_) {
    	return Else_<f1>{f1_};
    }

    template<class> struct DISABLE;
    template<bool x,class T> using only_if = conditional_t<x, T, DISABLE<T>>;

    template<class Function, class... args>
    struct Bind : tuple<args...> {
    	Function f;

    	Bind(Function f, args... args_)
    	: tuple<args...>(args_...), f(f) {}

    	template<int ADL=0>
    	auto operator () () {
    		return call(conditional_t<sizeof...(args) == 0, true_type, false_type>());
    	}
    	template<int ADL=0>
    	auto operator () () const {
    		return call(conditional_t<sizeof...(args) == 0, true_type, false_type>());
    	}
    private:
    	template<class... args_>
    	auto call(true_type, args_... params) {
    		return f(params...);
    	}
    	template<class... args_>
    	auto call(false_type, args_... params) {
    		return call(
    				conditional_t<sizeof...(args_) + 1 == sizeof...(args), true_type, false_type>(),
    				forward<args_>(params)...,
    				get<sizeof...(args_)>(static_cast<tuple<args...>&>(*this)));
    	}
    	template<class... args_>
    	auto call(true_type, args_... params) const {
    		return f(forward<args_>(params)...);
    	}
    	template<class... args_>
    	auto call(false_type, args_... params) const {
    		return call(
    				conditional_t<sizeof...(args_) + 1 == sizeof...(args), true_type, false_type>(),
    				forward<args_>(params)...,
    				get<sizeof...(args_)>(static_cast<tuple<args...>&>(*this)));
    	}

    };
    template<class Function, class... Args>
    Bind<Function, Args...> bind(Function f, Args... args) {
    	return {f, args...};
    }

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

	template<class T>
	using apply_const_t = conditional_t<is_const<T>::value, T, const T>;


}
}
#endif /* SIMPLE_H_ */
