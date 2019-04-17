/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef BC_CORE_META_H_
#define BC_CORE_META_H_


#include <type_traits>
#include <tuple>

namespace BC {
namespace meta {
template<class... Ts> using void_t = void;

	template<template<class> class func, class T, class voider=void>
	struct is_detected : std::false_type { };

	template<template<class> class func, class T>
	struct is_detected<func, T, std::enable_if_t<std::is_void<void_t<func<T>>>::value>> : std::true_type { };

	template<template<class> class func, class TestType, class DefaultType, class enabler=void>
	struct conditional_detected {
		using type = DefaultType;
	};
	template<template<class> class func, class TestType, class DefaultType>
	struct conditional_detected<func, TestType, DefaultType, std::enable_if_t<is_detected<func,TestType>::value>> {
		using type = func<TestType>;
	};


	/*
	 * example -- has allocator --> template<class t> using allocator = typename allocator_t;
	 *  --> is_detected<allocator, type_argument>::value
	 *
	 */

	template<template<class> class func, class T>
	static constexpr bool is_detected_v = is_detected<func, T>::value;


    template<class t,class u> static constexpr bool is_same = std::is_same<t,u>::value;

    BCINLINE static constexpr BC::size_t  max(int x) { return x; }
    BCINLINE static constexpr BC::size_t  min(int x) { return x; }

    template<class... integers>
    BCINLINE static constexpr BC::size_t  max(int x, integers... ints) { return x > max (ints...) ? x : max(ints...); }

    template<class... integers>
    BCINLINE static constexpr BC::size_t  min(int x, integers... ints) { return x < min (ints...) ? x : min(ints...); }

    static constexpr BC::size_t  sum(int x) { return x; }

    template<class... integers>
    static constexpr BC::size_t  sum(int x, integers... ints) { return x + sum(ints...); }

    template<class T, class... Ts> using head_t = T;

    template<class T, class... Ts>
    struct seq_contains_impl {
        static constexpr bool value= false;
    };
    template<class T, class U, class... Ts>
        struct seq_contains_impl<T,U,Ts...> {
            static constexpr bool value= std::is_same<T,U>::value || seq_contains_impl<T,Ts...>::value;
        };

    template<class T, class... Ts>
    struct seq_of_impl {
        static constexpr bool value = false;
    };
    template<class T, class U>
    struct seq_of_impl<T,U> {
        static constexpr bool value = std::is_same<T,U>::value;
    };
    template<class T, class U, class... Ts>
    struct seq_of_impl<T,U,Ts...> {
        static constexpr bool value = std::is_same<T,U>::value || seq_of_impl<T,Ts...>::value;
    };



    template<template<class> class function, class... Ts>
    struct seq_is : std::true_type {};

    template<template<class> class function, class T, class... Ts>
        struct seq_is<function, T, Ts...> :
        std::conditional_t<function<T>::value && seq_is<function, Ts...>::value, std::true_type, std::false_type> {};





    template<class T, class U, class... Ts> static constexpr bool seq_of = seq_of_impl<T,U,Ts...>::value;

    template<class... Ts> static constexpr bool seq_contains = seq_contains_impl<Ts...>::value;


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
        static auto impl(f1 path) {
            return path();
        }
    };
    template<bool b,class f>
    auto constexpr_if(f path) {
        return constexpr_if_impl<b>::impl(path);
    }
    template<bool cond, class f1, class f2>
    auto constexpr_if(f1 true_path, f2 false_path) {
        return constexpr_ternary_impl<cond>::impl(true_path, false_path);
    }

    template<bool, class...>
    struct Else_If;

    template<class function>
    struct Else_If<false, function>  {
		const function& f;
		Else_If(const function& f_) : f(f_) { }
		void operator () () const {}
    };
    template<class function>
    struct Else_If<true, function>  {
		function f;
		Else_If(function f_) : f(f_) { }

		template<int ADL=0>
		auto operator () () const {
			return f();
		}
		template<int ADL=0>
		auto operator () () {
			return f();
		}
    };
    template<class TruePath, class FalsePath>
    struct Else_If<false, TruePath, FalsePath>  {
    	TruePath f1;
    	FalsePath f2;
    	Else_If(TruePath f1_, FalsePath f2_)
    		: f1(f1_), f2(f2_) {}

		template<int ADL=0>
		auto operator () () const -> decltype(f2()){
			return f2();
		}
		template<int ADL=0>
		auto operator () () -> decltype(f2()){
			return f2();
		}

    };
    template<class TruePath, class FalsePath>
    struct Else_If<true, TruePath, FalsePath>  {
    	mutable TruePath f1;
    	mutable FalsePath f2;
    	Else_If(TruePath f1_, FalsePath f2_)
    		: f1(f1_), f2(f2_) {}

		template<int ADL=0>
		auto operator () () const {
			return f1();
		}
		template<int ADL=0>
		auto operator () ()  {
			return f1();
		}
    };

    template<bool cond, class... funcs>
    auto else_if(funcs... fs) {
    	return Else_If<cond,funcs...>(fs...);
    }

    //just for readability
    template<class function>
    auto else_(function f) {
    	return f;
    }

    template<class function, class... args>
    struct Bind {
    	mutable function f;
    	std::tuple<args...> m_args;

    	Bind(function f_, args... args_): f(f_), m_args(args_...) {}

    	template<int ADL=0>
    	auto operator () () const {
    		return call();
    	}
    	template<int ADL=0>
    	auto operator () () {
    		return call();
    	}
    private:
    	auto call(args... args_) const {
    		return f(args_...);
    	}

    	template<class... xargs>
    	auto call(xargs... args_)const {
    		return call(args_..., std::get<sizeof...(xargs)>(m_args));
    	}
    };

    template<class function, class... args>
    Bind<function, args...> bind(function f, args... args_) {
    	return  Bind<function,args...>(f, std::forward<args>(args_)...);
    }

    template<class arg>
    auto get_last(const arg& value) -> decltype(value) {
    	return value;
    }
    template<class arg>
	auto get_last(arg& value) -> decltype(value) {
		return value;
	}

    template<class head, class... args>
    auto get_last(head&, args&... args_) -> decltype(get_last(args_...)) {
    	return get_last(args_...);
    }
    template<class head, class... args>
	auto get_last(const head&, const args&... args_) -> decltype(get_last(args_...)) {
		return get_last(args_...);
	}


    template<class> struct DISABLE;
    template<bool x,class T> using only_if = std::conditional_t<x, T, DISABLE<T>>;

    //---------------------

	template<class T> BCINLINE
	T& bc_const_cast(const T& param) {
		return const_cast<T&>(param);
	}

	template<class T> BCINLINE
	T&& bc_const_cast(const T&& param) {
		return const_cast<T&&>(param);
	}

	template<class T> BCINLINE
	T* bc_const_cast(const T* param) {
		return const_cast<T*>(param);
	}


}
}
#endif /* SIMPLE_H_ */
