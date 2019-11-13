/*
 * Bind.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_BIND_H_
#define BLACKCAT_BIND_H_

#include "Common.h"

namespace BC {
namespace traits {

using namespace BC::traits::common;

/**
 * Similar to std::bind but the evaluation of the function
 * in respect to its bound arguments are deduced if and only if the function is called.
 * IE
 *
 * 	auto func = BC::traits::bind([](int x) {}, std::vector<double>());
 *
 * 	will compile, even though a std::vector is not a valid argument
 * 	for the given lambda.
 *
 */

namespace {
template<int index>
struct get_impl {
    template<class T, class... Ts> BCINLINE
    static auto impl(T&& head, Ts&&... params) -> decltype(get_impl<index - 1>::impl(params...)) {
         return get_impl<index - 1>::impl(params...);
    }
};

template<>
struct get_impl<0> {
    template<class T, class... Ts> BCINLINE
    static T&& impl(T&& head, Ts&&... params) {
         return head;
    }
};

}

/** Returns the Nth argument in the argument_pack
 */
template<int index, class... Ts> BCINLINE
auto get(Ts&&... params) -> decltype(get_impl<index>::impl(params...)) {
	return get_impl<index>::impl(params...);
}

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
Bind<Function, Args...> bind(Function&& f, Args&&... args) {
	return {std::forward<Function>(f), std::forward<Args>(args)...};
}

}
}




#endif /* BIND_H_ */
