/*
 * Bind.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_BIND_H_
#define BLACKCAT_BIND_H_

#include "common.h"
#include "../common.h"

namespace bc {
namespace traits {

using namespace bc::traits::common;

/**
 * Similar to std::bind but the evaluation of the function
 * in respect to its bound arguments are deduced if and only if the function is called.
 * IE
 *
 * 	auto func = bc::traits::bind([](int x) {}, std::vector<double>());
 *
 * 	will compile, even though a std::vector is not a valid argument
 * 	for the given lambda.
 *
 */

namespace {
template<int index>
struct get_impl {
    template<class T, class... Ts> BCINLINE
    static auto impl(T&& head, Ts&&... params)
    	-> decltype(get_impl<index - 1>::impl(params...))
    {
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

template<class Function, class... Args>
struct Bind : tuple<Args...>
{
	Function f;

	Bind(Function f, Args... AdditionalArgs):
		tuple<Args...>(AdditionalArgs...), f(f) {}

	static constexpr int num_args = sizeof...(Args);

	template<int ADL=0>
	auto operator () () {
		return call(conditional_t<num_args == 0, true_type, false_type>());
	}

	template<int ADL=0>
	auto operator () () const {
		return call(conditional_t<num_args == 0, true_type, false_type>());
	}

private:

	template<class... AdditionalArgs>
	auto call(true_type, AdditionalArgs&&... params) {
		return f(params...);
	}

	template<class... AdditionalArgs>
	auto call(false_type, AdditionalArgs&&... params) {
		return call(
				conditional_t<
						sizeof...(AdditionalArgs) + 1 == num_args,
						true_type,
						false_type>(),
				forward<AdditionalArgs>(params)...,
				get<sizeof...(AdditionalArgs)>(
						static_cast<tuple<Args...>&>(*this)));
	}

	template<class... AdditionalArgs>
	auto call(true_type, AdditionalArgs&&... params) const {
		return f(forward<AdditionalArgs>(params)...);
	}

	template<class... AdditionalArgs>
	auto call(false_type, AdditionalArgs&&... params) const {
		return call(
				conditional_t<
						sizeof...(AdditionalArgs) + 1 == num_args,
						true_type,
						false_type>(),
				forward<AdditionalArgs>(params)...,
				get<sizeof...(AdditionalArgs)>(
						static_cast<tuple<Args...>&>(*this)));
	}

};
template<class Function, class... Args>
Bind<Function, Args...> bind(Function&& f, Args&&... args) {
	return {std::forward<Function>(f), std::forward<Args>(args)...};
}

}
}




#endif /* BIND_H_ */
