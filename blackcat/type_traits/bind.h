/*
 * Bind.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_BIND_H_
#define BLACKCAT_BIND_H_

#include "type_traits.h"
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
template<class Function, class... Args>
struct Bind: tuple<Args...>
{
	Function func;

	Bind(Function func, Args... AdditionalArgs):
		tuple<Args...>(AdditionalArgs...), func(func) {}

	static constexpr int num_args = sizeof...(Args);

private:

	template<class... TupleArgs>
	auto call(true_type, TupleArgs&&... params) const
		-> decltype(func(forward<TupleArgs>(params)...))
	{
		return func(forward<TupleArgs>(params)...);
	}

	template<class... TupleArgs>
	auto call(true_type, TupleArgs&&... params)
		-> decltype(func(forward<TupleArgs>(params)...))
	{
		return func(forward<TupleArgs>(params)...);
	}

	template<class... TupleArgs, int ArgCount=sizeof...(TupleArgs)>
	auto call(false_type, TupleArgs&&... params) const
		-> decltype(call(
			truth_type<ArgCount + 1 == num_args>(),
			forward<TupleArgs>(params)...,
			std::get<ArgCount>(*this)))
	{
		return call(
			truth_type<ArgCount + 1 == num_args>(),
			forward<TupleArgs>(params)...,
			std::get<ArgCount>(*this));
	}

	template<class... TupleArgs, int ArgCount=sizeof...(TupleArgs)>
	auto call(false_type, TupleArgs&&... params)
	-> decltype(call(
			truth_type<ArgCount + 1 == num_args>(),
			forward<TupleArgs>(params)...,
			std::get<ArgCount>(*this)))
	{
		return call(
			truth_type<ArgCount + 1 == num_args>(),
			forward<TupleArgs>(params)...,
			std::get<ArgCount>(*this));
	}

public:

	template<int ADL=0, class=std::enable_if_t<ADL==0>>
	auto operator () ()
		-> decltype(call(truth_type<num_args == 0 && ADL==0>()))
	{
		return call(truth_type<num_args == 0>());
	}

	template<int ADL=0, class=std::enable_if_t<ADL==0>>
	auto operator () () const
		-> decltype(call(truth_type<num_args == 0 && ADL==0>()))
	{
		return call(truth_type<num_args == 0>());
	}
};

template<class Function, class... Args>
Bind<Function, Args&&...> bind(Function function, Args&&... args) {
	return { function, std::forward<Args>(args)... };
}

}
}




#endif /* BIND_H_ */
