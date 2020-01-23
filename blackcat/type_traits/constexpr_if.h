/*
 * ConstexprIf.h
 *
 *  Created on: Jun 30, 2019
 *      Author: joseph
 */

#ifndef BLACKCAT_CONSTEXPRIF_H_
#define BLACKCAT_CONSTEXPRIF_H_

#include "constexpr_int.h"
#include "get.h"

namespace bc {
namespace traits {

//-------------------------- constexpr if -----------------------//
/*
 * C++ 11/14 version of constexpr if.  (Required because NVCC doesn't support C++17)
 *
 * Accepts a constexpr bool template argument, and one or two functors (that take no arguments)
 * if true, calls and returns the first functor, else the second.
 *
 * Example:
 *
 * int main() {
 *  static constexpr bool value = false;
 *	return bc::meta:constexpr_if<false>(
 *		[]() { std::cout << " constexpr_boolean is true " << std::endl;  return true;},
 *		[]() { std::cout << " constexpr_boolean is false " << std::endl; return false; }
 *	);
 *}
 * output:  constexpr_boolean is false
 */
template<bool cond, class f1, class f2>
auto constexpr_ternary(f1 true_path, f2 false_path) {
	return bc::traits::get<int(!cond)>(true_path, false_path)();
}

namespace detail {

template<bool>
struct constexpr_if_impl
{
	template<class f1>
	static auto impl(f1 path) {
		return path();
	}
};

template<>
struct constexpr_if_impl<false>
{
	template<class f1>
	static void impl(f1 path) {}
};

}

template<bool b,class f>
auto constexpr_if(f path) {
	return detail::constexpr_if_impl<b>::impl(path);
}

template<bool cond, class f1, class f2>
auto constexpr_if(f1 true_path, f2 false_path) {
	return constexpr_ternary<cond>(true_path, false_path);
}

template<bool cond, class f1>
auto constexpr_else_if(f1 f1_) {
	return [&]() { return constexpr_if<cond>(f1_); };
}

template<bool cond, class f1, class f2>
auto constexpr_else_if(f1 f1_, f2 f2_) {
	return [&]() { return constexpr_ternary<cond>(f1_, f2_); };
}


template<class Function>
struct Constexpr_Else {

	Function function;

	template<int ADL=0>
	auto operator () () {
		return function();
	}

	template<int ADL=0>
	auto operator () () const {
		return function();
	}
};

template<class Function>
Constexpr_Else<Function> constexpr_else(Function function) {
	return Constexpr_Else<Function>{ function };
}



}
}




#endif /* CONSTEXPRIF_H_ */
