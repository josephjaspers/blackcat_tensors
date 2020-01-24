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
#include "bind.h"

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

template<bool Bool,class Function>
auto constexpr_if(Function function) {
	return constexpr_ternary<Bool>(function, [](){});
}

template<bool Bool, class F1, class F2>
auto constexpr_if(F1 f1, F2 f2) {
	return constexpr_ternary<Bool>(f1, f2);
}

template<bool Bool, class Function>
auto constexpr_else_if(Function function) {
	return [&]() { return constexpr_if<Bool>(function); };
}

template<bool Bool, class F1, class F2>
auto constexpr_else_if(F1 f1, F2 f2) {
	return [&]() { return constexpr_ternary<Bool>(f1, f2); };
}

template<class Function>
auto constexpr_else(Function function) {
	return bc::traits::bind(function);
	//	return detail::constexpr_else_impl<Function>{ function };
}

}
}

#endif /* CONSTEXPRIF_H_ */
