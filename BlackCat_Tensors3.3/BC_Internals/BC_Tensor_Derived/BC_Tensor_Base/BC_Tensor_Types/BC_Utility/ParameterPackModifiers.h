/*
 * ParameterPackModifiers.h
 *
 *  Created on: May 7, 2018
 *      Author: joseph
 */

#ifndef PARAMETERPACKMODIFIERS_H_
#define PARAMETERPACKMODIFIERS_H_

/*
 * Code for altering parameter packs
 * --> Compiler elides these alterations (no run time cost)
 */

namespace BC {
namespace PPack {

	template<int x> struct reverse_ {

	template<class function, class front, class... set>
	__BC_host_inline__ static auto impl(function f, front first, set... s){ return reverse_<x - 1>::impl(f, s..., first); }
	};
	template<> struct reverse_<1>  {

	template<class function, class... set>
	__BC_host_inline__ static auto impl(function f, set... s) {
		return f(s...);
	}
	};


	//reverses a list and calls function f with the parameters
	template<class function, class... set>
	__BC_host_inline__ auto reverse(function f, set... s){
		return reverse_<sizeof...(set)>::impl(f, s...);
	}

	//removes head of a parameter pack and calls function f
	template<class function, class front, class... set>
	__BC_host_inline__ auto pop_head(function f, front first, set... pack) {
		return f(pack...);
	};

	template<class function, class... set>
	auto pop_tail(function f, set... params) {
		 auto pop = [&](auto&... xs) { return pop_head(f, xs...); };
		 auto reverse_initial = [&](auto&... xs) { return reverse(pop, xs...); };

		return reverse_initial(params...);
	};
	//returns tail of parameter pack
	template<class last>
	__BC_host_inline__ auto tail(last& l) -> decltype (l) {
		return l;
	};

	template<class front, class... set>
	__BC_host_inline__ auto tail(front& f, set&... s) -> decltype(f) {
		return tail(s...);
	};
	//returns head of parameter pack
	template<class front, class... set>
	__BC_host_inline__ auto head(front& f, set&... s) -> decltype(f) {
		return f;
	};

}

}




#endif /* PARAMETERPACKMODIFIERS_H_ */
