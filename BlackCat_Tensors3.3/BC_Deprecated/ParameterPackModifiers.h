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

	template<int x> struct reverse_impl {

	template<class function, class front, class... set>
	__BCinline__ static auto impl(function f, front first, set... s){ return reverse_impl<x - 1>::impl(f, s..., first); }
	};

	template<> struct reverse_impl<1>  {

	template<class function, class... set>
	__BCinline__ static auto impl(function f, set... s) {
		return f(s...);
	}
	};

	//reverses a list and calls function f with the parameters
	template<class function, class... set>
	__BCinline__ auto reverse(function f, set... s){
		return reverse_impl<sizeof...(set)>::impl(f, s...);
	}

	//removes head of a parameter pack and calls function f
	template<class function, class front, class... set>
	__BCinline__ auto pop_head(function f, front first, set... pack) {
		return f(pack...);
	};

	//removes tail of a parameter pack and calls function f
	template<class function, class... set>
	__BCinline__ auto pop_tail(function f, set... params) {
		 auto pop = [&](auto&... xs) { return pop_head(f, xs...); };
		 auto reverse_initial = [&](auto&... xs) { return reverse(pop, xs...); };

		return reverse_initial(params...);
	};

	//returns tail of parameter pack
	template<class last>
	__BCinline__ auto tail(last& l) -> decltype (l) {
		return l;
	};

	template<class front, class... set>
	__BCinline__ auto tail(front& f, set&... s) -> decltype(f) {
		return tail(s...);
	};

	//returns head of parameter pack
	template<class front, class... set>
	__BCinline__ auto head(front& f, set&... s) -> decltype(f) {
		return f;
	};

	//push the tail to the front of the parameter pack and then calls function f
	template<int n, class function, class... set>
	__BCinline__ auto queue_tail(function& f, set&... s) {
		return pop_tail(f, tail(s...), s...);
	}


//	template<class function>
//	__BCinline__ auto add_impl(function& f) {
//		return [&](auto... summed_values) {
//			return [&](auto lv, auto... left_values) {
//				return [&](auto rv, auto... right_values) {
////					if constexpr (sizeof...(left_values) == 0 || sizeof...(right_values) == 0)
////						return f(lv + rv, summed_values..., left_values..., right_values...);
////					else
////						return add_impl(f)(summed_values..., lv + rv)(left_values...)(right_values...);
//				};
//			};
//		};
//	}
//
//	template<class function>
//	__BCinline__ auto add(function f) {
//		return [&](auto... left_values) {
//			return [&](auto... right_values) {
//				return add_impl(f)()(left_values...)(right_values...);
//			};
//		};
//	}

}

}




#endif /* PARAMETERPACKMODIFIERS_H_ */