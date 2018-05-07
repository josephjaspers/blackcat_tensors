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
 * --> In theory the compiler should be able to do this at compile time (as only the order of the parameters are being changed)
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

		 std::vector<int> ds= {pack...};

		 for (int i : ds) {
			 std::cout << i << " ";
		 }
		 std::cout<<std::endl;

		return f(pack...);
	};

//	//removes tail of parameter packs and calls function f
//	template<class function, class... set>
//	auto pop_tail(function f, set... params) {
//		//read this bottom-up
////
//		//than re-reverses the list and calls the actual function
//		 auto reverse_final = [&](auto&... xs) {
//			return reverse(f, xs...);
//		};
//
//		//then it removes the first element (which is the last element)
//		 auto pop = [&](auto&... xs) {
//			return pop_head(reverse_final, xs...);
//		};
//
//		//first reverse the lists
//		 auto reverse_initial = [&](auto&... xs) {
//
//			return reverse(pop, xs...);
//		};
//
//		return reverse_initial(params...);
//	};

	template<class function, class a, class b, class c, class d>
	__BC_host_inline__ auto pop_tail(function f, a A,b B,c C,d) {
		return f(A, B, C);

	};
	template<class function, class a, class b, class c>
	__BC_host_inline__ auto pop_tail(function f, a A,b B,c) {
		return f(A, B);

	};
	template<class function, class a, class b>
	__BC_host_inline__ auto pop_tail(function f, a A, b) {
		return f(A);
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
