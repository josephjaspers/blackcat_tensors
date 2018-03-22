/*
 * Expression_Utility_Structs.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_UTILITY_STRUCTS_H_
#define EXPRESSION_UTILITY_STRUCTS_H_

#include "BlackCat_Internal_Definitions.h"

//returns the class with the higher_order rank
template<class lv, class rv, class left = void>
struct dominant_type {
	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return l;
	}
};
template<class lv, class rv>
struct dominant_type<lv, rv, std::enable_if_t<(lv::RANK() < rv::RANK())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};
template<class lv, class rv>
struct dominant_type<lv, rv, std::enable_if_t<(lv::RANK() == rv::RANK())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};
//returns the class with the lower order rank
template<class lv, class rv, class left = void>
struct inferior_type {
	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return l;
	}
};
template<class lv, class rv>
struct inferior_type<lv, rv, std::enable_if_t<(lv::RANK() > rv::RANK())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};

template<class lv, class rv>
struct inferior_type<lv, rv, std::enable_if_t<(lv::RANK() == rv::RANK())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};

template<int size>
struct stack_list : stack_list<size - 1> {
	template<class... values>
	stack_list(int val, values... integers) : stack_list<size - 1>(integers...), dim(val) {}

	int dim;
	operator int() const { return dim; }
	auto& next() const { return static_cast<const stack_list<size - 1>&>(*this); }

	int operator [] (int i) const {
		return (&dim)[-i];
	}
};

template<>
struct stack_list<0> {	int operator [] (int i) const {
	std::cout << " out of bounds " << std::endl;
	return 0;
}};
template<class... integers>
auto generateDimList(integers... values) {

	return stack_list<sizeof...(values)>(values...);
}





#endif /* EXPRESSION_UTILITY_STRUCTS_H_ */
