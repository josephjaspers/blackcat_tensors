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
struct dominant_type<lv, rv, std::enable_if_t<(lv::DIMS() < rv::DIMS())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};
template<class lv, class rv>
struct dominant_type<lv, rv, std::enable_if_t<(lv::DIMS() == rv::DIMS())>> {

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
struct inferior_type<lv, rv, std::enable_if_t<(lv::DIMS() > rv::DIMS())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};

template<class lv, class rv>
struct inferior_type<lv, rv, std::enable_if_t<(lv::DIMS() == rv::DIMS())>> {

	__BCinline__ static const auto& shape(const lv& l, const rv& r) {
		return r;
	}
};

template<class T, int size>
struct stack_array : stack_array<T, size - 1> {

	template<class... values>
	stack_array(T val, values... integers) : stack_array<T, size - 1>(integers...), dim(val) {}
	stack_array() = default;

	T dim = 0;
	operator int() const { return dim; }
	operator const int*() const { return &dim; }

	auto& next() const { return static_cast<const stack_array<T, size - 1>& >(*this); }

	const T& operator [] (int i) const {
		return (&dim)[-i];
	}
	T& operator [] (int i) {
		return (&dim)[-i];
	}
};

template<class T>
struct stack_array<T, 0> {
	T operator [](int i) const {
		std::cout << " out of bounds " << std::endl;
		return 0;
	}
};

template<class ref>
struct reference_array {
	ref value;

	reference_array(ref a_) :
			value(a_) {
	}

	const auto operator [](int i) const {
		return value.dimension(i);
	}
	auto operator [](int i) {
		return value.dimension(i);
	}
};


template<class T, class... vals>
auto array(T front, vals... values) {
	return stack_array<T, sizeof...(values) + 1>(front, values...);
}

template<class T>
auto ref_array(T data) {
	return reference_array<T>(data);
}


template<bool b>
struct trueFalse {
	using type = std::conditional<b, std::true_type, std::false_type>;
};
template<bool b>
using tf = typename trueFalse<b>::type;



#endif /* EXPRESSION_UTILITY_STRUCTS_H_ */
