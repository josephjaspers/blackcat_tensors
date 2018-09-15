/*
 * Expression_Utility_Structs.h
 *
 *  Created on: Mar 22, 2018
 *      Author: joseph
 */

#ifndef EXPRESSION_UTILITY_STRUCTS_H_
#define EXPRESSION_UTILITY_STRUCTS_H_

#include "Utility_Structs.h"
#include <iostream>

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



#endif /* EXPRESSION_UTILITY_STRUCTS_H_ */
