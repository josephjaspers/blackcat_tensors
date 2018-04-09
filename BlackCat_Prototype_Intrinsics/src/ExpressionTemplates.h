/*
 * ExpressionTemplates.h
 *
 *  Created on: Apr 7, 2018
 *      Author: joseph
 */

#ifndef EXPRESSIONTEMPLATES_H_
#define EXPRESSIONTEMPLATES_H_


template<class operation, class lv, class rv>
struct expression {

	operation oper;
	lv& left;
	rv& right;

	auto operator [] (int i) const { return oper(left[i], right[i]); }
	static_assert(lv::SIZE() == rv::SIZE(), "MUST BE SAME SIZE");

};



#endif /* EXPRESSIONTEMPLATES_H_ */
