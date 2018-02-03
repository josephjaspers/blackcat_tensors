/*
 * BC_Expression_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef EXPRESSION_BASE_H_
#define EXPRESSION_BASE_H_
namespace BC {

template<class T, class derived>
struct expression {
	using type = derived;
	using scalar_type = T;
};



}

#endif /* EXPRESSION_BASE_H_ */
