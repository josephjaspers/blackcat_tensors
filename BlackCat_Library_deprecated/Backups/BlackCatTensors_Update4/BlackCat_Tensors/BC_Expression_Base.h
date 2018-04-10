/*
 * BC_Expression_Base.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_EXPRESSION_BASE_H_
#define BC_EXPRESSION_BASE_H_

/*
 * This is an identity class
 * (no real implementation)
 */

template<class derived>
struct expression {

	using type = derived;

};

template<class derived>
struct array_class {
	using type = derived;
};

#endif /* BC_EXPRESSION_BASE_H_ */
