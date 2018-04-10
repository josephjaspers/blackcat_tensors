/*
 * ArrayType_Base.h
 *
 *  Created on: Dec 19, 2017
 *      Author: joseph
 */

#ifndef ARRAYTYPE_BASE_H_
#define ARRAYTYPE_BASE_H_

namespace BC {

template<class T, class deriv>
struct ArrayType  {
	using type = deriv;
};

}

#endif /* ARRAYTYPE_BASE_H_ */
