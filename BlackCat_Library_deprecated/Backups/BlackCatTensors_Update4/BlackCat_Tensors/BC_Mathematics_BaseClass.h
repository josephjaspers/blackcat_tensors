/*
 * BC_Mathematics_BaseClass.h
 *
 *  Created on: Dec 4, 2017
 *      Author: joseph
 */

#ifndef BC_MATHEMATICS_BASECLASS_H_
#define BC_MATHEMATICS_BASECLASS_H_


template<class derived_math_library>
class BlackCat_Tensor_MathLibrary {
	using type = derived_math_library;
};



#endif /* BC_MATHEMATICS_BASECLASS_H_ */
