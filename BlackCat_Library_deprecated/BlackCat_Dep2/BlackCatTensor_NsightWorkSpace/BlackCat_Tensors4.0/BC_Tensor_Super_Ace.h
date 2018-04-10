/*
 * BC_Tensor_Super_Ace.h
 *
 *  Created on: Nov 17, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_ACE_H_
#define BC_TENSOR_SUPER_ACE_H_

class Tensor_Ace {

	int size() const = 0;
	int degree() const = 0;
	int dimension(int index) const = 0; //base 0

	int rows() const = 0;
	int cols() const = 0;

};



#endif /* BC_TENSOR_SUPER_ACE_H_ */
