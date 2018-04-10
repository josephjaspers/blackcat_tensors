/*
  * Tensor_Super.h
 *
 *  Created on: Oct 11, 2017
 *      Author: joseph
 */

#ifndef TENSOR_SUPER_H_
#define TENSOR_SUPER_H_


template<typename T, class Oper>
class Tensor_Super {

	T* tensor;

	const bool OWNERSHIP;
	unsigned order;
	unsigned sz;

	unsigned size() 							const = 0;
	unsigned rows() 							const = 0;
	unsigned cols() 							const = 0;
	unsigned depth()							const = 0;
	bool isMatrix() 							const = 0;
	bool isSquare() 							const = 0;
	bool isVector()								const = 0;
	bool isScalar() 							const = 0;
	unsigned rank(unsigned rank_index) 			const = 0;
	unsigned degree() 							const = 0;

	void zero() = 0;
	void fill() = 0;
	void randomize() = 0;

	void reset() = 0;
};


#endif /* TENSOR_SUPER_H_ */
