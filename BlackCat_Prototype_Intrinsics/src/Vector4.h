/*
 * Vector4.h
 *
 *  Created on: Apr 7, 2018
 *      Author: joseph
 */

#ifndef VECTOR4_H_
#define VECTOR4_H_

namespace BC {
namespace Intrinsics {

template<int sz, class T>
struct Vector {

	static constexpr int SIZE() { return sz; };
	static_assert(sz <= 4, "INSTRUCTION SET MAX SIZE == 4");

	T* array;

	Vector(T* array_) : array(array_) {}

	const T& operator [] (int i) const { return array[i]; }
		  T& operator [] (int i) 	   { return array[i]; }







};



}
}



#endif /* VECTOR4_H_ */
