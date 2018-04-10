/*
 * BC_Tensor_Primary_Vector.h
 *
 *  Created on: Nov 25, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_PRIMARY_VECTOR_H_
#define BC_TENSOR_PRIMARY_VECTOR_H_

#include "BC_Tensor_Super_Jack.h"
#include "BC_Tensor_Primary_Scalar.h"

//Illegal generic type
template<class T, class ml, int ...dims>
class Vector : public Tensor_Jack<T, ml, dims...> {
private:
	Vector<T, ml, dims...>() = delete;
};

//Standard column Vector
template<class T, class ml, int rows>
class Vector<T, ml, rows> : public Tensor_Jack<T, ml, rows> {
public:

	//constructors
	Vector<T, ml, rows>() {
		ml::initialize(this->array, this->size());
	}
	Vector<T, ml, rows>(T* ary) {
		this->array = ary;
	}

	//returns a row vector
	const Vector<T, ml, 1, rows> t() const {
		return Vector<T, ml, 1, rows>(this->array);
	}

	template<typename U>
	Vector<T, ml, rows> operator =(const Tensor_Queen<U, ml, rows>& tk) {
		ml::copy(this->data(), tk.data(), this->size());
		return *this;
	}

	Scalar<T, ml, true> operator[](int index) {
		return Scalar<T, ml, true>(&(this->array[index]));
	}
	const Scalar<T, ml, true> operator[](int index) const {
		return Scalar<T, ml, true>(this->array[index]);
	}
};

//Row Vector
template<class T, class ml, int rows>
class Vector<T, ml, 1, rows> : public Tensor_Jack<T, ml, 1, rows> {

public:
	Vector() {
		ml::initialize(this->array, this->size());
	}

	Vector(T* ary) {
		this->array = ary;
	}

	//returns a col vector
	const Vector<T, ml, 1, rows> t() const {
		return Vector<T, ml, rows>(this->array);
	}

};

template<class T, class ml>
class Vector<T, ml, 1> : public Scalar<T, ml, true> {
public:
	using Scalar<T, ml, true>::Scalar;

	template<class scal>
	Scalar<T, ml, true>& operator =(scal s) {
		this->Scalar<T, ml, true>::operator=(s);
		return *this;
	}
};

template<class T, class ml>
class Vector<T, ml, 1, 1> : public Scalar<T, ml, true> {
public:
	using Scalar<T, ml, true>::Scalar;

	template<class scal>
	Scalar<T, ml, true>& operator =(scal s) {
		this->Scalar<T, ml, true>::operator=(s);
		return *this;
	}
};

#endif /* BC_TENSOR_PRIMARY_VECTOR_H_ */
