/*
 * Cube.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_Cube_H
#define BC_Cube_H

#include "Vector.h"
#include "TensorBase.h"


namespace BC {
template<class T, class Mathlib>
class Cube : public TensorBase<T, Cube<T, Mathlib>, Mathlib, Rank<3>> {

	template<class,class>
	friend class Matrix;

	using parent_class = TensorBase<T, Cube<T, Mathlib>, Mathlib, Rank<3>>;

public:
	using scalar = T;
	using parent_class::operator=;
	using child = typename parent_class::child;
	static constexpr int RANK() { return 3; }

	Cube(const Cube&  v) : parent_class(v) {}
	Cube(		 Cube&& v) : parent_class(v) {}
	Cube(const Cube&& v) : parent_class(v) {}
	Cube(int rows, int cols = 1, int pages = 1) : parent_class(std::vector<int> {rows, cols, pages}) {}

	template<class U> 		  Cube(const Cube<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Cube(	   Cube<U, Mathlib>&& t) : parent_class(t) {}
	template<class... params> Cube(const params&... p) : parent_class(p...) {}

	Cube& operator =(const Cube& t)  { return parent_class::operator=(t); }
	Cube& operator =(const Cube&& t) { return parent_class::operator=(t); }
	Cube& operator =(	     Cube&& t) { return parent_class::operator=(t); }
	template<class U>
	Cube& operator = (const Cube<U, Mathlib>& t) { return parent_class::operator=(t); }

//	Matrix<accessor, Mathlib> operator [] (int index) { return Vector<accessor, Mathlib>(this->accessor_packet(index)); }
//	const Matrix<accessor, Mathlib> operator [] (int index) const { return Vector<accessor, Mathlib>(this->accessor_packet(index)); }

	auto& operator[] (int i) const { return this->data()[i];}
	auto& operator[] (int i)  { return this->data()[i];}

	const Cube<unary_expression_transpose<typename MTF::determine_scalar<T>::type, typename parent_class::functor_type>, Mathlib> t() const {
		return Cube<unary_expression_transpose<typename MTF::determine_scalar<T>::type, typename parent_class::functor_type>, Mathlib>(this->data());
	}



};

} //End Namespace BC

#endif /* Cube_H */
