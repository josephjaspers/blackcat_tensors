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
class Cube : public TensorBase<Cube<T, Mathlib>> {

	template<class,class>
	friend class Matrix;

	using parent_class = TensorBase<Cube<T, Mathlib>>;

public:
	using scalar = T;
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	static constexpr int DIMS() { return 3; }

	Cube(const Cube&  v) : parent_class(v) {}
	Cube(	   Cube&& v) : parent_class(v) {}
	Cube(const Cube&& v) : parent_class(v) {}
	explicit Cube(int rows, int cols = 1, int pages = 1) : parent_class(std::vector<int> {rows, cols, pages}) {}

	template<class U> 		  Cube(const Cube<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Cube(	   Cube<U, Mathlib>&& t) : parent_class(t) {}
	template<class... params> Cube(const params&... p) : parent_class(p...) {}

	Cube& operator =(const Cube& t)  { return parent_class::operator=(t); }
	Cube& operator =(const Cube&& t) { return parent_class::operator=(t); }
	Cube& operator =(	   Cube&& t) { return parent_class::operator=(t); }
	template<class U>
	Cube& operator = (const Cube<U, Mathlib>& t) { return parent_class::operator=(t); }
};

} //End Namespace BC

#endif /* Cube_H */
