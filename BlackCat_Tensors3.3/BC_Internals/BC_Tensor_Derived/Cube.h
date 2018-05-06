/*

 * Cube.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef BC_Cube_H
#define BC_Cube_H

#include "BC_Tensor_Base/TensorBase.h"


namespace BC {
template<class T, class Mathlib>
class Cube : public Tensor<Cube<T, Mathlib>> {

	using parent_class = Tensor<Cube<T, Mathlib>>;

public:
	using scalar = T;
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	__BCinline__ static constexpr int DIMS() { return 3; }

	Cube(const Cube&  v) : parent_class(v) {}
	Cube(	   Cube&& v) : parent_class(v) {}
	Cube(const Cube&& v) : parent_class(v) {}
	explicit Cube(int rows = 1, int cols = 1, int pages = 1) : parent_class(array(rows, cols, pages)) {}

	template<class U> 		  Cube(const Cube<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Cube(	     Cube<U, Mathlib>&& t) : parent_class(t) {}

	Cube& operator =(const Cube& t)  { return parent_class::operator=(t); }
	Cube& operator =(const Cube&& t) { return parent_class::operator=(t); }
	Cube& operator =(	   Cube&& t) { return parent_class::operator=(t); }
	template<class U>
	Cube& operator = (const Cube<U, Mathlib>& t) { return parent_class::operator=(t); }

private:

	template<class> friend class Tensor;
	template<class> friend class Tensor_Operations;
	template<class... params> Cube(const params&... p) : parent_class(p...) {}

};

} //End Namespace BC

#endif /* Cube_H */
