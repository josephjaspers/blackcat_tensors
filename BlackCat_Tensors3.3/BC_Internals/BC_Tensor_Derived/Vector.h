/*
 * Vector.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "BC_Tensor_Base/TensorBase.h"

namespace BC {
template<class T, class Mathlib>
class Vector : public TensorBase<Vector<T, Mathlib>> {

	using parent_class = TensorBase<Vector<T, Mathlib>>;

public:

	__BCinline__ static constexpr int DIMS() { return 1; }
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	Vector(const Vector&& t) : parent_class(std::move(t)) {}
	Vector(		 Vector&& t) : parent_class(std::move(t)) {}
	Vector(const Vector&  t) : parent_class(t) 	{}
	explicit Vector(int dim = 1) : parent_class(std::vector<int> {dim})  {}
	template<class U> 		  Vector(const Vector<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Vector(	   Vector<U, Mathlib>&& t) : parent_class(t) {}

	Vector& operator =(const Vector&  t) { return parent_class::operator=(t); }
	Vector& operator =(const Vector&& t) { return parent_class::operator=(std::move(t)); }
	Vector& operator =(	     Vector&& t) { return parent_class::operator=(std::move(t)); }
	template<class U>
	Vector& operator = (const Vector<U, Mathlib>& t) { return parent_class::operator=(t); }

	const Vector<unary_expression_transpose<typename parent_class::functor_type>, Mathlib> t() const {
		return Vector<unary_expression_transpose<typename parent_class::functor_type>, Mathlib>(this->data());
	}

private:

	template<class,class> friend class Vector;
	template<class> friend class TensorBase;
	template<class> friend class Tensor_Operations;
	template<class... params> Vector(const params&... p) : parent_class(p...) {}

};

} //End Namespace BCw

#endif /* VECTOR_H_ */

