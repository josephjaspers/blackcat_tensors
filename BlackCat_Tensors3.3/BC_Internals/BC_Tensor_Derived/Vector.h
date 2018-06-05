/*
 * Vector.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "BC_Tensor_Base/Tensor_Base.h"

namespace BC {
template<class T, class Mathlib>
class Vector : public Tensor_Base<Vector<T, Mathlib>> {

	using parent_class = Tensor_Base<Vector<T, Mathlib>>;

public:

	__BCinline__ static constexpr int DIMS() { return 1; }
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	Vector(const Vector&& t) : parent_class(std::move(t)) {}
	Vector(		 Vector&& t) : parent_class(std::move(t)) {}
	Vector(const Vector&  t) : parent_class(t) 	{}
	explicit Vector(int dim = 0) : parent_class(Shape<1>(dim))  {}
	explicit Vector(Shape<DIMS()> shape) : parent_class(shape)  {}
	template<class... params> Vector(const params&... p) : parent_class(p...) {}

	template<class U> 		  Vector(const Vector<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Vector(	   Vector<U, Mathlib>&& t) : parent_class(t) {}

	Vector& operator =(const Vector&  t) { return parent_class::operator=(t); }
	Vector& operator =(const Vector&& t) { return parent_class::operator=(std::move(t)); }
	Vector& operator =(	     Vector&& t) { return parent_class::operator=(std::move(t)); }
	template<class U>
	Vector& operator = (const Vector<U, Mathlib>& t) { return parent_class::operator=(t); }

	const Vector<internal::unary_expression<typename parent_class::functor_type, function::transpose>, Mathlib> t() const {
		return Vector<internal::unary_expression<typename parent_class::functor_type, function::transpose>, Mathlib>(this->internal());
	}
};
}

#endif /* VECTOR_H_ */

