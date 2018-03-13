/*
 * Vector.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef VECTOR_H_
#define VECTOR_H_

#include "Scalar.h"
#include "TensorBase.h"

namespace BC {
template<class T, class Mathlib>
class Vector : public TensorBase<Vector<T, Mathlib>> {

	template<class,class> friend class Vector;
	using parent_class = TensorBase<Vector<T, Mathlib>>;

public:

	static constexpr int RANK() { return 1; }
	using parent_class::operator=;
	using parent_class::operator[];
	using parent_class::operator();

	Vector(const Vector&& t) : parent_class(t) 		{}
	Vector(		 Vector&& t) : parent_class(t) 		{}
	Vector(const Vector&  t) : parent_class(t) 		{}
	explicit Vector(int dim) 		 : parent_class(std::vector<int> {dim})  {}

	template<class U> 		  Vector(const Vector<U, Mathlib>&  t) : parent_class(t) {}
	template<class U> 		  Vector(	   Vector<U, Mathlib>&& t) : parent_class(t) {}
	template<class... params> Vector(const params&... p) : parent_class(p...) {}

	Vector& operator =(const Vector&  t) { return parent_class::operator=(t); }
	Vector& operator =(const Vector&& t) { return parent_class::operator=(t); }
	Vector& operator =(	     Vector&& t) { return parent_class::operator=(t); }
	template<class U>
	Vector& operator = (const Vector<U, Mathlib>& t) { return parent_class::operator=(t); }

	const Vector<unary_expression_transpose<typename MTF::determine_scalar<T>::type, typename parent_class::functor_type>, Mathlib> t() const {
		return Vector<unary_expression_transpose<typename MTF::determine_scalar<T>::type, typename parent_class::functor_type>, Mathlib>
		(this->data());
	}
};

} //End Namespace BCw

#endif /* VECTOR_H_ */
