/*
 * Core_Vectorize.h
 *
 *  Created on: May 29, 2018
 *      Author: joseph
 */

#ifndef CORE_VECTORIZE_H_
#define CORE_VECTORIZE_H_
#include "Core_Base.h"

namespace BC {
template<class PARENT>
	struct Tensor_Vectorizer : Tensor_Core_Base<Tensor_Vectorizer<PARENT>, 1> {

	using scalar_type = _scalar<PARENT>;

	__BCinline__ static constexpr int ITERATOR() { return max(PARENT::ITERATOR() - 1, 0); }
	__BCinline__ static constexpr int DIMS() { return 1; }

	operator const PARENT() const	{ return parent; }

	const PARENT parent;
	scalar_type* array_slice;

	__BCinline__ Tensor_Vectorizer(const scalar_type* array, PARENT parent_) : array_slice(const_cast<scalar_type*>(array)), parent(parent_) {}
	__BCinline__ const auto innerShape() const 			{ return &parent.inner_shape()[PARENT::DIMS() - 1]; }
	__BCinline__ const auto outerShape() const 			{ return &parent.inner_shape()[PARENT::DIMS() - 1]; }
	__BCinline__ const auto& operator [] (int i) const { return array_slice[parent(this->index_to_dims(i))]; };
	__BCinline__  auto& operator [] (int i)  { return array_slice[parent(this->index_to_dims(i))]; };

	__BCinline__ const scalar_type* getIterator() const { return array_slice; }
	__BCinline__	   scalar_type* getIterator()   	{ return array_slice; }

	};
}



#endif /* CORE_VECTORIZE_H_ */
