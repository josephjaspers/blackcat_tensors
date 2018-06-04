/*
 * AlternateAsterixDenoter.h
 *
 *  Created on: May 5, 2018
 *      Author: joseph
 */

#ifndef ALTERNATEASTERIXDENOTER_H_
#define ALTERNATEASTERIXDENOTER_H_

namespace BC {
namespace Base {
	template<class> struct Tensor_Operations;
}
template<class> struct alternate_asterix_denoter;

template<class A>
struct unsafe_AAD {

	unsafe_AAD(const alternate_asterix_denoter<A>& ref_) : ref(ref_) {}
	const alternate_asterix_denoter<A>& ref;
	const Base::Tensor_Operations<A>& operator() () const { return ref.get(); }
	const Base::Tensor_Operations<A>& get () const { return ref.get(); }

};

template<class A>
struct alternate_asterix_denoter {
	//This class is returned from the overloaded unary (*) operator, we use it to create a secondary subset of operators IE **, %*
	const Base::Tensor_Operations<A>& ref;
	const Base::Tensor_Operations<A>& operator() () const { return ref; }
	const Base::Tensor_Operations<A>& get () const { return ref; }

	unsafe_AAD<A> operator * () const { return unsafe_AAD<A>(*this); }

	alternate_asterix_denoter(const Base::Tensor_Operations<A>& r) : ref(const_cast<Base::Tensor_Operations<A>&>(r)) {}
};
}

#endif /* ALTERNATEASTERIXDENOTER_H_ */
