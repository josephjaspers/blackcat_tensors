/*
 * AlternateAsterixDenoter.h
 *
 *  Created on: May 5, 2018
 *      Author: joseph
 */

#ifndef ALTERNATEASTERIXDENOTER_H_
#define ALTERNATEASTERIXDENOTER_H_

namespace BC {
template<class> struct Tensor_Operations;

template<class A>
struct alternate_asterix_denoter {
	//This class is returned from the overloaded unary (*) operator, we use it to create a secondary subset of operators IE **, %*
	const Tensor_Operations<A>& ref;
	const Tensor_Operations<A>& operator() () const { return ref; }
	const Tensor_Operations<A>& get () const { return ref; }

	alternate_asterix_denoter(const Tensor_Operations<A>& r) : ref(const_cast<Tensor_Operations<A>&>(r)) {}
};


}



#endif /* ALTERNATEASTERIXDENOTER_H_ */
