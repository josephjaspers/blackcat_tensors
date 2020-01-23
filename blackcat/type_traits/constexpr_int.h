/*
 * Int.h
 *
 *  Created on: Jan 22, 2020
 *      Author: joseph
 */

#ifndef BC_TYPE_TRAITS_INT_H_
#define BC_TYPE_TRAITS_INT_H_

namespace bc {
namespace traits {

template<int X> struct Integer {

	static constexpr int value = X;

#define BC_INTEGER_OP(op)                               \
                                                        \
    template<int Y>                                     \
    auto operator op (const Integer<Y>& other) const {  \
    	return Integer<X op Y>();                       \
    }

	BC_INTEGER_OP(+)
	BC_INTEGER_OP(-)
	BC_INTEGER_OP(/)
	BC_INTEGER_OP(*)

#undef BC_INTEGER_OP

};

}
}



#endif /* INT_H_ */

