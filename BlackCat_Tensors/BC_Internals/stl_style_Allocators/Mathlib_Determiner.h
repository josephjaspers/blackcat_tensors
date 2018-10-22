/*

 * Mathlib_Determiner.h
 *
 *  Created on: Oct 22, 2018
 *      Author: joseph
 */

#ifndef MATHLIB_DETERMINER_H_
#define MATHLIB_DETERMINER_H_

namespace BC {
class BC::CPU;
class BC::GPU;

namespace module {
namespace stl {

template<class mathlib_t, class enabler=void>
struct mathlib_determiner {
	using type = BC::CPU;
};

template<class mathlib_t>
struct mathlib_determiner<mathlib_t,
	std::enable_if_t<!std::is_same<void, typename mathlib_t::mathlib_t>::value>> {
	using type = typename mathlib_t::mathlib_t;
};

template<class alloc_t>
using alloc_based_mathlib = typename mathlib_determiner<alloc_t>::type;


}
}
}






#endif /* MATHLIB_DETERMINER_H_ */
