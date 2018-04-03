/*
 * Blackcat_ForwardsDecls.h
 *
 *  Created on: Jan 9, 2018
 *      Author: joseph
 */

#ifndef BLACKCAT_FORWARDSDECLS_H_
#define BLACKCAT_FORWARDSDECLS_H_

namespace BC {

class CPU;
class GPU;

#ifndef BLACKCAT_DEFAULT_MATHLIB_GPU
using DEFAULT_MATHLIB = CPU;
#else
using DEFAULT_MATHLIB = GPU
#endif

//Uncomment this upon release,
//this is to ensure the backend-code mandates the correct Mathlibs
//template<class, class ML = DEFAULT_MATHLIB> class Scalar;
//template<class, class ML = DEFAULT_MATHLIB> class Vector;
//template<class, class ML = DEFAULT_MATHLIB> class Matrix;
//template<class, class ML = DEFAULT_MATHLIB> class Cube;
}


#endif /* BLACKCAT_FORWARDSDECLS_H_ */
