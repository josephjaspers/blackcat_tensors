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

//recommended for GPU writing in a CPU context
//#define BLACKCAT_PURELY_FUNCTIONAL -> This convert the operator= into a expresion based return type | this is useful if you want to create very long winded expressions
//Direct eval just iterates through while (eval) creates an iterated copy. Eval is for non-purely functional direct eval is for purely functional
//This lets you pass = operators as a function types (this is good for CUDA programming so you can avoid the overhead of a method


#ifndef BLACKCAT_DEFAULT_MATHLIB_GPU
using DEFAULT_MATHLIB = CPU;
#else
using DEFAULT_MATHLIB = GPU
#endif

template<class, class> struct Shape;
struct inner_shape;
struct outer_shape;

template<class, class ML = DEFAULT_MATHLIB> class Vector;
template<class, class ML = DEFAULT_MATHLIB> class RowVector;
template<class, class ML = DEFAULT_MATHLIB> class Matrix;
template<class, class ML = DEFAULT_MATHLIB> class Scalar;

template<class T, class deriv> class expession;
}



#endif /* BLACKCAT_FORWARDSDECLS_H_ */
