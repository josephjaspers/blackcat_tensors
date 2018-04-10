/*
 * BC_Internal_ForwardDeclerations.h
 *
 *  Created on: Dec 18, 2017
 *      Author: joseph
 */

#ifndef BLACKCAT_INTERNAL_FORWARDDECLS_H_
#define BLACKCAT_INTERNAL_FORWARDDECLS_H_

namespace BC {


/*
 * Forward dec of the core classes
 */
class CPU;
template<class> class DEFAULT_LD;
template<int...>class Inner_Shape;

#ifndef BLACKCAT_TENSORS_DEFAULT_LIB_GPU
typedef  CPU DEFAULT_LIBRARY;
#else
struct GPU;
typedef GPU DEFAULT_LIBRARY;
#endif

template<class T, class ml = DEFAULT_LIBRARY>
class Scalar;

template<
	class T,
	int rows = 0,
	class math_library = DEFAULT_LIBRARY,
	class outer_shape = typename DEFAULT_LD<Inner_Shape<rows>>::type>
class Vector;

template<
	class T,
	int rows = 0,
	class math_library = DEFAULT_LIBRARY,
	class outer_shape = typename DEFAULT_LD<Inner_Shape<1, rows>>::type>
class RowVector;


template<
	class T,
	int rows = 0, int cols = 0,
	class math_library = DEFAULT_LIBRARY,
	class outer_shape  =  typename DEFAULT_LD<Inner_Shape<rows, cols>>::type
>
class Matrix;

template<
	class T,
	int rows = 0, int cols = 0, int depth = 0,
	class math_library = DEFAULT_LIBRARY,
	class outer_shape  =  typename DEFAULT_LD<Inner_Shape<rows, cols, depth>>::type
>
class Cube;

template<
	class T,
	class Shape,
	class math_library = CPU,
	class outer_shape =  typename DEFAULT_LD<Shape>::type
>
class Tensor;

/*
 * Forward dec of auxillary classes
 */

template<class T, class deriv>
struct expression;

template<class T, class oper>
struct unary_expression;

template<class T, class oper, class lv, class rv>
struct binary_expression;

class VOID_CLASS {};
}

#include "BlackCat_Internal_GlobalUnifier.h"


#endif /* BLACKCAT_INTERNAL_FORWARDDECLS_H_ */
