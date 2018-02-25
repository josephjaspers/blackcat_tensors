/*
 * Determiners.h
 *
 *  Created on: Feb 25, 2018
 *      Author: joseph
 */

#ifndef DETERMINERS_H_
#define DETERMINERS_H_

namespace BC {

//DETERMINE IF ITS A CORE_TYPE (DETERMINE IF THE METHOD SHOULD GAIN ACCESS TO UTILITY METHODS)-------------------------------------

template<class,class,class> struct Tensor_Core;
template<class> struct Tensor_Slice;

template<class > 					struct isCore 						{ static constexpr bool conditional = false; };
template<class a, class b, class c> struct isCore<Tensor_Core<a, b, c>> { static constexpr bool conditional = true;  };
template<class a> struct isCore<Tensor_Slice<a>> { static constexpr bool conditional = true;  };

template<class T> constexpr bool is_core = isCore<T>::conditional;





template<class,class> class DISABLED;
template<class,class> class Scalar;
template<class,class> class Vector;
template<class,class> class Matrix;
template<class,class> class Cube;

template<int> struct base;

template<> struct base<0> { template<class t, class m> using type = Scalar<t,m>;  template<class t, class m> using slice = DISABLED<t, m>; };
template<> struct base<1> { template<class t, class m> using type = Vector<t, m>; template<class t,class m> using slice = Scalar<t, m>; };
template<> struct base<2> { template<class t, class m> using type = Matrix<t, m>; template<class t,class m> using slice = Vector<t, m>; };
template<> struct base<3> { template<class t, class m> using type = Cube<t, m>;   template<class t,class m> using slice = Matrix<t, m>; };

template<int,int, class> struct Rank;
template<class> struct ranker;
template<class a, class b> struct ranker<Scalar<a,b>> { static constexpr int value = 0; using type = Rank<value, value>; };
template<class a, class b> struct ranker<Vector<a,b>> { static constexpr int value = 1; using type = Rank<value, value>;  };
template<class a, class b> struct ranker<Matrix<a,b>> { static constexpr int value = 2; using type = Rank<value, value>;  };
template<class a, class b> struct ranker<Cube<a,b>>   { static constexpr int value = 3; using type = Rank<value, value>;  };

template<class T> using _scalar = typename MTF::determine_scalar<T>::type;
template<class T> using _mathlib = typename MTF::determine_mathlibrary<T>::type;
template<class T> using _ranker  = typename ranker<T>::type;

template<class T, class voider>  struct determine_functor_type { using type = T; };		//non primitive (if class) functor = itself
template<class T> struct determine_functor_type<T, std::enable_if_t<MTF::isPrimitive<T>::conditional>>
{ using type = Tensor_Core<_scalar<T>, _mathlib<T>, _ranker<T>>; };	//if the class is a primitive type == convert the functor_type = TensorCore



}



#endif /* DETERMINERS_H_ */
