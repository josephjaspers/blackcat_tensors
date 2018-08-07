/*
 * tensor_type_determiner.h
 *
 *  Created on: May 20, 2018
 *      Author: joseph
 *
 *
 *      meta-template functions for:
 *
 *      	 dimension_of 	- return the dimensionality of a tensor (number of 'sides')
 *      	 tensor_of 		- return the tensor_type based upon a dimensionality
 *			 isTensor		- return boolean if the type is a tensor
 */

#ifndef TENSOR_TYPE_DETERMINER_H_
#define TENSOR_TYPE_DETERMINER_H_

#ifndef BC_MAXIMIM_DIMENSIONS
#define BC_MAXIMIM_DIMENSIONS 8
#endif

namespace BC {

template<class,class> class Scalar;
template<class,class> class Vector;
template<class,class> class Matrix;
template<class,class> class Cube;
template<class,class> class Tensor4;
template<class,class> class Tensor5;
template<int> struct DISABLED;

//---------------------------------returns template-tensor type based on dimensionality----------------------------------//

template<int> struct tensor_of;
template<> struct tensor_of<0> { template<class t, class m> using type = Scalar<t,m>;  	 };
template<> struct tensor_of<1> { template<class t, class m> using type = Vector<t, m>; 	template<class t,class m> using slice = Scalar<t, m>; };
template<> struct tensor_of<2> { template<class t, class m> using type = Matrix<t, m>; 	template<class t,class m> using slice = Vector<t, m>; };
template<> struct tensor_of<3> { template<class t, class m> using type = Cube<t, m>;   	template<class t,class m> using slice = Matrix<t, m>; };
template<> struct tensor_of<4> { template<class t, class m> using type = Tensor4<t, m>;  template<class t,class m> using slice = Cube<t, m>; };
template<> struct tensor_of<5> { template<class t, class m> using type = Tensor5<t, m>;  template<class t,class m> using slice = Tensor4<t, m>; };

//shorthand
template<int x, class a, class b> using tensor_of_t = typename tensor_of<x>::template type<a,b>;

//---------------------------------iteratores through dimensions till it finds the correct type----------------------------------//
//---------------------------------based on tensor, returns the dimensionality of a tensor----------------------------------//

template<class> struct dimension_of_impl;
template<class a, class b> struct dimension_of_impl<Scalar<a,b>> { static constexpr int value = 0; using ml = b;};
template<class a, class b> struct dimension_of_impl<Vector<a,b>> { static constexpr int value = 1; using ml = b;};
template<class a, class b> struct dimension_of_impl<Matrix<a,b>> { static constexpr int value = 2; using ml = b;};
template<class a, class b> struct dimension_of_impl<Cube<a,b>>   { static constexpr int value = 3; using ml = b;};
template<class a, class b> struct dimension_of_impl<Tensor4<a,b>>   { static constexpr int value = 4; using ml = b;};
template<class a, class b> struct dimension_of_impl<Tensor5<a,b>>   { static constexpr int value = 5; using ml = b;};
//template<class a, class b, template<class,class> class tensor> struct dimension_of_impl<tensor<a, b>>
//{ static constexpr int value = determine_variable_tensor<tensor<a,b>>::value; using ml = b;};

//shorthand
template<class T> static constexpr int dimension_of = dimension_of_impl<T>::value;

//---------------------------------determine if the type is a BC_tensor------------------------------------------------------//
template<class T> struct is_tensor_impl { static constexpr bool conditional =false; };
template<class a, class b> struct is_tensor_impl<Scalar<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct is_tensor_impl<Vector<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct is_tensor_impl<Matrix<a,b>> { static constexpr bool conditional = true; };
template<class a, class b> struct is_tensor_impl<Cube<a,b>>   { static constexpr bool conditional = true; };
template<class a, class b> struct is_tensor_impl<Tensor4<a,b>>   { static constexpr bool conditional = true; };
template<class a, class b> struct is_tensor_impl<Tensor5<a,b>>   { static constexpr bool conditional = true; };

//shorthand
template<class T> static constexpr bool is_tensor = is_tensor_impl<T>::conditional;







}


#endif /* TENSOR_TYPE_DETERMINER_H_ */
