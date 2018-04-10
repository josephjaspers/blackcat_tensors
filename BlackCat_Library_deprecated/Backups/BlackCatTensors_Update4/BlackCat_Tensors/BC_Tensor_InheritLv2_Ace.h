/*
 * BC_Tensor_Super_Definer.h
 *
 *  Created on: Dec 1, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_SUPER_DEFINYER_H_
#define BC_TENSOR_SUPER_DEFINER_H_

#include "BC_Tensor_InheritLv1_Shape.h"
#include "BC_Tensor_InheritLv1_FunctionType.h"
#include "BC_Tensor_SupplmentalLv1_Shape_impl.h"
#include "BC_Tensor_SupplmentalLv2_Identity.h"
#include "BC_Internals_Include.h"

/*
 * Types relevant characteristics of the Tensor .
 *
 * Defines eval() --> returns an evaluated T that always results in T being a -non class type
 * Defines data() --> returns the internal function type (either primitive array or expression class)
 * States eval<U_type> --> same as eval but instead adjusts the return type to some U type
 */

//template<class T, class ml, class dimensions>
//struct Tensor_Ace;

template<class T, class ml, int... dimensions>
struct Tensor_Ace :  public Shape<dimensions... >, public Tensor_FunctorType<T>
{

	using functor_type    = typename Tensor_FunctorType<T>::type;
	using identity_type   = typename BC_Shape_Identity::Identity<T, ml, dimensions...>::type;
	using evaluation_type = typename BC_Evaluation_Identity::Identity<T, ml, dimensions...>::type;
	using array_type 	  = typename BC_ArrayType::Identity<T>::type;

	functor_type array;
	
	template<class... params> Tensor_Ace<T, ml ,dimensions...> (	  params&... p) : array(p...) {};					//If it is an expression initialize with correct parameters
	template<class... params> Tensor_Ace<T, ml ,dimensions...> (const params&... p) : array(p...) {};					//If it is an expression initialize with correct parameters
							  Tensor_Ace<T, ml ,dimensions...> (				  ) { ml::initialize(array, this->size()); }			//Else just initialize the array

		  functor_type& data() 		 { return array; }
	const functor_type& data() const { return array; }


		  evaluation_type eval()	 	{ return static_cast<evaluation_type>(*this); }
	const evaluation_type eval() const	{ return static_cast<evaluation_type>(*this); }


	template<typename U>	   typename BC_Evaluation_Identity::Identity<U, ml, dimensions...>::type eval() 	  { return *this; }
	template<typename U> const typename BC_Evaluation_Identity::Identity<U, ml, dimensions...>::type eval() const { return *this; }

};

#endif /* BC_TENSOR_SUPER_DEFINER_H_ */
