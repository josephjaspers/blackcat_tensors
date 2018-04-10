/*
 * BC_Tensor_Base.h
 *
 *  Created on: Dec 11, 2017
 *      Author: joseph
 */

#ifndef BC_TENSOR_BASE_H_
#define BC_TENSOR_BASE_H_

#include "../BlackCat_Internal_GlobalUnifier.h"
namespace BC {
	/*
	 * Some notes:
	 * 		array is the only field of this class for static-shaped tensors.
	 * 		For dynamic tensors, the only fields are the array, inner_shape (int[Rank + 1]) and its outer_shape (int[Rank + 1])
	 *
	 * 		Static Tensors are faster as they do not require any runtime checks (assuming used only with other static tensors).
	 */


template <
class array_T,									//The internal array type
class derived,									//The derived class
class math_lib,					    		    //Math_library
class shape,									//The shape of the tensor
bool  isParent = BC_MTF::is_same<typename shape::inner_shape, DEFAULT_LD<typename shape::inner_shape>>::conditional
>
struct Tensor_Type : shape {

	/*
	 * This is the specialization for primitive-data types and array_types/array_wrappers
	 */

	using this_type       = Tensor_Type<array_T, derived, math_lib, shape, isParent>;
	using functor_type    = typename Tensor_FunctorType<array_T>::type;			//either T* if primitive else an array wrapper
	using scalar_type 	  = typename BC_ArrayType::Identity<array_T>::type;		//T type is different if expression type
	using identity_type   = typename BC_Shape_Identity::Identity<array_T, math_lib, typename shape::inner_shape>::type; //The derived class of this with DEFAULT_LD
	using evaluation_type = this_type&; //typename BC_Evaluation_Identity::Identity<array_T, math_lib, typename shape::inner_shape>::type;
	using math_library 	  = math_lib;											//The math library generally either the default CPU lib or a CUDA lib (Cuda lib not yet written)

	static constexpr bool ASSIGNABLE = true;
	static constexpr bool MOVEABLE   = isParent;
	static constexpr Tensor_Shape  RANK =  derived::RANK;

	functor_type array; 														//This is the internal array

	Tensor_Type()  { math_lib::initialize(array, this->size());} 				//Assumes array is a pointer to a primitive type, initializes with the apropriate math_lib call
	template<class... params> Tensor_Type(const params&... p)  : array(p...) {}

	~Tensor_Type() { if (isParent) math_lib::destroy(array); }


		  evaluation_type eval() 	   { return static_cast<evaluation_type>(static_cast<derived&>(*this)); } //yes you need to double cast
	const evaluation_type eval() const { return static_cast<evaluation_type>(static_cast<derived&>(*this)); } //sorry its awkward

		  functor_type    data()	   { return array; }
	const functor_type	  data() const { return array; }

	constexpr bool isAssignable() const { return true; }
	constexpr bool isMoveable()   const { return isParent; }

	bool ensureAssignability() const { if (!isAssignable())  throw std::invalid_argument("non assignable"); return isMoveable();}
	bool ensureMoveability()   const { if (!isMoveable())  throw std::invalid_argument("non moveable"); return isAssignable();}
};


/*
 * This specialization is for Tensors that are not parents of their internal array AND are not array_wrappers (IE expressions)
 */

template <
class array_T,									//The internal array type
class derived,									//The derived class
class math_lib,					    		    //Math_library
class shape									   //The shape of the tensor
>
struct Tensor_Type<array_T, derived, math_lib, shape, false> : shape {

	using this_type       = Tensor_Type<array_T, derived, math_lib, shape, false>;
	using functor_type    = typename Tensor_FunctorType<array_T>::type;			//either T* if primitive else an array wrapper
	using scalar_type 	  = typename BC_ArrayType::Identity<array_T>::type;
	using identity_type   = typename BC_Shape_Identity::Identity<array_T, math_lib, typename shape::inner_shape>::type;
	using evaluation_type = typename BC_Evaluation_Identity::Identity<array_T, math_lib, typename shape::inner_shape>::type;
	using math_library 	  = math_lib;											//The math library generally either the default CPU lib or a CUDA lib (Cuda lib not yet written)

	static constexpr bool ASSIGNABLE = Tensor_FunctorType<array_T>::supports_utility_functions;
	static constexpr bool MOVEABLE   = false;
	static constexpr Tensor_Shape  RANK =  derived::RANK;


	functor_type array; 														//This is the internal array

	template<class... params> Tensor_Type(const params&... p) : array(p...) {}				//Assumes array is a pointer to a primitive type, initializes with the appropriate math_lib call
	~Tensor_Type() = default;


		  evaluation_type eval() 	   { return static_cast<evaluation_type>(static_cast<derived&>(*this)); } //yes you need to double cast
	const evaluation_type eval() const { return static_cast<evaluation_type>(static_cast<derived&>(*this)); } //sorry its awkward

		  functor_type&  data()	   	  { return array; }
	const functor_type&	 data() const { return array; }


	constexpr bool isAssignable() const { return false; }
	constexpr bool isMoveable()   const { return false; }

	bool ensureAssignability() const { if (!isAssignable())  throw std::invalid_argument("non assignable"); return isMoveable();}
	bool ensureMoveability()   const { if (!isMoveable())    throw std::invalid_argument("non moveable");   return isAssignable();}
};


/*
 * SPECIALIZATION FOR DYNAMIC TENSORS  SPECIALIZATION FOR DYNAMIC TENSORS  SPECIALIZATION FOR DYNAMIC TENSORS  SPECIALIZATION FOR DYNAMIC TENSORS  SPECIALIZATION FOR DYNAMIC TENSORS
 */





template <
class array_T,									//The internal array type
class derived,									//The derived class
class math_lib,					    		    //Math_library
class LD,
int... arbitrary
>																																		///chk here
struct Tensor_Type<array_T, derived, math_lib, Shape<Static_Inner_Shape<0,arbitrary...>, LD>, true>
	: Dynamic_Shape<
	  	  Dynamic_Inner_Shape<BC_Derived_Rank::Identity<derived>::value>,
		  Dynamic_Outer_Shape<BC_Derived_Rank::Identity<derived>::value>,
							  BC_Derived_Rank::Identity<derived>::value>
{

	using is = Dynamic_Inner_Shape<BC_Derived_Rank::Identity<derived>::value>;  //chk
	using os = LD;

	using this_type       = Tensor_Type<array_T, derived, math_lib, Dynamic_Shape<is, os, 1>, true>; ///chk
	using functor_type    = typename Tensor_FunctorType<array_T>::type;			//either T* if primitive else an array wrapper
	using scalar_type 	  = typename BC_ArrayType::Identity<array_T>::type;		//T type is different if expression type

	using parent_class =  Dynamic_Shape<Dynamic_Inner_Shape<1>, Dynamic_Outer_Shape<1>, 1>;
	static constexpr Tensor_Shape  RANK =  derived::RANK;

	//find type of dynamic
	using evaluation_type = this_type&; //typename BC_Evaluation_Identity::Identity<array_T, math_lib, typename shape::inner_shape>::type;
	using math_library 	  = math_lib;											//The math library generally either the default CPU lib or a CUDA lib (Cuda lib not yet written)

	static constexpr bool ASSIGNABLE = true;
	static constexpr bool MOVEABLE   = true; ///chk

	functor_type array; 														//This is the internal array

	template<class... params>
	Tensor_Type(const params&... p)  : array(p...) {}
	Tensor_Type() = default;

protected:
	void DYNAMIC_ARRAY_INITIALIZE() { math_lib::initialize(array, this->size()); }
	~Tensor_Type() { math_lib::destroy(array); }
public:

		  evaluation_type eval() 	   { return static_cast<evaluation_type>(static_cast<derived&>(*this)); } //yes you need to double cast
	const evaluation_type eval() const { return static_cast<evaluation_type>(static_cast<derived&>(*this)); } //sorry its awkward

		  functor_type    data()	   { return array; }
	const functor_type	  data() const { return array; }

	constexpr bool isAssignable() const { return true; }
	constexpr bool isMoveable()   const { return true; } //// chk

	bool ensureAssignability() const { if (!isAssignable())  throw std::invalid_argument("non assignable"); return isMoveable();}
	bool ensureMoveability()   const { if (!isMoveable())  throw std::invalid_argument("non moveable"); return isAssignable();}
};
}
#endif /* BC_TENSOR_BASE_H_ */
