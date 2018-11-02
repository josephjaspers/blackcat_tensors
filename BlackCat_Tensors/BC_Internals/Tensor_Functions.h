/*  Project: BlackCat_Tensors
 *  Author: JosephJaspers
 *  Copyright 2018
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */


#ifndef TENSOR_FUNCTIONS_H_
#define TENSOR_FUNCTIONS_H_

#include <algorithm>

namespace BC{
class CPU;
class GPU;
template<class internal> class Tensor_Base;

namespace module {
template<class derived> class Tensor_Functions;

enum functional_library {
	std = 0,
	thrust = 1
};

template<class internal_t>
class Tensor_Functions<Tensor_Base<internal_t>> {
	template<class> friend class Tensor_Functions;

	using derived	    = Tensor_Base<internal_t>;
	using scalar_t 		= typename internal_t::scalar_t;
	using allocator_t   = typename internal_t::allocator_t;

	 const derived& as_derived() const { return static_cast<const derived&>(*this);  }
	 	   derived& as_derived() 	   { return static_cast<	  derived&>(*this); }


public:

//-------------------------in-place functions-------------------------//
	//TODO enable support for multi-dimensional randomize

	void fill(scalar_t value)   { as_derived() = value; }
	void zero()                 { as_derived() = scalar_t(0); }
	void ones()					{ as_derived() = scalar_t(1); }

   	void randomize(scalar_t lb=0, scalar_t ub=1)  {
   		static_assert(internal_t::ITERATOR() == 0 || internal_t::ITERATOR() == 1,
   				"randomize not available to non-continuous tensors");
   		allocator_t::randomize(this->as_derived().internal(), lb, ub);
   	}

}; //end_of class 'Tensor_Functions'


}  //end_of namespace 'module'



//template<class internal, class T=void> using enable_if_thrust = std::enable_if_t<false and std::is_same<GPU, typename internal::mathlib_t>::value, T>;
//template<class internal, class T=void> using enable_if_std    = std::enable_if_t<std::is_same<CPU, typename internal::mathlib_t>::value, T>;
//
////TODO figure out a way to convert thrust and std namespace into template parameter
//
//#define BC_FUNCTION_STRUCT_FRWD_DEF(function) \
//template<class tensor_t, class enabler=void> struct functor_##function;\
//
//#define BC_FUNCTION_STRUCT_DEF(function, impl)\
//template<class tensor_t> 													\
//struct functor_##function <tensor_t, enable_if_##impl <tensor_t>> {					\
//	template<class iter_begin, class iter_end, class arg>					\
//	static auto run (iter_begin begin, iter_end end, arg param) {			\
//		return impl::function(begin, end, param);								\
//	}																		\
//};
//#define BC_FUNCTION_DEF(function) \
//																		\
//		BC_FUNCTION_STRUCT_DEF(function, std)																	\
//\
//template<class functor, class internal_t>																		\
//auto function (const Tensor_Base<internal_t>& tensor, functor func) {											\
//	return functor_##function <Tensor_Base<internal_t>>::run(tensor.cbegin(), tensor.cend(), func);				\
//}																												\
//template<class functor, class internal_t>																		\
//auto function (Tensor_Base<internal_t>& tensor, functor func) {													\
//	return functor_##function <Tensor_Base<internal_t>>::run(tensor.cbegin(), tensor.cend(), func);				\
//}
//
////#define BC_FUNCTION_DEF(function, impl)	\
////template<class iter_begin, class iter_end, class functor>														\
////struct functor_##function {\
////	static enable_if_##impl <typename iter_begin::tensor_t,																\
////	decltype( impl :: function (std::declval<iter_begin>(), std::declval<iter_end>(), std::declval<functor>()))>	\
////	function (iter_begin begin, iter_end end, functor func) {														\
////		using tensor_t = typename iter_begin::tensor_t;																\
////																													\
////		return  impl :: function (begin, end, func);																	\
////	}				\
////};																								\
////template<class functor, class internal_t>																		\
////auto function (const Tensor_Base<internal_t>& tensor, functor func) {											\
////	return function (tensor.cbegin(), tensor.cend(), func);														\
////}																												\
////template<class functor, class internal_t>										\
////auto function (Tensor_Base<internal_t>& tensor, functor func) {													\
////	return function (tensor.begin(), tensor.end(), func);														\
////}																												\
//BC_FUNCTION_DEF( all_of , std)
//BC_FUNCTION_STRUCT_FRWD_DEF( all_of )
//BC_FUNCTION_STRUCT_FRWD_DEF( any_of )
//BC_FUNCTION_STRUCT_FRWD_DEF( none_of)
//BC_FUNCTION_STRUCT_FRWD_DEF( find )
//BC_FUNCTION_STRUCT_FRWD_DEF( find_if )
//BC_FUNCTION_STRUCT_FRWD_DEF( find_if_not )
//BC_FUNCTION_STRUCT_FRWD_DEF( find_end )
//BC_FUNCTION_STRUCT_FRWD_DEF( find_first_of)
//BC_FUNCTION_STRUCT_FRWD_DEF( for_each  )
////
////
//BC_FUNCTION_DEF( all_of )
//BC_FUNCTION_DEF( any_of )
//BC_FUNCTION_DEF( none_of )
//BC_FUNCTION_DEF( find )
//BC_FUNCTION_DEF( find_if )
//BC_FUNCTION_DEF( find_if_not )
//BC_FUNCTION_DEF( find_end )
//BC_FUNCTION_DEF( find_first_of )
//BC_FUNCTION_DEF( for_each )
//////
////#ifdef __CUDACC__
////BC_FUNCTION_DEF( all_of , thrust)
////
////BC_FUNCTION_DEF( any_of , thrust)
////BC_FUNCTION_DEF( none_of , thrust)
////BC_FUNCTION_DEF( find , thrust)
////BC_FUNCTION_DEF( find_if , thrust)
////BC_FUNCTION_DEF( find_if_not , thrust)
////BC_FUNCTION_DEF( find_end , thrust)
////BC_FUNCTION_DEF( find_first_of , thrust)
////BC_FUNCTION_DEF( for_each , thrust)
////#endif
////
//

//--------------------------lazy functions-------------------------//
//TODO Add more loss functions, move loss functions to seperate module

template<class scalar>
struct norm {
	scalar min;
	scalar max;

	norm(scalar min_, scalar max_) : min(min_), max(max_) {}

	__BCinline__ auto operator () (scalar v) const {
		return (v - min) / (max - min);
	}
};

template<class internal_t, class min_, class max_>
static auto normalize(const Tensor_Base<internal_t>& tensor, min_ min, max_ max) {
	using scalar_t = typename internal_t::scalar_t;
	return tensor.un_expr(norm<scalar_t>(scalar_t(min), scalar_t(max)));
}

}






#endif /* TENSOR_FUNCTIONS_H_ */
