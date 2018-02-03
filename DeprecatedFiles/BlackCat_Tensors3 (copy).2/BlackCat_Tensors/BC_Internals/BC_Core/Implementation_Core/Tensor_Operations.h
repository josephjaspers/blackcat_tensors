/*
 * Tensor_Core.h
 *
 *  Created on: Dec 30, 2017
 *      Author: joseph
 */

#ifndef TENSOR_CORE_H_
#define TENSOR_CORE_H_


//#include "../../BC_MetaTemplateFunctions/Adhoc.h"
#include "Tensor_Operations_impl.h"
namespace BC {

	/*
	 * Tensor Core is a syntactic sugar class
	 * All the ugly inheritance code goes here.
	 */

	template <class T, class derived, class lib>
	struct Tensor_Math_Core
			:
			  public Tensor_Operations_impl<
			  derived,
			  _TRAITS<
			  	  typename MTF::determine_scalar<T>::type,
			  	  typename MTF::determine_functor<T>::type,
			  	  typename MTF::determine_evaluation<derived>::type, lib>> {


			using parent_class = Tensor_Operations_impl<
					  derived,

					  _TRAITS<
					  	  typename MTF::determine_scalar<T>::type,
					  	  typename MTF::determine_functor<T>::type,
					  	  typename MTF::determine_evaluation<derived>::type, lib>>;

			using functor_type = typename parent_class::functor_type;

#ifdef BLACKCAT_PURELY_FUNCTIONAL
			using parent_class::operator=;
#endif

	};

}



#endif /* TENSOR_CORE_H_ */
