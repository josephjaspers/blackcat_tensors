/*
 * functional_layer_interface.h
 *
 *  Created on: Jun 26, 2018
 *      Author: joseph
 */

#ifndef FUNCTIONAL_LAYER_INTERFACE_H_
#define FUNCTIONAL_LAYER_INTERFACE_H_

namespace BC {
namespace NN {

template<class derived, class function_derived>
struct functional_layer {


	const auto& as_functional_layer() const { return static_cast<const function_derived&>(*this); }
		  auto& as_functional_layer() 		{ return static_cast<	   function_derived&>(*this); }

	const auto& as_derived() const { return static_cast<const derived&>(*this); }
		  auto& as_derived() 	   { return static_cast<	  derived&>(*this); }



};


}
}


#endif /* FUNCTIONAL_LAYER_INTERFACE_H_ */
